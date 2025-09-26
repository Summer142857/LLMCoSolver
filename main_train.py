import argparse
import torch
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
import re
from utils import *
import os
import ast
#os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
from utils import calculate_total_distance, compute_euclidean_distance_matrix, calculate_pfsp_makespan
import warnings
import numpy as np

class SafeDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    A safer version of DataCollatorForCompletionOnlyLM that handles cases where
    the response template is not found in the input.
   """
    def __init__(self, response_template, tokenizer, mlm=False, fallback_strategy="last_portion"):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        self.tokenizer = tokenizer
        
        # Tokenize the template once to improve efficiency
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.fallback_strategy = fallback_strategy
        
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        
        # Process each example in the batch
        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i].tolist()
            
            # Try to find the template in the input
            template_found = False
            for idx in range(len(input_ids) - len(self.response_template_ids) + 1):
                if input_ids[idx:idx+len(self.response_template_ids)] == self.response_template_ids:
                    template_found = True
                    # Adjust labels to only include tokens after the template
                    response_start_idx = idx + len(self.response_template_ids)
                    batch["labels"][i, :response_start_idx] = -100
                    break
            
            # Handle case where template is not found
            if not template_found:
                warnings.warn(f"Response template not found in example {i}")
                
                if self.fallback_strategy == "last_portion":
                    # Use the last 10% of tokens for loss computation
                    seq_length = len(input_ids)
                    start_pos = int(0.9 * seq_length)
                    # Only compute loss on the last portion
                    batch["labels"][i, :start_pos] = -100
                    # Keep labels for the remaining portion
                
                elif self.fallback_strategy == "full_example":
                    # Use the entire example as-is (fallback to regular LM training)
                    pass  # Labels are already set to input_ids by the parent collator
                
                elif self.fallback_strategy == "skip":
                    # Skip this example completely by setting all labels to -100
                    batch["labels"][i, :] = -100
        
        return batch

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trainer for solving combinatorial optimization problems')

    # Model and data parameters
    parser.add_argument('--max_seq_length', type=int, default=20000, help='Maximum sequence length')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],
                        help='Data type (bfloat16 or float16)')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                        help='Use 4-bit quantization to reduce memory usage')
    # Model name
    parser.add_argument('--model_name', type=str, default='unsloth/Qwen2.5-7B', help='Model name')
    # Problem name
    parser.add_argument('--problem', type=str, default='jssp', help='Problem name')

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=64, help='Rank of the LoRA decomposition')
    parser.add_argument('--lora_alpha', type=int, default=64, help='Scaling factor for LoRA updates')
    parser.add_argument('--bias', type=str, default='lora_only', choices=['none', 'all', 'lora_only'], help='Bias type')

    # Additional configurations
    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth', help='Use gradient checkpointing')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--use_rslora', action='store_true', default=False, help='Use RSLoRA')
    parser.add_argument('--loftq_config', type=str, default=None, help='LoFT-Q configuration')

    # Training hyperparameters
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                        help='Batch size per device during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=20, help='Number of warmup steps')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps')
    parser.add_argument('--optim', type=str, default='adamw_8bit', help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Learning rate scheduler type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_total_limit', type=int, default=50, help='Total save limit for model checkpoints')
    parser.add_argument('--save_step', type=int, default=5000, help='Steps interval to save model checkpoints')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4,
                        help='Batch size per device during evaluation')
    parser.add_argument('--train_lm_head', action='store_true', default=False,
                        help='Whether to train the language model head or not')
    parser.add_argument('--train_embed_tokens', action='store_true', default=False,
                        help='Whether to train the embed_tokens or not')

    # Output and evaluation
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Steps interval to evaluate the model')

    args = parser.parse_args()

    return args


def get_dataset(tokenizer, problem):
    # Define the Alpaca-style prompt template
    alpaca_prompt = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance. 
    Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective.

    ### Instruction:{}

    ### Input:{}

    ### Response:{}"""
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    # =========================
    # Load and Prepare Dataset
    # =========================

    # put the data in the data folder
    # Load training dataset from its own directory
    train_dataset = load_dataset('./data/' + problem + '/train', split="train", cache_dir="../datasets").shuffle(seed=42)
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

    # Load evaluation dataset from its own directory
    eval_dataset = load_dataset('./data/' + problem + '/eval', split="test", cache_dir="../datasets") 
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

    # instances = load_pkl_dataset('./data/' + problem + '/instances.pkl')
    instances = None
    print(train_dataset[0])

    return train_dataset, eval_dataset, instances


def compute_metric_cop(predictions, labels, instances, problem):
    """
    Compute custom feasibility and optimality gap metrics for TSP, OP, CVRP, MVC, or MIS.

    :param predictions: A list of full decoded strings (including prompt + response).
    :param labels: A list of reference/label strings (including prompt + gold solution).
    :param instances: The raw instances data for each example.
    :param problem: "tsp", "op", "cvrp", "mvc", or "mis".
    :return: (feasibility_rate, mean_optimality_gap).
    """
    gaps = []
    infeasibility = 0

    for i, prediction_full_text in enumerate(predictions):
        # Extract just the solution part from predicted text
        prediction_solution = extract_predicted_solution(prediction_full_text)

        # Extract the solution part from the gold/label text
        label_solution = extract_predicted_solution(labels[i])
        

        if problem == "tsp":
            # Parse route from the label
            label_match = re.search(r"Route:\s*\[([^\]]+)\]", label_solution)
            if not label_match:
                # If we can't parse the label route for some reason, skip
                infeasibility += 1
                continue

            tour_label_str = label_match.group(1)
            tour_label = list(map(int, tour_label_str.split(", ")))

            # Parse the route from the prediction
            pred_match = re.search(r"Route:\s*\[([^\]]+)\]", prediction_solution)
            if not pred_match:
                # No route found in predicted text
                infeasibility += 1
                continue

            tour_str = pred_match.group(1)
            tour_list = list(map(int, tour_str.split(", ")))

            # Check feasibility
            if len(set(tour_list)) != len(set(tour_label)):
                infeasibility += 1
                continue
            if tour_list[-1] != tour_list[0]:
                infeasibility += 1
                continue

            # Parse the objective from the gold (label_solution)
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                infeasibility += 1
                continue
            solution_distance = float(label_obj_match.group(1))

            # Calculate the LLM's objective
            llm_distance = calculate_total_distance(
                tour_list,
                compute_euclidean_distance_matrix(instances[i])
            )

            gap = (llm_distance - solution_distance) / solution_distance
            gaps.append(gap)

        elif problem == "op":
            # instance structure is [locs, prizes, max_length]
            pred_match = re.search(r"Route:\s*\[([^\]]+)\]", prediction_solution)
            if not pred_match:
                infeasibility += 1
                continue

            tour_str = pred_match.group(1)
            tour_list = list(map(int, tour_str.split(", ")))

            total_distance = calculate_total_distance(tour_list, compute_euclidean_distance_matrix(instances[i][0]))

            if total_distance > instances[i][2]:
                # The route is too long
                infeasibility += 1
                continue

            # Parse objective from the gold label
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                continue
            solution_prize = float(label_obj_match.group(1))

            llm_prize = sum(instances[i][1][j] for j in tour_list)
            gap = (solution_prize - llm_prize) / solution_prize
            gaps.append(gap)

        elif problem == "cvrp":
            # instance structure is [locs, demands, capacity]
            locs = instances[i][0]       # list of (x, y) coordinates
            demands = instances[i][1]    # list of demands for each node
            capacity = instances[i][2]   # vehicle capacity

            # Attempt to parse predicted routes from the text
            # Example: "Routes: [[0,1,2,0], [0,3,4,0]]"
            pred_match = re.search(r"Routes:\s*\[\s*(.*)\]", prediction_solution, re.DOTALL)
            if not pred_match:
                # No match: we cannot parse the solution
                infeasibility += 1
                continue

            routes_str = pred_match.group(1).strip()
            # Safely parse the string as a Python list of lists
            try:
                # Wrap in brackets to ensure it's recognized as a single Python list:
                # "[[0,1,2,0], [0,3,4,0]]" => We just do literal_eval directly.
                predicted_routes = ast.literal_eval(f'[{routes_str}]')
            except (SyntaxError, ValueError):
                infeasibility += 1
                continue

            # Check that we got a list of lists
            if not all(isinstance(r, list) for r in predicted_routes):
                infeasibility += 1
                continue
            

            # 1) Capacity check & route start/end check
            any_route_infeasible = False
            distance_matrix = compute_euclidean_distance_matrix(locs)
            
            try:
                for route in predicted_routes:
                    # Must start/end at depot 0
                    if not route or route[0] != 0 or route[-1] != 0:
                        any_route_infeasible = True
                        break

                    # Sum the demands
                    total_demand = sum(demands[node] for node in route if node != 0)
                    if total_demand > capacity:
                        any_route_infeasible = True
                        break

                if any_route_infeasible:
                    infeasibility += 1
                    continue
            except:
                infeasibility += 1
                continue

            # 2) Check that each customer (1..N-1) is visited exactly once.
            #    (Assuming the problem requires visiting *all* customers.)
            n_customers = len(demands)
            required_customers = set(range(1, n_customers))  # Usually 0 is the depot
            visited_customers = set()
            for route in predicted_routes:
                # Exclude the depot from visited customers
                # (first and last node are typically 0).
                visited_customers.update(route[1:-1])

            if visited_customers != required_customers:
                infeasibility += 1
                continue
            
            # 3) Parse reference objective
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                infeasibility += 1
                continue
            solution_cost = float(label_obj_match.group(1))

            # 4) Compute predicted cost
            pred_cost = 0.0
            for route in predicted_routes:
                pred_cost += calculate_total_distance(route, distance_matrix)

            # 5) Compute gap
            gap = (pred_cost - solution_cost) / solution_cost
            gaps.append(gap)

        elif problem == "mvc":

            # 1) Parse predicted cover from "Set: [ ... ]"
            pred_match = re.search(r"Response:\s*\[([^\]]+)\]", prediction_solution)
            if not pred_match:
                # No cover set found
                infeasibility += 1
                continue

            cover_str = pred_match.group(1).strip()
            # Attempt to parse the string as a list of ints
            try:
                predicted_cover = list(map(int, cover_str.split(",")))
            except ValueError:
                # Could not parse the set properly
                infeasibility += 1
                continue

            # 2) Check feasibility: every edge must be covered
            edges_mvc = instances[i][1]  # edges is the second element: (num_nodes, edges, ...)
            cover_set = set(predicted_cover)

            # If any edge is not covered, solution is infeasible
            not_covered = False
            for (u, v) in edges_mvc:
                if u not in cover_set and v not in cover_set:
                    not_covered = True
                    break

            if not_covered:
                infeasibility += 1
                continue

            # 3) Parse the reference (gold) objective from label
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                # If we can't parse the gold objective, skip
                infeasibility += 1
                continue

            optimal_cover_size = float(label_obj_match.group(1))

            # 4) Compute gap = (pred_cover_size - optimal_cover_size) / optimal_cover_size
            pred_cover_size = len(cover_set)
            gap = (pred_cover_size - optimal_cover_size) / (optimal_cover_size if optimal_cover_size != 0 else 1e-9)
            gaps.append(gap)
            
        elif problem == "mis":
            # 1) Parse predicted independent set from "Set: [ ... ]"
            pred_match = re.search(r"Response:\s*\[([^\]]+)\]", prediction_solution)
            if not pred_match:
                # No independent set found
                infeasibility += 1
                continue

            indset_str = pred_match.group(1).strip()
            # Attempt to parse the string as a list of ints
            try:
                predicted_indset = list(map(int, indset_str.split(",")))
            except ValueError:
                # Could not parse the set properly
                infeasibility += 1
                continue

            # 2) Check feasibility: no two vertices in the set should be adjacent
            # instance structure is expected to be (num_nodes, edges, ...)
            edges_mis = instances[i][1]  # edges is the second element
            indset = set(predicted_indset)
            
            # Check if any pair of vertices in the independent set are adjacent
            is_independent = True
            for (u, v) in edges_mis:
                if u in indset and v in indset:
                    is_independent = False
                    break
                    
            if not is_independent:
                infeasibility += 1
                continue

            # 3) Parse the reference (gold) objective from label
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                # If we can't parse the gold objective, skip
                infeasibility += 1
                continue

            optimal_indset_size = float(label_obj_match.group(1))

            # 4) Compute gap = (optimal_indset_size - pred_indset_size) / optimal_indset_size
            # Note: For MIS, we maximize the set size, so the gap is reversed compared to MVC
            pred_indset_size = len(indset)
            gap = (optimal_indset_size - pred_indset_size) / (optimal_indset_size if optimal_indset_size != 0 else 1e-9)
            gaps.append(gap)
        
        elif problem == "pfsp":
            # Parse the predicted job order
            pred_match = re.search(r"Order:\s*\[([^\]]+)\]", prediction_solution)
            if not pred_match:
                infeasibility += 1
                continue

            # Extract and parse the job order
            job_order_str = pred_match.group(1)
            try:
                job_order = list(map(int, job_order_str.split(", ")))
            except ValueError:
                infeasibility += 1
                continue

            # Extract instance data
            n_jobs = instances[i].shape[0]
            m_machines = instances[i].shape[1]
            
            # Parse processing times
            processing_times = instances[i].T
            
            
            # Check if the job order contains all jobs exactly once
            expected_jobs = set(range(1, n_jobs + 1))  # Jobs are 1-indexed
            if set(job_order) != expected_jobs or len(job_order) != n_jobs:
                infeasibility += 1
                continue

            # Parse the objective from the gold solution
            label_obj_match = re.search(r"Objective:\s*([\d.]+)", label_solution)
            if not label_obj_match:
                infeasibility += 1
                continue
            optimal_makespan = float(label_obj_match.group(1))
            
            # Calculate the makespan for the predicted job order
            predicted_makespan = calculate_pfsp_makespan(job_order, processing_times, m_machines)
            
            # Calculate optimality gap
            gap = (predicted_makespan - optimal_makespan) / optimal_makespan
            gaps.append(gap)

        else:
            raise NotImplementedError(f"Problem {problem} is not implemented!")

    # After looping over all predictions, compute feasibility and gap
    # If all are infeasible, handle corner case in gap average
    feasibility_rate = 1 - (infeasibility / len(predictions))
    if len(gaps) == 0:
        mean_gap = float('inf')  # or some fallback
    else:
        mean_gap = sum(gaps) / len(gaps)
    std_gap = np.std(gaps) if len(gaps) > 1 else 0.0

    return feasibility_rate, mean_gap, std_gap


def generate_and_compute_metrics(model, tokenizer, eval_dataset, instances, problem):
    """
    1) Generate predictions from the model on the entire eval_dataset.
    2) Decode predictions and references.
    3) Compute custom metrics via compute_metric_cop().
    4) Return a dict of metrics.
    """

    # We'll store predictions and labels to feed into compute_metric_cop
    all_preds = []
    all_labels = []

    # NOTE: For large datasets, consider sampling or doing batch inference to avoid OOM.
    for example_idx in range(len(eval_dataset)):
        sample_text = eval_dataset[example_idx]["text"]

        # -----------------------------------------------------------------
        # 1) Generate a prediction
        # -----------------------------------------------------------------
        # Convert to tokens
        input_ids = tokenizer(sample_text, return_tensors="pt")["input_ids"].to(model.device)

        # Generate. Customize parameters as needed (max_new_tokens, etc.).
        gen_tokens = model.generate(
            input_ids=input_ids,
            max_new_tokens=30000,
            do_sample=False
        )
        # Convert tokens to text
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

        # We'll treat the original example["text"] as the 'label' text
        # but in your usage, you might have a separate gold reference in the dataset.
        # Here, you said "eval_dataset = ... =>  'text':  <prompt + gold>"
        label_text = sample_text  # or a separate field if you have one

        all_preds.append(gen_text)
        all_labels.append(label_text)

    # -----------------------------------------------------------------
    # 2) Compute your custom feasibility & gap metrics
    # -----------------------------------------------------------------
    feasibility_rate, optimality_gap = compute_metric_cop(
        all_preds,
        all_labels,
        instances,
        problem
    )

    # Return the results as a dict
    return {
        "feasibility_rate": feasibility_rate,
        "optimality_gap": optimality_gap,
    }


class WandbEvalCallback(TrainerCallback):
    """
    A custom callback to run evaluation via model.generate
    and then log metrics to W&B at a given interval.
    """
    def __init__(self, model, tokenizer, eval_dataset, instances, problem, eval_steps):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.instances = instances
        self.problem = problem
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, **kwargs):
        """
        Hook that is called at the end of each training step.
        We'll run our custom evaluation if `state.global_step` is at
        an evaluation interval, and is not 0.
        """
        if state.global_step > 0 and (state.global_step % self.eval_steps == 0):
            metrics = generate_and_compute_metrics(
                model=self.model,
                tokenizer=self.tokenizer,
                eval_dataset=self.eval_dataset,
                instances=self.instances,
                problem=self.problem
            )
            # Log to W&B
            wandb.log({**metrics, "step": state.global_step})


def train_model(args):
    if args.output_dir is None:
        dir_out = (f"output_alpha{args.lora_alpha}_r{args.lora_r}_"
                   f"{args.problem}_gamma_"
                   f"train_embed_tok_{args.train_embed_tokens}_"
                   f"seq{args.max_seq_length}_"
                   f"b{args.per_device_train_batch_size}_"
                   f"ep{args.num_train_epochs}")
    else:
        dir_out = args.output_dir

    # =========================
    # Initialize WandB
    # =========================
    wandb.init(
        project=args.model_name.split('/')[1] + "_" + args.problem + "_cop_solver",
        name=dir_out,
    )

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    problem = args.problem

    # Load the pre-trained model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        cache_dir="/gpfs/work4/0/prjs0685/cache"
    )
    # model = model.to("cuda")

    # Setup LoRA / PEFT
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    if args.train_lm_head:
        target_modules.append('lm_head')
    if args.train_embed_tokens:
        target_modules.append('embed_tokens')

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        bias=args.bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        loftq_config=args.loftq_config
    )
    
    # response_template = " ### Response:"
    # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    collator = SafeDataCollatorForCompletionOnlyLM(
        response_template="### Response:", 
        tokenizer=tokenizer,
        fallback_strategy="full_example"  # Options: "last_portion", "full_example", "skip"
    )
    # Get data
    train_dataset, eval_dataset, instances = get_dataset(tokenizer, problem)
    # print(train_dataset[0:8])

    # =========================
    # Create the Trainer
    # =========================
    # - We do NOT set eval_dataset, eval_strategy, or compute_metrics.
    # - We'll rely on our custom callback to do the evaluation steps.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=16,
        packing=False,
        data_collator=collator,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=dir_out,
            report_to="wandb",
            evaluation_strategy="no",
            metric_for_best_model="eval_loss",
            # eval_steps=args.eval_steps,
            # load_best_model_at_end=True,
            save_total_limit=args.save_total_limit,
            save_steps=args.save_step,
        ),
    )


    # =========================
    # Train
    # =========================
    trainer.train(resume_from_checkpoint=True)
    # trainer.train()

    return trainer

if __name__ == "__main__":
    args = parse_args()
    train_model(args)

# srun -n 1 -p gpu_h100 -t 00:10:00 --gpus-per-node 1 --pty /bin/bash

