import unsloth
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from utils import load_pkl_dataset
from utils import compute_metric_cop
from tqdm import tqdm
from rl_train import get_dataset
from Envs.eval_utils import (
    optimality_reward_func_op,
    optimality_reward_func_tsp,
    optimality_reward_func_mvc,
    optimality_reward_func_cvrp,
    optimality_reward_func_mis,
    optimality_reward_func_pfsp,
    optimality_reward_func_jssp
)
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Tester for solving combinatorial optimization problems')

    # Model and data parameters
    parser.add_argument('--model_id', type=str, default='saved_models', help='Model path')
    parser.add_argument('--problem', type=str, default='cvrp', help='Problem name')
    
    # Evaluation method selection
    parser.add_argument('--eval_method', type=str, default='vanilla', choices=['vanilla', 'best_of_n'], 
                        help='Evaluation method: vanilla or best_of_n')
    
    # Parameters for both methods
    parser.add_argument('--num_samples', type=int, default=100, 
                        help='Number of samples to evaluate (default: 100)')
    
    # Parameters specific to Best-of-N evaluation
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation (best_of_n only)')
    parser.add_argument('--best_of_n', type=int, default=8, help='Number of solutions to generate per prompt (best_of_n only)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling parameter (best_of_n only)')
    
    # Dataset loading method
    parser.add_argument('--dataset_method', type=str, default='auto', choices=['auto', 'load_dataset', 'get_dataset'],
                        help='Method to load dataset: auto (based on eval_method), load_dataset, or get_dataset')

    args = parser.parse_args()
    
    return args


def load_model_and_tokenizer(model_id):
    """
    Load the model and tokenizer from the given model ID.
    
    Args:
        model_id (str): Path to the model
        
    Returns:
        tuple: (model, tokenizer, pipeline)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    return model, tokenizer, pipe


def load_datasets(problem, tokenizer, dataset_method='auto', eval_method='vanilla'):
    """
    Load the evaluation datasets using the specified method.
    
    Args:
        problem (str): Problem name
        tokenizer: The tokenizer for the model
        dataset_method (str): Method to load dataset ('auto', 'load_dataset', or 'get_dataset')
        eval_method (str): Evaluation method ('vanilla' or 'best_of_n')
        
    Returns:
        tuple: (eval_dataset, number_dataset)
    """
    # Determine which dataset loading method to use
    if dataset_method == 'auto':
        # For vanilla eval, use load_dataset; for best_of_n, use get_dataset
        dataset_method = 'load_dataset' if eval_method == 'vanilla' else 'get_dataset'
    
    if dataset_method == 'load_dataset':
        # Method used in main_eval.py
        eval_dataset = load_dataset(f'./data/{problem}/eval', split="test")
    else:  # get_dataset
        # Method used in BestofNEval.py
        _, eval_dataset = get_dataset(problem, tokenizer, num_samples=None, train=False)
    
    # Load number dataset (common to both methods)
    number_dataset = load_pkl_dataset(f'./data/{problem}/instances.pkl')
    
    return eval_dataset, number_dataset


def get_generation_kwargs(tokenizer, eval_method, n=8, temperature=0.7, top_p=0.9):
    """
    Prepare generation kwargs based on the evaluation method.
    
    Args:
        tokenizer: The tokenizer for the model
        eval_method (str): Evaluation method ('vanilla' or 'best_of_n')
        n (int): Number of solutions to generate per prompt (for best_of_n)
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling parameter (for best_of_n)
        
    Returns:
        dict: Generation kwargs
    """
    if eval_method == 'vanilla':
        # Generation kwargs for vanilla evaluation (from main_eval.py)
        return {
            "max_new_tokens": 3000,
            "do_sample": False,
            "temperature": 0.1,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }
    else:  # best_of_n
        # Generation kwargs for Best-of-N evaluation (from BestofNEval.py)
        return {
            "max_new_tokens": 3000,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": n,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }


def prepare_batch_prompts(eval_dataset, batch_indices):
    """
    Prepare batch prompts for the model.
    
    Args:
        eval_dataset: The evaluation dataset
        batch_indices (list): List of indices for the current batch
        
    Returns:
        list: Batch prompts
    """
    alpaca_prompt = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance. 
    Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective.

    ### Instruction:{}

    ### Input:{}

    ### Response:"""
    
    batch_prompts = []
    for idx in batch_indices:
        instruction = eval_dataset[idx]['instruction']
        user_input = eval_dataset[idx]['input']
        prompt = alpaca_prompt.format(instruction, user_input)
        batch_prompts.append(prompt)
    
    return batch_prompts


def select_best_solution(completions, ground_truth, idx, n, problem, eval_dataset):
    """
    Select the best solution from the generated completions.
    
    Args:
        completions (list): List of generated completions
        ground_truth: Ground truth for the current instance
        idx (int): Index of the current instance
        n (int): Number of solutions generated per prompt
        problem (str): Problem name
        eval_dataset: The evaluation dataset
        
    Returns:
        str: The best completion
    """
    # Replicate ground truth N times to match the shape
    repeated_ground_truth = [ground_truth[idx]] * n
    
    if problem == 'op':
        # Prepare data for the orienteering problem
        instance_coords = eval_dataset["instance_coords"]
        instance_max_dist = eval_dataset["instance_max_dist"]
        instance_prizes = eval_dataset["instance_prizes"]
        
        repeated_instance_coords = [instance_coords[idx]] * n
        repeated_instance_max_dist = [instance_max_dist[idx]] * n
        repeated_instance_prizes = [instance_prizes[idx]] * n
        
        # Compute reward for each candidate
        rewards = optimality_reward_func_op(
            completions,
            repeated_ground_truth,
            repeated_instance_coords,
            repeated_instance_max_dist,
            repeated_instance_prizes
        )
    elif problem == 'tsp':
        # Prepare data for the TSP problem
        instance = eval_dataset["instance"][idx]  # Get the distance matrix for this instance
        repeated_instance = [instance] * n
        
        # Compute reward for each candidate
        rewards = optimality_reward_func_tsp(
            completions,
            repeated_ground_truth,
            repeated_instance
        )
    elif problem == 'mvc':
        # Prepare data for the MVC problem
        instance = eval_dataset["instance"][idx]  # Get the graph instance for this example
        repeated_instance = [instance] * n
        
        # Compute reward for each candidate
        rewards = optimality_reward_func_mvc(
            completions,
            repeated_ground_truth,
            repeated_instance
        )
    elif problem == 'cvrp':
        # Prepare data for the CVRP problem
        # instance structure is [locs, demands, capacity]
        locs = eval_dataset["instance_coords"][idx]       # list of (x, y) coordinates
        demands = eval_dataset["instance_demands"][idx]   # list of demands for each node
        capacity = eval_dataset["instance_capacity"][idx] # vehicle capacity

        
        # Compute reward for each candidate
        rewards = optimality_reward_func_cvrp(
            completions,
            repeated_ground_truth,
            [locs] * n,
            [demands] * n,
            [capacity] * n
        )
    elif problem == 'mis':
        instance = eval_dataset["instance"][idx]
        repeated_instance = [instance] * n
        
        # Compute reward for each candidate
        rewards = optimality_reward_func_mis(
            completions,
            repeated_ground_truth,
            repeated_instance
        )
    elif problem == 'pfsp':
        instance = eval_dataset["instance"][idx]
        repeated_instance = [instance] * n
        
        # Compute reward for each candidate
        rewards = optimality_reward_func_pfsp(
            completions,
            repeated_ground_truth,
            repeated_instance
        )
    elif problem == 'jssp':
        instance = eval_dataset["instance"][idx]
        repeated_instance = [instance] * n
        
        # Compute reward for each candidate
        rewards = optimality_reward_func_jssp(
            completions,
            repeated_ground_truth,
            repeated_instance
        )
    
    # Select the candidate with the highest reward
    best_idx = int(np.argmax(rewards))
    return completions[best_idx]


def evaluate_vanilla(args, pipe, eval_dataset, number_dataset):
    """
    Evaluate the model using vanilla evaluation (from main_eval.py).
    
    Args:
        args: Command line arguments
        pipe: Text generation pipeline
        eval_dataset: The evaluation dataset
        number_dataset: The number dataset
        
    Returns:
        tuple: (feasibility_rate, optimality_gap)
    """
    predictions = []
    labels = [item['output'] for item in eval_dataset]
    
    # Generate a single solution for each prompt
    for idx in tqdm(range(args.num_samples)):
        prompt = prepare_batch_prompts(eval_dataset, [idx])[0]
        generation = pipe(prompt, **get_generation_kwargs(pipe.tokenizer, 'vanilla', temperature=args.temperature))
        predictions.append(generation[0]['generated_text'])
    # print(predictions)    
    # Evaluate the predictions
    fea_rate, opt_gap, std_gap = compute_metric_cop(predictions, labels, number_dataset, problem=args.problem)
    
    return fea_rate, opt_gap, std_gap


def evaluate_best_of_n(args, pipe, eval_dataset, number_dataset):
    """
    Evaluate the model using Best-of-N evaluation (from BestofNEval.py).
    
    Args:
        args: Command line arguments
        pipe: Text generation pipeline
        eval_dataset: The evaluation dataset
        number_dataset: The number dataset
        
    Returns:
        tuple: (feasibility_rate, optimality_gap)
    """
    # Prepare evaluation parameters
    batch_size = args.batch_size
    n = args.best_of_n
    indices = range(args.num_samples)
    
    # Get generation kwargs
    generation_kwargs = get_generation_kwargs(
        pipe.tokenizer, 'best_of_n', n=n, temperature=args.temperature, top_p=args.top_p
    )
    
    # Prepare labels and predictions
    labels = [item['output'] for item in eval_dataset]
    predictions = []
    
    # Get ground truth
    ground_truth = eval_dataset["ground_truth"]
    
    # Evaluate in batches
    for start_idx in tqdm(range(0, len(indices), batch_size)):
        end_idx = min(start_idx + batch_size, len(indices))
        batch_indices = list(indices)[start_idx:end_idx]
        
        # Build batch prompts
        batch_prompts = prepare_batch_prompts(eval_dataset, batch_indices)
        
        # Generate best-of-N solutions for each prompt in the batch
        raw_generations = pipe(batch_prompts, **generation_kwargs)
        
        # Regroup generations by each prompt
        grouped_generations = [
            raw_generations[i * n : (i + 1) * n]
            for i in range(len(batch_prompts))
        ]
        
        # For each example in the batch, pick the best solution
        for i, idx in enumerate(batch_indices):
            # Extract all candidate completions for this example
            completions = [
                gen["generated_text"] for gen in grouped_generations[i][0]
            ]
            
            # Select the best solution
            best_completion = select_best_solution(
                completions, ground_truth, idx, n, args.problem, eval_dataset
            )
            
            predictions.append(best_completion)
    
    # Evaluate the predictions
    # print(predictions)
    fea_rate, opt_gap, std_gap = compute_metric_cop(predictions, labels, number_dataset, problem=args.problem)
    
    return fea_rate, opt_gap, std_gap


def evaluate_model(args):
    """
    Evaluate the model using the specified evaluation method.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (feasibility_rate, optimality_gap)
    """
    # Load model, tokenizer, and pipeline
    _, tokenizer, pipe = load_model_and_tokenizer(args.model_id)
    
    # Load datasets
    eval_dataset, number_dataset = load_datasets(
        args.problem, tokenizer, args.dataset_method, args.eval_method
    )
    
    # Evaluate using the specified method
    if args.eval_method == 'vanilla':
        return evaluate_vanilla(args, pipe, eval_dataset, number_dataset)
    else:  # best_of_n
        return evaluate_best_of_n(args, pipe, eval_dataset, number_dataset)


def main():
    """Main function to run the evaluation."""
    args = parse_args()
    
    print(f"Running {args.eval_method} evaluation on {args.problem} problem...")
    print(f"Model: {args.model_id}")
    print(f"Number of samples: {args.num_samples}")
    
    if args.eval_method == 'best_of_n':
        print(f"Best-of-N: {args.best_of_n}")
        print(f"Batch size: {args.batch_size}")
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {args.top_p}")
    
    # Evaluate the model
    fea_rate, opt_gap, std_gap = evaluate_model(args)
    
    # Print metrics
    print(f"Feasibility Rate: {fea_rate}")
    print(f"Optimality Gap: {opt_gap}")
    print(f"Standard Deviation of Optimality Gap: {std_gap}")


if __name__ == "__main__":
    main()
