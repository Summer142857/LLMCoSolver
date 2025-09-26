import argparse
import torch
import wandb
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, TrainerCallback
import re
from utils import *
import json
from trl import GRPOConfig, GRPOTrainer
from rewards import *
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

PatchFastRL("GRPO", FastLanguageModel)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RL Trainer for solving combinatorial optimization problems')

    # Model and data parameters
    parser.add_argument('--max_prompt_length', type=int, default=20000, help='Maximum prompt length')
    parser.add_argument('--max_completion_length', type=int, default=1000, help='Maximum completion length')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],
                        help='Data type (bfloat16 or float16)')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                        help='Use 4-bit quantization to reduce memory usage')
    # Model name
    parser.add_argument('--model_name', type=str, default='output_alpha64_r64_cvrp_gamma_train_embed_tok_False_seq20000_b4_ep1/checkpoint-31250', help='Model name')
    # Problem name
    parser.add_argument('--problem', type=str, default='cvrp', help='Problem name')

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=64, help='Rank of the LoRA decomposition')
    parser.add_argument('--lora_alpha', type=int, default=64, help='Scaling factor for LoRA updates')
    parser.add_argument('--bias', type=str, default='lora_only', choices=['none', 'all', 'lora_only'], help='Bias type')

    # GRPO hyperparameters
    parser.add_argument('--num_generations', type=int, default=8, help='Number of generations')
    parser.add_argument('--beta', type=float, default=0.05, help='KL coefficient')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for clipping')
    parser.add_argument('--epsilon_high', type=float, default=0.28, help='Upper-bound epsilon value for clipping')
    parser.add_argument('--reward_weights', type=list, default=[1, 1], help='Weights for the reward functions')

    # Additional configurations
    # previous seed: 42
    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth', help='Use gradient checkpointing')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--use_rslora', action='store_true', default=False, help='Use RSLoRA')
    parser.add_argument('--loftq_config', type=str, default=None, help='LoFT-Q configuration')

    # Training hyperparameters
    parser.add_argument('--per_device_train_batch_size', type=int, default=8,
                        help='Batch size per device during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--warmup_ratio', type=int, default=0.05, help='Number of warmup steps')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps')
    parser.add_argument('--optim', type=str, default='adamw_8bit', help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='Weight decay')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Learning rate scheduler type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_total_limit', type=int, default=20, help='Total save limit for model checkpoints')
    parser.add_argument('--save_step', type=int, default=50, help='Steps interval to save model checkpoints')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8,
                        help='Batch size per device during evaluation')
    parser.add_argument('--train_lm_head', action='store_true', default=False,
                        help='Whether to train the language model head or not')
    parser.add_argument('--train_embed_tokens', action='store_true', default=False,
                        help='Whether to train the embed_tokens or not')

    # Output and evaluation
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name')
    parser.add_argument('--eval_steps', type=int, default=100, help='Steps interval to evaluate the model')

    args = parser.parse_args()

    return args


def train_model(args):
    if args.output_dir is None:
        dir_out = (f"output_alpha{args.lora_alpha}_r{args.lora_r}_"
                   f"{args.problem}_gemma_"
                   f"train_embed_tok_{args.train_embed_tokens}_"
                   f"seq{args.max_completion_length}_"
                   f"b{args.per_device_train_batch_size}_"
                   f"ep{args.num_train_epochs}")
    else:
        dir_out = args.output_dir

    # =========================
    # Initialize WandB
    # =========================
    wandb.init(
        project=args.model_name.split('/')[1] + "_gemma_cop_solver",
        name=dir_out,
    )

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    problem = args.problem

    # Load the pre-trained model and tokenizer
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name,
    #     device_map="auto",
    #     torch_dtype=dtype
    # )
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_prompt_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        fast_inference=False
    )
    model = model.to("cuda")

    # Setup LoRA / PEFT
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


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


    # Get data - now using the function from utils.py
    train_dataset, eval_dataset = get_dataset(problem, tokenizer)
    
    # =========================
    # Create the Trainer
    # =========================
    if args.problem == 'cvrp':
        reward_funcs = [optimality_reward_func_cvrp, feasibility_reward_func_cvrp]
    elif args.problem == 'op':
        reward_funcs = [optimality_reward_func_op, feasibility_reward_func_op]
    elif args.problem == 'tsp':
        reward_funcs = [optimality_reward_func_tsp, feasibility_reward_func_tsp]
    elif args.problem == 'mvc':
        reward_funcs = [optimality_reward_func_mvc, feasibility_reward_func_mvc]
    elif args.problem == 'mis':
        reward_funcs = [optimality_reward_func_mis, feasibility_reward_func_mis]
    elif args.problem == 'jssp':
        reward_funcs = [optimality_reward_func_jssp, feasibility_reward_func_jssp]
    else:
        raise ValueError("Problem not supported for reward functions.")
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        # reward_funcs=reward_funcs,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=GRPOConfig(
            use_vllm = False,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            beta = args.beta,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            # epsilon =args.epsilon,
            # epsilon_high = args.epsilon_high,
            seed=args.seed,
            num_generations=args.num_generations,
            # reward_weights=args.reward_weights,
            max_prompt_length = args.max_prompt_length,
            max_completion_length = args.max_completion_length,
            output_dir=dir_out,
            report_to="wandb",
            evaluation_strategy="no",
            # eval_steps=args.eval_steps,
            # load_best_model_at_end=True,
            save_total_limit=args.save_total_limit,
            save_steps=args.save_step,
        )
    )
    

    # =========================
    # Train
    # =========================
    trainer.train()

    return trainer


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
# srun -n 1 -p gpu_h100 -t 00:25:00 --gpus-per-node 1 --pty /bin/bash
