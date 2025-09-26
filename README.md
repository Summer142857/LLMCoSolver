# LLMCoSolver: Large Language Models as End-to-end Combinatorial Optimization Solvers

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://openreview.net/forum?id=qr5uMEs6iR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper **"Large Language Models as End-to-end Combinatorial Optimization Solvers"** presented at The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025).

## 📖 TL; DR

A framework for training Large Language Models (LLMs) to solve combinatorial optimization problems using supervised fine-tuning (SFT) followed by reinforcement learning (RL).

## 📰 Paper

**Title:** Large Language Models as End-to-end Combinatorial Optimization Solvers

**Authors:** Xia Jiang, Yaoxin Wu, Minshuo Li, Zhiguang Cao, Yingqian Zhang

**Conference:** The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)

**Paper Link:** [Arxiv](https://arxiv.org/abs/2509.16865)

## 🚀 Overview

It now supports training and evaluation on multiple combinatorial optimization problems:
- **TSP** (Traveling Salesman Problem)
- **CVRP** (Capacitated Vehicle Routing Problem) 
- **OP** (Orienteering Problem)
- **MVC** (Minimum Vertex Cover)
- **MIS** (Maximum Independent Set)
- **PFSP** (Permutation Flow Shop Problem)
- **JSSP** (Job Shop Scheduling Problem)

## 🔔 Data Format

You can generate your own data through the problem-specific environments under /Envs/, or use the data generated in the original paper: https://drive.google.com/drive/folders/1bE1coGUa00gfuMkPXnfvldi1-WHGNnEb?usp=sharing

Place your training and evaluation data in the following structure:
```
data/
├── <problem_name>/
│   ├── train/           # Training data
│   ├── eval/            # Evaluation data  
│   └── instances.pkl    # Problem instances
```


## 💻 Training Pipeline

The training consists of three main stages:

### 1. Supervised Fine-Tuning (SFT)

First, train the model using supervised learning on problem-specific data:

```bash
python main_train.py --problem <problem_name> [options]
```

**Key parameters:**
- `--problem`: Problem type (tsp, cvrp, op, mvc, mis, pfsp, jssp)
- `--model_name`: Base model to fine-tune (default: unsloth/Qwen2.5-7B)
- `--max_seq_length`: Maximum sequence length (default: 20000)
- `--per_device_train_batch_size`: Batch size per device (default: 4)
- `--num_train_epochs`: Number of training epochs (default: 1)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--lora_r`: LoRA rank (default: 64)
- `--lora_alpha`: LoRA alpha (default: 64)

**Example:**
```bash
python main_train.py --problem cvrp --num_train_epochs 1 --per_device_train_batch_size 4
```

### 2. Reinforcement Learning (RL)

After SFT, improve the model using reinforcement learning (GRPO):

```bash
python rl_train.py --problem <problem_name> --model_name <sft_checkpoint_path> [options]
```

**Key parameters:**
- `--model_name`: Path to SFT checkpoint (e.g., `output_alpha64_r64_cvrp_gamma_train_embed_tok_False_seq20000_b4_ep1/checkpoint-31250`)
- `--num_generations`: Number of generations for GRPO (default: 8)
- `--beta`: KL coefficient (default: 0.05)
- `--learning_rate`: Learning rate (default: 1e-6)
- `--max_prompt_length`: Maximum prompt length (default: 20000)
- `--max_completion_length`: Maximum completion length (default: 1000)

**Example:**
```bash
python rl_train.py --problem cvrp --model_name output_alpha64_r64_cvrp_gamma_train_embed_tok_False_seq20000_b4_ep1/checkpoint-31250
```

### 3. Model Merging

After training, merge the LoRA weights with the base model:

1. Edit `cmd.sh` to specify your model checkpoint path:
   ```bash
   MODEL_DIR="./path/to/your/checkpoint"
   ```

2. Run the merge script:
   ```bash
   bash cmd.sh
   ```

This creates a `saved_models/` directory with the merged model.

## 🧪 Evaluation

Evaluate the trained model using two methods:

### Vanilla Evaluation
```bash
python eval.py --model_id saved_models --problem <problem_name> --eval_method vanilla --num_samples 100
```

### Best-of-N Evaluation
```bash
python eval.py --model_id saved_models --problem <problem_name> --eval_method best_of_n --num_samples 100 --best_of_n 8 --temperature 0.7
```

**Evaluation parameters:**
- `--model_id`: Path to the merged model (default: saved_models)
- `--eval_method`: Evaluation method (vanilla or best_of_n)
- `--num_samples`: Number of test instances to evaluate
- `--best_of_n`: Number of solutions to generate per instance (for best_of_n)
- `--temperature`: Sampling temperature
- `--batch_size`: Batch size for evaluation

### Output Metrics

The evaluation provides:
- **Feasibility Rate**: Percentage of valid solutions
- **Optimality Gap**: Average gap from optimal/reference solutions  

## 📊 Quick Start Example

Here's a complete example for training on CVRP:

```bash
# 1. Supervised Fine-Tuning
python main_train.py --problem cvrp --num_train_epochs 1

# 2. Reinforcement Learning  
python rl_train.py --problem cvrp --model_name output_alpha64_r64_cvrp_gamma_train_embed_tok_False_seq20000_b4_ep1/checkpoint-31250

# 3. Merge Model (edit MODEL_DIR in cmd.sh first)
bash cmd.sh

# 4. Evaluate
python eval.py --model_id saved_models --problem cvrp --eval_method vanilla --num_samples 100
```


## 🤝 Contributing

We welcome contributions to this project. Please feel free to submit issues and pull requests.

## 📜 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{
jiang2025large,
title={Large Language Models as End-to-end Combinatorial Optimization Solvers},
author={Xia Jiang, Yaoxin Wu, Minshuo Li, Zhiguang Cao, Yingqian Zhang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=qr5uMEs6iR}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

