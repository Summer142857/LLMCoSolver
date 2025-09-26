#!/bin/bash

# Set the model directory path (change this if needed)
MODEL_DIR="./output_alpha64_r64_jssp_gamma_train_embed_tok_False_seq20000_b4_ep1/checkpoint-31250"

# Remove saved_models directory if it exists
if [ -d "./saved_models" ]; then
    echo "Removing existing saved_models directory..."
    rm -rf "./saved_models"
fi

# Create a temporary Python script
cat > merge_model.py << EOF
import torch
from peft import AutoPeftModelForCausalLM

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    "${MODEL_DIR}",
    #load_in_4bit = True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./saved_models", safe_serialization=True, max_shard_size="2GB")
EOF

# Execute the Python script
echo "Merging LoRA and base model from ${MODEL_DIR}..."
if python merge_model.py; then
    echo "Model merging completed successfully!"
else
    echo "Error: Model merging failed!"
    rm merge_model.py
    exit 1
fi

# Clean up the temporary Python script
rm merge_model.py

# Copy all files from MODEL_DIR to saved_models
echo "Copying files from ${MODEL_DIR} to saved_models..."
if cp -r "${MODEL_DIR}"/* ./saved_models/; then
    echo "File copying completed successfully!"
else
    echo "Error: File copying failed!"
    exit 1
fi

echo "All operations completed successfully!"
