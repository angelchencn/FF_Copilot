"""
Configuration file for Qwen2.5-Coder-7B FastFormula fine-tuning.
"""

import torch

# Output settings
out_dir = 'qwen2_5_coder_fastformula_output'
eval_interval = 250
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True

# Model settings
model_name = 'Qwen/Qwen2.5-Coder-7B'
data_path = '../data/fastformula'

# Training settings
num_epochs = 3
batch_size = 2  # Smaller batch size for 7B model
gradient_accumulation_steps = 8  # Higher accumulation for effective batch size
block_size = 4096  # Qwen2.5-Coder context length
max_length = 2048  # For processing

# Optimizer settings
learning_rate = 2e-4
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 100
lr_decay_iters = 1000

# LoRA settings (recommended for 7B model)
use_lora = True
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# System settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Qwen2.5-Coder uses Flash Attention

# Checkpoint settings
save_steps = 500
logging_steps = 10
eval_steps = 250

# Wandb logging (optional)
wandb_log = False
wandb_project = 'qwen2_5_coder-fastformula'
wandb_run_name = 'qwen2_5_coder-ff-finetune'

