"""
Configuration file for Phi-2 FastFormula fine-tuning.
This follows the same style as other config files in the project.
"""

import torch

# Output settings
out_dir = 'phi2_fastformula_output'
eval_interval = 250
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True

# Model settings
model_name = 'microsoft/phi-2'
data_path = '../data/fastformula/input.txt'

# Training settings
num_epochs = 3
batch_size = 4  # Reduced for Phi-2 (could be 8-16 for larger GPUs)
gradient_accumulation_steps = 4
block_size = 2048  # Phi-2 context length
max_length = 1024  # For processing

# Optimizer settings
learning_rate = 2e-4
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 100
lr_decay_iters = 1000  # Adjust based on epochs

# LoRA settings (for efficient fine-tuning)
use_lora = False
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

# System settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Phi-2 uses Flash Attention, compilation not needed

# Checkpoint settings
save_steps = 500
logging_steps = 10
eval_steps = 250

# Wandb logging (optional)
wandb_log = False
wandb_project = 'phi2-fastformula'
wandb_run_name = 'phi2-ff-finetune'

