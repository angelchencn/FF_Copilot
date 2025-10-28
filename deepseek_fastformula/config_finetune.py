"""
Configuration file for DeepSeek-Coder FastFormula fine-tuning.
This file contains all the hyperparameters and settings for training.

You can import this in your training script or modify values as needed.
"""

# Model Configuration
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
MODEL_MAX_LENGTH = 16384  # DeepSeek-Coder supports up to 16K context

# Data Configuration
DATA_PATH = "../data/fastformula"
TRAIN_VAL_SPLIT = 0.9  # 90% train, 10% validation

# Training Configuration
OUTPUT_DIR = "./deepseek_coder_fastformula_output"
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 2048  # Actual sequence length used in training

# Optimizer Configuration
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
LR_SCHEDULER_TYPE = "cosine"

# LoRA Configuration
USE_LORA = True
LORA_R = 16  # LoRA rank (increase for more parameters)
LORA_ALPHA = 32  # LoRA alpha (typically 2x the rank)
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

# Logging and Checkpointing
LOGGING_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 250
SAVE_TOTAL_LIMIT = 2  # Keep only the 2 best checkpoints
LOAD_BEST_MODEL_AT_END = True

# Generation Configuration (for inference)
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
}

# Custom Tokens for FastFormula
CUSTOM_TOKENS = [
    # Control flow keywords
    "DEFAULT FOR", "IF", "THEN", "ELSIF", "ELSE", "RETURN",

    # Common time variables
    "HOURS_WORKED", "REGULAR_HOURS", "TOTAL_HOURS", "OVERTIME_HOURS",

    # Common pay variables
    "PAY_RATE", "HOURLY_RATE", "OVERTIME_RATE", "OVERTIME_PAY",
    "GROSS_PAY", "BASE_SALARY", "TAXABLE_PAY", "NET_PAY",

    # Bonus and deduction variables
    "BONUS_AMOUNT", "BONUS_PERCENT", "DEDUCTION_RATE", "ALLOWANCE_RATE",
    "TAX_RATE", "TAX_AMOUNT",

    # Generic variables
    "INPUT_VALUE", "OUTPUT_VALUE", "THRESHOLD", "LIMIT", "AMOUNT",
    "MESSAGE", "RESULT", "VALUE", "FACTOR", "RATE", "PERCENTAGE",

    # Days of week
    "WEEKDAY", "SUNDAY", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
    "FRIDAY", "SATURDAY",

    # Special markers
    "### START", "### END", "FORMULA_BEGIN", "FORMULA_END",
    "### Instruction:", "### Response:"
]

# Hardware Configuration
USE_GPU = True
MIXED_PRECISION = "bf16"  # "bf16", "fp16", or "no"
GRADIENT_CHECKPOINTING = True  # Save memory at the cost of speed

# Training Features
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 evaluations
METRIC_FOR_BEST_MODEL = "eval_loss"
GREATER_IS_BETTER = False

# Data Processing
PADDING_SIDE = "right"
TRUNCATION_SIDE = "right"
RETURN_OVERFLOWING_TOKENS = False

# Advanced Settings
DEEPSPEED_CONFIG = None  # Set to deepspeed config file if using DeepSpeed
FSDP_CONFIG = None  # Set to FSDP config if using Fully Sharded Data Parallel
GRADIENT_CHECKPOINTING_KWARGS = {"use_reentrant": False}

# Reporting
REPORT_TO = "none"  # Options: "none", "wandb", "tensorboard"
WANDB_PROJECT = "fastformula-deepseek"
WANDB_RUN_NAME = None  # Auto-generate if None

# Prompt Templates
INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
USER_TEMPLATE = "User: {question}\n\nAssistant: {answer}"

# Model Size Information (for reference)
MODEL_INFO = {
    "parameters": "6.7B",
    "context_length": "16K tokens",
    "training_tokens": "2T tokens (87% code, 13% natural language)",
    "supported_languages": "86+ programming languages",
    "architecture": "Transformer decoder",
    "special_features": [
        "Fill-in-the-Middle (FIM) capability",
        "Code completion",
        "Long context understanding",
        "Multi-language support"
    ]
}

# Example Usage
"""
from config_finetune import *

# Use in training script
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    ...
)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    ...
)
"""

if __name__ == "__main__":
    print("DeepSeek-Coder FastFormula Fine-tuning Configuration")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Data: {DATA_PATH}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"LoRA: {'Enabled' if USE_LORA else 'Disabled'}")
    if USE_LORA:
        print(f"  - Rank: {LORA_R}")
        print(f"  - Alpha: {LORA_ALPHA}")
        print(f"  - Dropout: {LORA_DROPOUT}")
    print(f"Custom tokens: {len(CUSTOM_TOKENS)}")
    print("=" * 60)
