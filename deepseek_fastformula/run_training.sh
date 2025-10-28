#!/bin/bash

# Quick start script for DeepSeek-Coder FastFormula fine-tuning
# Usage: ./run_training.sh [options]

set -e  # Exit on error

# Default values
OUTPUT_DIR="./deepseek_coder_fastformula_output"
DATA_PATH="../data/fastformula"
MODEL_NAME="deepseek-ai/deepseek-coder-6.7b-base"
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=2e-4
MAX_LENGTH=2048
USE_LORA="--use_lora"
LORA_R=16
LORA_ALPHA=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --no-lora)
            USE_LORA=""
            shift
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_training.sh [options]"
            echo ""
            echo "Options:"
            echo "  --output_dir DIR        Output directory (default: $OUTPUT_DIR)"
            echo "  --data_path PATH        Training data path (default: $DATA_PATH)"
            echo "  --model_name NAME       Model name (default: $MODEL_NAME)"
            echo "  --epochs N              Number of epochs (default: $NUM_EPOCHS)"
            echo "  --batch_size N          Batch size (default: $BATCH_SIZE)"
            echo "  --learning_rate LR      Learning rate (default: $LEARNING_RATE)"
            echo "  --max_length N          Max sequence length (default: $MAX_LENGTH)"
            echo "  --no-lora               Disable LoRA (full fine-tuning)"
            echo "  --lora_r N              LoRA rank (default: $LORA_R)"
            echo "  --lora_alpha N          LoRA alpha (default: $LORA_ALPHA)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "============================================================"
echo "DeepSeek-Coder FastFormula Fine-tuning"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model:           $MODEL_NAME"
echo "  Data path:       $DATA_PATH"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Epochs:          $NUM_EPOCHS"
echo "  Batch size:      $BATCH_SIZE"
echo "  Learning rate:   $LEARNING_RATE"
echo "  Max length:      $MAX_LENGTH"
echo "  LoRA:            $([ -n "$USE_LORA" ] && echo "Enabled (r=$LORA_R, alpha=$LORA_ALPHA)" || echo "Disabled")"
echo ""
echo "============================================================"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_PATH" ] && [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data path does not exist: $DATA_PATH"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import torch; import transformers; import peft" 2>/dev/null || {
    echo "Error: Required packages not installed"
    echo "Please run: pip install -r requirements.txt"
    exit 1
}

# Check CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Build command
CMD="python train_deepseek_coder.py \
    --model_name \"$MODEL_NAME\" \
    --data_path \"$DATA_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH"

if [ -n "$USE_LORA" ]; then
    CMD="$CMD --use_lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA"
fi

# Run training
echo "Starting training..."
echo "Command: $CMD"
echo ""

eval $CMD

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Training completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "To test the model, run:"
    echo "  python inference_deepseek_coder.py --model_path \"$OUTPUT_DIR\" --interactive"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "Training failed!"
    echo "============================================================"
    echo ""
    echo "Please check the error messages above."
    exit 1
fi
