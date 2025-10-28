#!/bin/bash
# Quick start script for Qwen2.5-Coder-7B FastFormula fine-tuning

# Set defaults
OUTPUT_DIR="./qwen2_5_coder_output"
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
BATCH_SIZE=2
NUM_EPOCHS=3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-lora)
            USE_LORA=false
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Display configuration
echo "========================================="
echo "Qwen2.5-Coder-7B FastFormula Fine-tuning"
echo "========================================="
echo "Output Directory: $OUTPUT_DIR"
echo "LoRA: $USE_LORA"
if [ "$USE_LORA" = true ]; then
    echo "LoRA Rank: $LORA_R"
    echo "LoRA Alpha: $LORA_ALPHA"
fi
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "========================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
else
    echo "Warning: No NVIDIA GPU detected. Training will be slow on CPU."
fi

# Build command
CMD="python train_qwen2_5_coder.py --output_dir $OUTPUT_DIR --batch_size $BATCH_SIZE --num_epochs $NUM_EPOCHS"

if [ "$USE_LORA" = true ]; then
    CMD="$CMD --use_lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA"
fi

echo "Running: $CMD"
echo ""

# Execute
eval $CMD

echo ""
echo "Training completed! Model saved to: $OUTPUT_DIR"
echo "To test the model, run:"
echo "  python inference_qwen2_5_coder.py --model_path $OUTPUT_DIR"

