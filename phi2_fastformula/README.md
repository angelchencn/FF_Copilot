# Phi-2 FastFormula Fine-tuning

This directory contains scripts to fine-tune Microsoft's Phi-2 model on Oracle FastFormula Q&A data.

## Overview

This setup fine-tunes the Phi-2 language model on Oracle FastFormula documentation to create a specialized assistant for FastFormula-related questions.

## Files

- `train_phi2.py`: Main training script using HuggingFace Transformers
- `inference.py`: Script to test the fine-tuned model
- `requirements.txt`: Python dependencies

## Setup

### 1. Install Dependencies

```bash
cd phi2_fastformula
pip install -r requirements.txt
```

### 2. Prepare Data

The data should be in the format:
```
用户: <question>
助手: <answer>
```

Data is expected at: `../data/fastformula/input.txt`

## Training

### Basic Training (Full Fine-tuning)

```bash
python train_phi2.py --output_dir ./phi2_output
```

### LoRA Training (Parameter-Efficient)

Recommended for limited GPU memory:

```bash
python train_phi2.py \
    --output_dir ./phi2_lora_output \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16
```

### Advanced Training Options

```bash
python train_phi2.py \
    --output_dir ./phi2_output \
    --num_epochs 5 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --max_length 2048 \
    --use_lora \
    --lora_r 16 \
    --save_steps 200 \
    --logging_steps 50
```

### Parameters

- `--model_name`: Model name/path (default: microsoft/phi-2)
- `--data_path`: Path to training data (default: ../data/fastformula/input.txt)
- `--output_dir`: Output directory for checkpoints
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size per device (default: 4)
- `--gradient_accumulation_steps`: Steps for gradient accumulation (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--max_length`: Maximum sequence length (default: 1024)
- `--use_lora`: Enable LoRA for efficient training
- `--lora_r`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--save_steps`: Save checkpoint every N steps
- `--logging_steps`: Log every N steps
- `--eval_steps`: Evaluate every N steps

## Testing the Model

After training, test the model with:

```bash
python inference.py --model_path ./phi2_output
```

## Expected Hardware Requirements

### Full Fine-tuning:
- GPU: 16GB+ VRAM (A100, V100, RTX 3090/4090)
- Model Size: ~2.7B parameters
- Memory: ~12GB for model + ~4GB for training

### LoRA Fine-tuning:
- GPU: 8GB+ VRAM (RTX 3060, RTX 4060)
- Reduced Memory: ~6-8GB total

## Model Details

- **Base Model**: microsoft/phi-2 (2.7B parameters)
- **Architecture**: Transformer decoder
- **Context Length**: 2048 tokens (configurable)
- **Training Format**: Question-Answer pairs in Chinese

## Notes

- The model will save checkpoints periodically during training
- Best model is automatically selected based on validation loss
- LoRA training is recommended for most users due to lower memory requirements
- Training typically takes 1-3 hours on a modern GPU depending on data size and epochs

## Troubleshooting

### Out of Memory
- Reduce `batch_size` (try 1 or 2)
- Increase `gradient_accumulation_steps` to compensate
- Use `--use_lora` for memory-efficient training
- Reduce `--max_length`

### Model Not Learning
- Increase learning rate (try 5e-4)
- Check data format
- Verify data quality and size
- Increase training epochs

## References

- [Phi-2 Paper](https://arxiv.org/abs/2310.05213)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)


