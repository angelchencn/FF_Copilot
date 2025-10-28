# Qwen2.5-Coder-7B FastFormula Fine-tuning

This directory contains scripts to fine-tune Qwen2.5-Coder-7B model on Oracle FastFormula Q&A data.

## Overview

This setup fine-tunes the Qwen2.5-Coder-7B language model on Oracle FastFormula documentation to create a specialized assistant for FastFormula-related questions and code generation.

## Files

- `train_qwen2_5_coder.py`: Main training script using HuggingFace Transformers
- `inference_qwen2_5_coder.py`: Script to test the fine-tuned model
- `run_training.sh`: Quick start shell script
- `config_finetune.py`: Configuration file
- `requirements.txt`: Python dependencies

## Model Details

- **Base Model**: Qwen2.5-Coder-7B (7B parameters)
- **Architecture**: Transformer decoder with code-specific training
- **Context Length**: 4096 tokens (configurable)
- **Specialization**: Code generation and understanding
- **Languages**: Multilingual support including English

## Setup

### 1. Install Dependencies

```bash
cd qwen_fastformula
pip install -r requirements.txt
```

### 2. Prepare Data

The data should be in JSONL format with `instruction`/`output` or `prompt`/`completion` fields.

Data is expected at: `../data/fastformula/` (directory with JSONL files)

## Training

### Basic Training (LoRA - Recommended)

```bash
# Quick start with LoRA
./run_training.sh

# Custom settings
./run_training.sh --output_dir ./my_output --batch_size 4 --epochs 5

# Full fine-tuning (requires more GPU memory)
./run_training.sh --output_dir ./full_output --no-lora
```

### Advanced Training Options

```bash
python train_qwen2_5_coder.py \
    --output_dir ./qwen2_5_coder_output \
    --num_epochs 5 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_length 4096 \
    --use_lora \
    --lora_r 32 \
    --save_steps 200 \
    --logging_steps 50
```

### Parameters

- `--model_name`: Model name/path (default: Qwen/Qwen2.5-Coder-7B)
- `--data_path`: Path to training data (default: ../data/fastformula)
- `--output_dir`: Output directory for checkpoints
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size per device (default: 2)
- `--gradient_accumulation_steps`: Steps for gradient accumulation (default: 8)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--max_length`: Maximum sequence length (default: 2048)
- `--use_lora`: Enable LoRA for efficient training
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)

## Testing the Model

After training, test the model with:

```bash
# Single query
python inference_qwen2_5_coder.py --model_path ./qwen2_5_coder_output --query "Write a formula to calculate overtime pay"

# Interactive mode
python inference_qwen2_5_coder.py --model_path ./qwen2_5_coder_output --interactive
```

## Expected Hardware Requirements

### LoRA Fine-tuning (Recommended):
- GPU: 12GB+ VRAM (RTX 3080, RTX 4070 Ti, RTX 4080)
- Model Size: ~7B parameters
- Memory: ~10-12GB for model + training

### Full Fine-tuning:
- GPU: 24GB+ VRAM (RTX 3090, RTX 4090, A100)
- Memory: ~20-24GB total

## Performance Comparison

| Model | Parameters | Memory (LoRA) | Memory (Full) | Speed | Code Quality |
|-------|------------|---------------|---------------|-------|--------------|
| Phi-2 | 2.7B | 8GB | 16GB | Fast | Good |
| Qwen2.5-Coder-7B | 7B | 12GB | 24GB | Medium | Excellent |

## Advantages of Qwen2.5-Coder-7B

1. **Code Specialization**: Trained specifically for code generation
2. **Better Context**: 4096 token context length
3. **Multilingual**: Better support for multiple languages
4. **Code Understanding**: Superior understanding of programming concepts
5. **Formula Generation**: Better at generating structured code/formulas

## Custom Tokens

The model includes FastFormula-specific tokens:
- `DEFAULT FOR`, `IF`, `THEN`, `ELSIF`, `ELSE`, `RETURN`
- `HOURS_WORKED`, `PAY_RATE`, `OVERTIME_PAY`, etc.
- `FORMULA_BEGIN`, `FORMULA_END` for structured output

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps` to 16
- Use `--use_lora` (recommended)
- Reduce `--max_length` to 1024

### Model Not Learning
- Increase learning rate to 5e-4
- Train for more epochs (5-10)
- Check data quality and format
- Verify custom tokens are being used

## Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./qwen2_5_coder_output")
tokenizer = AutoTokenizer.from_pretrained("./qwen2_5_coder_output")

question = "Write a formula to calculate bonus based on salary"
prompt = f"User: {question}\n\nAssistant:"

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## References

- [Qwen2.5-Coder Paper](https://arxiv.org/abs/2407.10671)
- [Qwen2.5-Coder Model](https://huggingface.co/Qwen/Qwen2.5-Coder-7B)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

