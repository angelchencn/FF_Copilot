# Quick Start Guide for Qwen2.5-Coder-7B FastFormula Fine-tuning

## Prerequisites

1. **Python 3.8+** 
2. **CUDA-capable GPU** (12GB+ VRAM recommended for LoRA)
3. **HuggingFace Account** (for downloading Qwen2.5-Coder-7B)

## Installation

```bash
cd qwen_fastformula
pip install -r requirements.txt
```

## Quick Start (3 Steps)

### 1. Verify Data

Make sure your training data is at `../data/fastformula/` in JSONL format:

```json
{"instruction": "Write a Fast Formula to calculate insurance_rate", "output": "DEFAULT FOR TOTAL_EARNINGS IS 0\n..."}
```

### 2. Run Training

#### Option A: Using the Shell Script (Easiest)

```bash
# Default settings (LoRA, recommended)
./run_training.sh

# Custom settings
./run_training.sh --output_dir ./my_output --batch_size 4 --epochs 5

# Full fine-tuning (requires more GPU memory)
./run_training.sh --output_dir ./full_output --no-lora
```

#### Option B: Using Python Directly

```bash
# LoRA fine-tuning (recommended)
python train_qwen2_5_coder.py --use_lora

# Full fine-tuning
python train_qwen2_5_coder.py --output_dir ./qwen2_5_coder_full_output

# Custom configuration
python train_qwen2_5_coder.py \
    --output_dir ./qwen2_5_coder_custom \
    --num_epochs 5 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --use_lora \
    --lora_r 32
```

### 3. Test Your Model

```bash
# Single query
python inference_qwen2_5_coder.py --model_path ./qwen2_5_coder_output --query "Write a formula to calculate overtime pay"

# Interactive mode
python inference_qwen2_5_coder.py --model_path ./qwen2_5_coder_output --interactive
```

## Expected Output

```
Qwen2.5-Coder-7B FastFormula Fine-tuning
============================================
Loading model: Qwen/Qwen2.5-Coder-7B
Added custom tokens: ['DEFAULT FOR', 'IF', 'THEN', ...]
Vocabulary size before: 152064
Vocabulary size after: 152104

Found 2 JSONL files: ['../data/fastformula/ff_payroll_train.jsonl', '../data/fastformula/ff_time_labor_train.jsonl']
Found 1500 Q&A pairs
Train samples: 1350, Val samples: 150

Setting up LoRA with r=16, alpha=32
Starting training...
============================================
...
[Training logs]
...
============================================
Training completed!
Model saved to: ./qwen2_5_coder_output
```

## Common Questions

### Q: How long does training take?
A: 
- ~2-3 hours on RTX 4080 (16GB) with LoRA
- ~4-6 hours on RTX 3080 (12GB) with LoRA
- ~8-12 hours on RTX 3060 (12GB) with LoRA
- ~12+ hours on CPU (not recommended)

### Q: What's the difference between LoRA and full fine-tuning?
A: 
- **LoRA**: Fine-tunes only adapter weights (50-100MB). Recommended for most cases.
- **Full**: Fine-tunes all weights (14GB). Better quality but needs more memory.

### Q: How much memory do I need?
A:
- LoRA: 12GB+ VRAM
- Full: 24GB+ VRAM

### Q: My training is too slow
A:
1. Reduce batch_size (e.g., 1 or 2)
2. Use LoRA (--use_lora)
3. Reduce max_length (e.g., 1024)
4. Use gradient_accumulation_steps to simulate larger batches

### Q: The model doesn't give good answers
A:
1. Check data quality
2. Train for more epochs (--num_epochs 5-10)
3. Use full fine-tuning instead of LoRA
4. Increase learning rate (e.g., 5e-4)

## Troubleshooting

### Error: Out of Memory
```bash
# Solution: Use smaller batch size + LoRA
python train_qwen2_5_coder.py --use_lora --batch_size 1 --gradient_accumulation_steps 16
```

### Error: "trust_remote_code=True" required
This is normal for Qwen2.5-Coder. The script already includes this flag.

### Error: CUDA out of memory
Try:
1. Add `--use_lora`
2. Reduce `--batch_size` to 1
3. Increase `--gradient_accumulation_steps` to maintain effective batch size

## Next Steps

After training, you can:
1. Integrate the model into your application
2. Further fine-tune on domain-specific data
3. Experiment with different hyperparameters
4. Export for deployment (e.g., ONNX format)

## Example Usage in Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./qwen2_5_coder_output")
tokenizer = AutoTokenizer.from_pretrained("./qwen2_5_coder_output")

# Ask a question
question = "Write a formula to calculate overtime pay"
prompt = f"User: {question}\n\nAssistant:"

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

