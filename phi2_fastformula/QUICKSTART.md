# Quick Start Guide for Phi-2 FastFormula Fine-tuning

## Prerequisites

1. **Python 3.8+** 
2. **CUDA-capable GPU** (8GB+ VRAM recommended)
3. **HuggingFace Account** (for downloading Phi-2)

## Installation

```bash
cd phi2_fastformula
pip install -r requirements.txt
```

## Quick Start (3 Steps)

### 1. Verify Data

Make sure your training data is at `../data/fastformula/input.txt` in the correct format:

```
用户: 这里解释了哪些关键概念？
助手: [Oracle FastFormula相关回答]
```

### 2. Run Training

#### Option A: Using the Shell Script (Easiest)

```bash
# Default settings (LoRA, recommended)
./run_training.sh

# Custom settings
./run_training.sh --output_dir ./my_output --batch_size 8 --epochs 5

# Full fine-tuning (requires more GPU memory)
./run_training.sh --output_dir ./full_output --no-lora
```

#### Option B: Using Python Directly

```bash
# LoRA fine-tuning (recommended)
python train_phi2.py --use_lora

# Full fine-tuning
python train_phi2.py --output_dir ./phi2_full_output

# Custom configuration
python train_phi2.py \
    --output_dir ./phi2_custom \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --use_lora \
    --lora_r 16
```

### 3. Test Your Model

```bash
# Single query
python inference.py --model_path ./phi2_output --query "公式写法是什么？"

# Interactive mode
python inference.py --model_path ./phi2_output --interactive
```

## Expected Output

```
Phi-2 FastFormula Fine-tuning
============================================
Loading model: microsoft/phi-2
Model loaded: PhiConfig(...)
Total parameters: 2,776,200,704

Found 150 Q&A pairs
Train samples: 135, Val samples: 15

Tokenizing data...
Starting training...
============================================
...
[Training logs]
...
============================================
Training completed!
Model saved to: ./phi2_output
```

## Common Questions

### Q: How long does training take?
A: 
- ~1-2 hours on RTX 4090 (24GB) with LoRA
- ~3-4 hours on RTX 3060 (12GB) with LoRA
- ~6-8 hours on CPU (not recommended)

### Q: What's the difference between LoRA and full fine-tuning?
A: 
- **LoRA**: Fine-tunes only adapter weights (20-50MB). Recommended for most cases.
- **Full**: Fine-tunes all weights (5.5GB). Better quality but needs more memory.

### Q: How much memory do I need?
A:
- LoRA: 8GB+ VRAM
- Full: 16GB+ VRAM

### Q: My training is too slow
A:
1. Reduce batch_size (e.g., 2 or 4)
2. Use LoRA (--use_lora)
3. Reduce max_length (e.g., 512)
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
python train_phi2.py --use_lora --batch_size 1 --gradient_accumulation_steps 8
```

### Error: "trust_remote_code=True" required
This is normal for Phi-2. The script already includes this flag.

### Error: CUDA out of memory
Try:
1. Add `--use_lora`
2. Reduce `--batch_size` to 1 or 2
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
model = AutoModelForCausalLM.from_pretrained("./phi2_output")
tokenizer = AutoTokenizer.from_pretrained("./phi2_output")

# Ask a question
question = "公式写法是什么？"
prompt = f"用户: {question}\n\n助手:"

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```


