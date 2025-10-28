# DeepSeek-Coder FastFormula Fine-tuning

This directory contains scripts to fine-tune DeepSeek-Coder model on Oracle FastFormula Q&A data.

## Overview

This setup fine-tunes the DeepSeek-Coder-6.7B language model on Oracle FastFormula documentation to create a specialized assistant for FastFormula code generation and understanding.

## Files

- `train_deepseek_coder.py`: Main training script using HuggingFace Transformers
- `inference_deepseek_coder.py`: Script to test the fine-tuned model
- `run_training.sh`: Quick start shell script
- `config_finetune.py`: Configuration file
- `requirements.txt`: Python dependencies

## Model Details

- **Base Model**: deepseek-ai/deepseek-coder-6.7b-base (6.7B parameters)
- **Architecture**: Transformer decoder optimized for code
- **Context Length**: 16K tokens (supports up to 16384 tokens)
- **Specialization**: Code generation, understanding, and completion
- **Training**: Pre-trained on 2T tokens (87% code, 13% natural language)
- **Languages**: 86+ programming languages

## Why DeepSeek-Coder?

DeepSeek-Coder is specifically designed for code-related tasks and offers several advantages:

1. **Code-Specialized**: Pre-trained extensively on code data
2. **Long Context**: 16K token context window for complex formulas
3. **High Performance**: Outperforms many larger models on code tasks
4. **Efficient**: 6.7B parameters with excellent performance
5. **Fill-in-the-Middle**: Supports code completion and infilling
6. **Multi-language**: Excellent support for domain-specific languages

## Setup

### 1. Install Dependencies

```bash
cd deepseek_fastformula
pip install -r requirements.txt
```

### 2. Prepare Data

The data should be in JSONL format with `instruction`/`output` or `prompt`/`completion` fields.

Data is expected at: `../data/fastformula/` (directory with JSONL files)

Example data format:
```json
{"instruction": "Write a Fast Formula to calculate overtime pay", "output": "DEFAULT FOR HOURS_WORKED IS 0\n..."}
```

## Training

### Quick Start with LoRA (Recommended)

```bash
# Using the shell script
./run_training.sh

# Or directly with Python
python train_deepseek_coder.py \
    --output_dir ./deepseek_coder_output \
    --use_lora \
    --num_epochs 3
```

### Advanced Training Options

```bash
python train_deepseek_coder.py \
    --model_name deepseek-ai/deepseek-coder-6.7b-base \
    --data_path ../data/fastformula \
    --output_dir ./deepseek_coder_output \
    --num_epochs 5 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_length 2048 \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --save_steps 200 \
    --logging_steps 50
```

### Training Parameters

- `--model_name`: Model name/path (default: deepseek-ai/deepseek-coder-6.7b-base)
- `--data_path`: Path to training data (default: ../data/fastformula)
- `--output_dir`: Output directory for checkpoints
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size per device (default: 2)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--max_length`: Maximum sequence length (default: 2048)
- `--use_lora`: Enable LoRA (highly recommended)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)

## Testing the Model

### Single Query

```bash
python inference_deepseek_coder.py \
    --model_path ./deepseek_coder_output \
    --query "Write a Fast Formula to calculate overtime pay when hours exceed 40"
```

### Interactive Mode

```bash
python inference_deepseek_coder.py \
    --model_path ./deepseek_coder_output \
    --interactive
```

### Custom Generation Parameters

```bash
python inference_deepseek_coder.py \
    --model_path ./deepseek_coder_output \
    --query "Your question here" \
    --max_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9
```

## Hardware Requirements

### LoRA Fine-tuning (Recommended):
- **GPU**: 12GB+ VRAM (RTX 3080, RTX 4070 Ti, RTX 4080, L4)
- **Model Size**: 6.7B parameters
- **Training Memory**: ~10-12GB
- **Inference Memory**: ~8GB

### Full Fine-tuning:
- **GPU**: 24GB+ VRAM (RTX 3090, RTX 4090, A100)
- **Training Memory**: ~20-22GB
- **Not recommended**: Use LoRA instead

## Performance Comparison

| Model | Parameters | Context | Memory (LoRA) | Speed | Code Quality | Specialization |
|-------|------------|---------|---------------|-------|--------------|----------------|
| Phi-2 | 2.7B | 2K | 8GB | Fast | Good | General |
| DeepSeek-Coder | 6.7B | 16K | 12GB | Medium | Excellent | Code |
| Qwen2.5-Coder | 7B | 4K | 12GB | Medium | Excellent | Code |

## Advantages of DeepSeek-Coder

1. **Long Context Window**: 16K tokens vs 2-4K for other models
2. **Code-First Design**: Specifically trained for code generation
3. **Fill-in-the-Middle**: Can complete code snippets anywhere
4. **Strong Reasoning**: Better at understanding complex logic
5. **Cost-Effective**: Smaller size with comparable performance
6. **Production Ready**: Extensively tested on code tasks

## Best Practices

### For Best Results:
1. Use LoRA training to save memory and prevent overfitting
2. Train for 3-5 epochs with early stopping
3. Use gradient checkpointing for memory efficiency
4. Start with learning rate 2e-4 and adjust if needed
5. Monitor validation loss to prevent overfitting

### Data Preparation:
1. Ensure JSONL format with clear instruction/output pairs
2. Include diverse examples of FastFormula code
3. Mix simple and complex formulas
4. Include explanations and documentation
5. Use consistent formatting

## Custom Tokens

The model includes FastFormula-specific tokens for better performance:
- Control flow: `DEFAULT FOR`, `IF`, `THEN`, `ELSIF`, `ELSE`, `RETURN`
- Variables: `HOURS_WORKED`, `PAY_RATE`, `OVERTIME_PAY`, etc.
- Delimiters: `### START`, `### END`, `FORMULA_BEGIN`, `FORMULA_END`

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python train_deepseek_coder.py --batch_size 1 --gradient_accumulation_steps 16 --use_lora

# Reduce sequence length
python train_deepseek_coder.py --max_length 1024 --use_lora

# Enable gradient checkpointing (already enabled by default)
```

### Slow Training
- Ensure you're using GPU (check with `nvidia-smi`)
- Use bfloat16 if available (automatic on supported GPUs)
- Reduce logging frequency: `--logging_steps 50`
- Increase batch size if you have memory: `--batch_size 4`

### Poor Model Quality
- Train for more epochs: `--num_epochs 5`
- Increase LoRA rank: `--lora_r 32 --lora_alpha 64`
- Lower learning rate: `--learning_rate 1e-4`
- Check data quality and quantity
- Ensure proper data formatting

### Model Not Loading
- Check model path is correct
- Ensure all files are saved (model, tokenizer, config)
- Try loading without `device_map="auto"`
- Check CUDA availability with `torch.cuda.is_available()`

## Example Usage in Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model_path = "./deepseek_coder_output"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate
question = "Write a Fast Formula to calculate bonus based on performance rating"
prompt = f"### Instruction:\n{question}\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("### Response:")[-1].strip())
```

## Training Time Estimates

- **Small dataset (1-2K samples)**: 30-60 minutes with LoRA
- **Medium dataset (2-5K samples)**: 1-2 hours with LoRA
- **Large dataset (5K+ samples)**: 2-4 hours with LoRA

Time estimates based on RTX 4080 GPU with batch_size=2, gradient_accumulation=8

## Model Checkpoints

During training, checkpoints are saved:
- Every `save_steps` (default: 500)
- At the end of each epoch
- Best model based on validation loss

To resume from a checkpoint:
```bash
python train_deepseek_coder.py \
    --model_name ./deepseek_coder_output/checkpoint-1000 \
    --output_dir ./deepseek_coder_output_resumed
```

## References

- [DeepSeek-Coder Paper](https://arxiv.org/abs/2401.14196)
- [DeepSeek-Coder GitHub](https://github.com/deepseek-ai/DeepSeek-Coder)
- [DeepSeek-Coder Model Card](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Oracle FastFormula Documentation](https://docs.oracle.com/en/cloud/saas/human-resources/25a/fafor/index.html)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the training logs for errors
3. Verify data format and paths
4. Check GPU memory and availability

## License

This project uses the DeepSeek-Coder model which is licensed under the DeepSeek License.
Please review the model's license before commercial use.
