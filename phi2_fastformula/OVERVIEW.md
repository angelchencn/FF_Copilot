# Phi-2 FastFormula Fine-tuning Overview

## What Was Created

A complete fine-tuning setup for Microsoft's Phi-2 model on Oracle FastFormula Q&A data.

### Directory Structure

```
phi2_fastformula/
├── train_phi2.py          # Main training script
├── inference.py            # Model inference script  
├── config_finetune.py      # Configuration file
├── requirements.txt        # Python dependencies
├── run_training.sh         # Quick start shell script
├── README.md              # Full documentation
├── QUICKSTART.md          # Quick start guide
└── OVERVIEW.md            # This file
```

## Key Features

### 1. Training Script (`train_phi2.py`)

- **Base Model**: Microsoft Phi-2 (2.7B parameters)
- **Data Format**: Q&A pairs in Chinese about Oracle FastFormula
- **Training Modes**:
  - Full fine-tuning (all weights)
  - LoRA fine-tuning (parameter-efficient)
- **Features**:
  - Automatic train/val split (90/10)
  - Checkpoint saving
  - Validation monitoring
  - FP16/BF16 mixed precision
  - Gradient accumulation
  - Cosine learning rate decay

### 2. Model Architecture

- **Model**: Phi-2 (Transformer decoder)
- **Parameters**: 2.7 billion
- **Context Length**: 2048 tokens
- **Vocabulary**: 51,200 tokens
- **Architecture**: Attention mechanism with RoPE

### 3. Data

- **Source**: Oracle FastFormula documentation
- **Format**: User-Assistant dialogue pairs
- **Language**: Chinese (Simplified)
- **Domain**: Oracle HCM FastFormula

### 4. Fine-tuning Approaches

#### LoRA (Recommended)
- **Rank**: 8 (configurable)
- **Alpha**: 16 (configurable)
- **Target Modules**: Q, K, V projections
- **Memory**: ~8GB GPU
- **Speed**: Faster training

#### Full Fine-tuning
- **Memory**: ~16GB GPU
- **Quality**: Better adaptation
- **Time**: Slower training

## Usage Examples

### Training

```bash
# Quick start with LoRA
./run_training.sh

# Full fine-tuning
python train_phi2.py --output_dir ./phi2_full

# Custom configuration
python train_phi2.py \
    --num_epochs 5 \
    --batch_size 8 \
    --use_lora \
    --lora_r 16 \
    --output_dir ./phi2_custom
```

### Inference

```bash
# Single query
python inference.py --model_path ./phi2_output --query "公式怎么写？"

# Interactive mode
python inference.py --model_path ./phi2_output --interactive
```

## Technical Details

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| epochs | 3 | Training epochs |
| batch_size | 4 | Batch per device |
| grad_accum | 4 | Gradient accumulation |
| lr | 2e-4 | Learning rate |
| max_length | 1024 | Sequence length |
| lora_r | 8 | LoRA rank |
| lora_alpha | 16 | LoRA alpha |

### Training Process

1. **Data Loading**: Parse Q&A pairs from input.txt
2. **Tokenization**: Convert text to model tokens
3. **Splitting**: 90% train, 10% validation
4. **Batching**: Create batches with specified size
5. **Training**: 
   - Forward pass
   - Loss calculation
   - Backward pass
   - Gradient clipping
   - Optimizer step
6. **Evaluation**: Compute validation loss
7. **Checkpointing**: Save best model

### Expected Results

After training, the model should:
- Understand FastFormula terminology
- Answer questions about formula syntax
- Provide relevant examples
- Explain Oracle HCM concepts

## Performance Benchmarks

### Training Speed
- RTX 4090 (24GB): ~1-2 hours (LoRA), ~3-4 hours (Full)
- RTX 3060 (12GB): ~3-4 hours (LoRA), ~12+ hours (Full)
- CPU: Not practical

### Memory Usage
- LoRA: ~8GB VRAM
- Full: ~16GB VRAM

### Model Size
- Base Phi-2: ~5.5GB
- LoRA adapters: ~20-50MB
- Fine-tuned (Full): ~5.5GB

## Integration

### Using in Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./phi2_output")
tokenizer = AutoTokenizer.from_pretrained("./phi2_output")

question = "公式写法是什么？"
inputs = tokenizer(f"用户: {question}\n\n助手:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

### API Integration

The fine-tuned model can be:
- Deployed with FastAPI
- Used in Oracle applications
- Integrated into chatbots
- Exported to ONNX

## Best Practices

1. **Use LoRA** for most cases (faster, less memory)
2. **Monitor validation loss** to avoid overfitting
3. **Start with 3 epochs**, increase if needed
4. **Use gradient accumulation** for larger effective batch size
5. **Save checkpoints** frequently
6. **Test regularly** during training

## Troubleshooting

### Common Issues

**Out of Memory**
- Use LoRA: `--use_lora`
- Reduce batch size: `--batch_size 1`
- Increase gradient accumulation

**Poor Quality**
- Increase epochs
- Use full fine-tuning
- Check data quality
- Adjust learning rate

**Slow Training**
- Use LoRA
- Reduce max_length
- Ensure GPU is being used
- Check data loading speed

## References

- [Phi-2 Model](https://huggingface.co/microsoft/phi-2)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [HuggingFace Documentation](https://huggingface.co/docs/transformers)
- [Oracle FastFormula Guide](https://docs.oracle.com)

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Check QUICKSTART.md for common scenarios
3. Review training logs for error messages
4. Verify data format matches requirements

## License

Same as parent project. Fine-tuned model inherits Phi-2's licensing terms.

