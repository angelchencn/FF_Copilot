# DeepSeek-Coder FastFormula Quick Start Guide

Get started with DeepSeek-Coder fine-tuning in 5 minutes!

## Prerequisites

- Python 3.8+
- CUDA-capable GPU with 12GB+ VRAM (recommended)
- Training data in JSONL format at `../data/fastformula/`

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd deepseek_fastformula
pip install -r requirements.txt
```

### Step 2: Start Training

```bash
# Quick start with default settings (LoRA enabled)
./run_training.sh

# Or with custom settings
./run_training.sh --epochs 5 --batch_size 4 --output_dir ./my_model
```

### Step 3: Test Your Model

```bash
# Interactive mode
python inference_deepseek_coder.py --model_path ./my_model --interactive

# Single query
python inference_deepseek_coder.py \
    --model_path ./my_model \
    --query "Write a Fast Formula to calculate overtime pay for hours over 40"
```

python inference_deepseek_coder.py \
    --model_path ./my_model \
    --query "Write a formula to calculate bonus" \
    --input "Base salary is SALARY, bonus rate is 10%"



## Training Options

### Basic Training (Recommended)
```bash
./run_training.sh
```
Default: 3 epochs, batch_size=2, LoRA enabled, ~1-2 hours

### Fast Training (Quick test)
```bash
./run_training.sh --epochs 1 --batch_size 4
```
1 epoch, larger batches if you have memory, ~20-30 minutes

### High Quality Training
```bash
./run_training.sh --epochs 5 --lora_r 32 --lora_alpha 64
```
More epochs and larger LoRA rank, ~2-4 hours

### Full Fine-tuning (Not recommended)
```bash
./run_training.sh --no-lora --batch_size 1
```
Requires 24GB+ VRAM, much slower

## Data Format

Your JSONL files should be in one of these formats:

**Format 1: instruction/output**
```json
{"instruction": "Write a formula to calculate overtime", "output": "DEFAULT FOR HOURS IS 0\n..."}
```

**Format 2: prompt/completion**
```json
{"prompt": "Calculate overtime pay", "completion": "IF HOURS > 40 THEN..."}
```

**Format 3: user/assistant**
```json
{"user": "How to calculate bonus?", "assistant": "BONUS = SALARY * RATE"}
```

## Common Commands

### Check GPU Status
```bash
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Monitor Training
```bash
# Training logs show:
# - Loss decreasing (good!)
# - Evaluation metrics
# - GPU memory usage
```

### Resume from Checkpoint
```bash
python train_deepseek_coder.py \
    --model_name ./deepseek_coder_fastformula_output/checkpoint-1000 \
    --output_dir ./resumed_training
```

## Troubleshooting

### Out of Memory
```bash
# Solution 1: Reduce batch size
./run_training.sh --batch_size 1

# Solution 2: Reduce sequence length
./run_training.sh --max_length 1024

# Solution 3: Ensure LoRA is enabled (it should be by default)
./run_training.sh --lora_r 8
```

### Slow Training
- Make sure you're using GPU: `nvidia-smi` should show Python process
- Close other applications using GPU
- Reduce logging frequency if needed

### Data Not Found
```bash
# Check your data path
ls -la ../data/fastformula/

# Or specify custom path
./run_training.sh --data_path /path/to/your/data
```

## Expected Results

After training, you should see:
- Training loss decreasing from ~3.0 to ~0.5-1.0
- Validation loss following similar trend
- Model saved in output directory
- Ready for inference!

## Next Steps

1. **Test the model** with various FastFormula questions
2. **Fine-tune hyperparameters** if needed (more epochs, different learning rate)
3. **Evaluate on your specific use cases**
4. **Deploy** the model for production use

## Tips for Best Results

1. **Quality Data**: More diverse, high-quality examples = better model
2. **Don't Overtrain**: Stop when validation loss stops decreasing
3. **Use LoRA**: Faster, uses less memory, prevents overfitting
4. **Monitor Progress**: Watch the training logs
5. **Test Early**: Try inference after 1 epoch to see progress

## Help

If you encounter issues:
1. Check the full README.md for detailed documentation
2. Review training logs for specific errors
3. Verify data format and paths
4. Check GPU availability and memory

## Example Session

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (grab a coffee, takes 1-2 hours)
./run_training.sh

# 3. Test
python inference_deepseek_coder.py \
    --model_path ./deepseek_coder_fastformula_output \
    --interactive

# 4. Ask questions!
User: Write a formula to calculate bonus based on performance
# Model generates FastFormula code...
```

Happy Training!
