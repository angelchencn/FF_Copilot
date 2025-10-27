"""
Fine-tune Microsoft Phi-2 model on Oracle FastFormula Q&A data using HuggingFace Transformers.

Usage:
    python train_phi2.py --output_dir ./phi2_fastformula_output
"""

import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-2 on FastFormula data")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2", help="Model name or path")
    parser.add_argument("--data_path", type=str, default="../data/fastformula", help="Path to training data (file or directory)")
    parser.add_argument("--output_dir", type=str, default="./phi2_fastformula_output", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=250, help="Evaluate every N steps")
    
    return parser.parse_args()


def format_chat_template(prompt, response, tokenizer):
    """Format prompt-response pair for training"""
    # Format as a conversation: User: <prompt>\n\nAssistant: <response>
    text = f"用户: {prompt}\n\n助手: {response}"
    return text


def preprocess_function(examples, tokenizer, max_length=1024):
    """Preprocess the data for training"""
    texts = []
    for i in range(len(examples['user'])):
        text = format_chat_template(examples['user'][i], examples['assistant'][i], tokenizer)
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return tokenized


def parse_jsonl_file(file_path):
    """Parse a single JSONL file (supports both 'prompt/completion' and 'instruction/output' formats)"""
    import json
    
    qa_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Support multiple field name formats
                if 'prompt' in data and 'completion' in data:
                    qa_pairs.append({'user': data['prompt'], 'assistant': data['completion']})
                elif 'instruction' in data and 'output' in data:
                    qa_pairs.append({'user': data['instruction'], 'assistant': data['output']})
                elif 'user' in data and 'assistant' in data:
                    qa_pairs.append({'user': data['user'], 'assistant': data['assistant']})
            except json.JSONDecodeError:
                continue
    
    return qa_pairs


def parse_qa_data(data_path):
    """Parse Q&A data from file or directory (supports JSONL format)"""
    import json
    import os
    from pathlib import Path
    
    qa_pairs = []
    
    # Check if path is a directory or file
    if os.path.isdir(data_path):
        # Load all JSONL files in the directory
        print(f"Loading JSONL files from directory: {data_path}")
        jsonl_files = []
        for file in os.listdir(data_path):
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(data_path, file))
        
        if not jsonl_files:
            raise ValueError(f"No JSONL files found in {data_path}")
        
        print(f"Found {len(jsonl_files)} JSONL files: {jsonl_files}")
        
        # Load all files
        for file_path in jsonl_files:
            print(f"Loading: {file_path}")
            pairs = parse_jsonl_file(file_path)
            qa_pairs.extend(pairs)
            print(f"  - Loaded {len(pairs)} Q&A pairs")
        
        return qa_pairs
    
    # Otherwise, treat as single file
    file_path = data_path
    print(f"Loading data from file: {file_path}")
    
    # Check if file is JSONL format
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        
        # Try to parse as JSONL
        try:
            json.loads(first_line)
            # File is JSONL format
            f.seek(0)  # Reset to beginning
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Support multiple field name formats
                    if 'prompt' in data and 'completion' in data:
                        qa_pairs.append({'user': data['prompt'], 'assistant': data['completion']})
                    elif 'instruction' in data and 'output' in data:
                        qa_pairs.append({'user': data['instruction'], 'assistant': data['output']})
                    elif 'user' in data and 'assistant' in data:
                        qa_pairs.append({'user': data['user'], 'assistant': data['assistant']})
                except json.JSONDecodeError:
                    continue
            return qa_pairs
        except json.JSONDecodeError:
            pass
    
    # Fallback to old format parsing (User: / Assistant:)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    current_q = None
    current_a = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('User:') or line.startswith('用户:'):
            prefix = 'User:' if line.startswith('User:') else '用户:'
            # Save previous Q&A if exists
            if current_q and current_a:
                qa_pairs.append({'user': current_q, 'assistant': current_a})
            current_q = line[len(prefix):].strip()
            current_a = None
        elif (line.startswith('Assistant:') or line.startswith('助手:')) and current_q:
            prefix = 'Assistant:' if line.startswith('Assistant:') else '助手:'
            current_a = line[len(prefix):].strip()
    
    # Don't forget the last pair
    if current_q and current_a:
        qa_pairs.append({'user': current_q, 'assistant': current_a})
    
    return qa_pairs


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Phi-2 FastFormula Fine-tuning")
    print("=" * 60)
    
    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add custom tokens for FastFormula domain
    new_tokens = ["DEFAULT FOR", "IF", "THEN", "ELSIF", "ELSE", "RETURN",
                    "HOURS_WORKED", "REGULAR_HOURS", "TOTAL_HOURS",
                    "PAY_RATE", "HOURLY_RATE", "OVERTIME_RATE", "OVERTIME_HOURS", "OVERTIME_PAY",
                    "GROSS_PAY", "BASE_SALARY", "TAXABLE_PAY", "BONUS_AMOUNT",
                    "TAX_RATE", "ALLOWANCE_RATE", "DEDUCTION_RATE", "BONUS_PERCENT",
                    "INPUT_VALUE", "THRESHOLD", "LIMIT", "AMOUNT",
                    "MESSAGE", "RESULT", "VALUE", "FACTOR", "RATE",
                    "WEEKDAY", "SUNDAY", "MONDAY",
                    "### START", "### END", "FORMULA_BEGIN", "FORMULA_END"]
    tokenizer.add_tokens(new_tokens)
    print(f"Added custom tokens: {new_tokens}")
    print(f"Vocabulary size before: {len(tokenizer)}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Resize token embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    print(f"Vocabulary size after: {len(tokenizer)}")
    print(f"Model embedding layer resized to accommodate new tokens")
    
    print(f"Model loaded: {model.config}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup LoRA if requested
    if args.use_lora:
        print(f"\nSetting up LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "dense"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Prepare dataset
    print(f"\nLoading and preprocessing data from: {args.data_path}")
    qa_pairs = parse_qa_data(args.data_path)
    print(f"Found {len(qa_pairs)} Q&A pairs")
    
    # Convert to dataset format
    dataset_dict = {
        'user': [pair['user'] for pair in qa_pairs],
        'assistant': [pair['assistant'] for pair in qa_pairs]
    }
    
    # Create train/val split (90/10)
    split_idx = int(len(qa_pairs) * 0.9)
    train_data = {k: v[:split_idx] for k, v in dataset_dict.items()}
    val_data = {k: v[split_idx:] for k, v in dataset_dict.items()}
    
    print(f"Train samples: {len(train_data['user'])}, Val samples: {len(val_data['user'])}")
    
    # Format texts for training
    train_texts = [format_chat_template(q, a, tokenizer) for q, a in zip(train_data['user'], train_data['assistant'])]
    val_texts = [format_chat_template(q, a, tokenizer) for q, a in zip(val_data['user'], val_data['assistant'])]
    
    # Tokenize
    print("Tokenizing data...")
    train_tokenized = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    
    val_tokenized = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt"
    )
    
    # Convert to torch datasets
    class FastFormulaDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __getitem__(self, idx):
            return {key: tensor[idx] for key, tensor in self.encodings.items()}
        
        def __len__(self):
            return len(self.encodings['input_ids'])
    
    train_dataset = FastFormulaDataset(train_tokenized)
    val_dataset = FastFormulaDataset(val_tokenized)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16,
        bf16=bf16,
        report_to="none",  # Can change to "wandb" or "tensorboard" if desired
        save_safetensors=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Model saved to: {args.output_dir}")
    

if __name__ == "__main__":
    main()

