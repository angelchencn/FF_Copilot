"""
Inference script for fine-tuned Qwen2.5-Coder-7B model on FastFormula data.

Usage:
    python inference_qwen2_5_coder.py --model_path ./qwen2_5_coder_output --query "What is the formula syntax?"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(question):
    """Format the question for the model"""
    return f"User: {question}\n\nAssistant:"


def generate_response(model, tokenizer, question, max_new_tokens=200, temperature=0.3, top_p=0.9):
    """Generate a response from the model"""
    # Format the prompt
    prompt = format_prompt(question)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen2.5-Coder-7B model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--query", type=str, default="What is the formula syntax?", help="Question to ask")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Qwen2.5-Coder-7B FastFormula Inference")
    print("=" * 60)
    
    # Load model and tokenizer
    print(f"\nLoading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print("Model loaded successfully!\n")
    
    if args.interactive:
        # Interactive mode
        print("Enter interactive mode. Type 'quit' to exit.")
        print("-" * 60)
        
        while True:
            question = input("\nUser: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\nAssistant: ", end="", flush=True)
            response = generate_response(
                model, tokenizer, question,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            print(response)
    else:
        # Single query mode
        print(f"Question: {args.query}\n")
        print("Answer:", end=" ")
        
        response = generate_response(
            model, tokenizer, args.query,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(response)


if __name__ == "__main__":
    main()

