"""
Inference script for fine-tuned Phi-2 model on FastFormula data.

Usage:
    python inference.py --model_path ./phi2_output --query "公式写法是什么？"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(question):
    """Format the question for the model"""
    return f"用户: {question}\n\n助手:"


def generate_response(model, tokenizer, question, max_new_tokens=120, temperature=0.8, top_p=0.9):
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
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "助手:" in response:
        response = response.split("助手:")[-1].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Phi-2 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--query", type=str, default="公式写法是什么？", help="Question to ask")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phi-2 FastFormula Inference")
    print("=" * 60)
    
    # Load model and tokenizer
    print(f"\nLoading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
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
            question = input("\n用户: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\n助手: ", end="")
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


