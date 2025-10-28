"""
Inference script for fine-tuned DeepSeek-Coder model on FastFormula data.

Usage:
    # Single query
    python inference_deepseek_coder.py --model_path ./deepseek_coder_output --query "Write a formula to calculate overtime pay"

    # Interactive mode
    python inference_deepseek_coder.py --model_path ./deepseek_coder_output --interactive
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(question):
    """Format the question for the model"""
    return f"### Instruction:\n{question}\n\n### Response:\n"


def generate_response(model, tokenizer, question, max_new_tokens=512, temperature=0.7, top_p=0.9):
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
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned DeepSeek-Coder model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--query", type=str, default="Write a Fast Formula to calculate overtime pay?", help="Question to ask")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0-1.0)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    print("=" * 60)
    print("DeepSeek-Coder FastFormula Inference")
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
    print("Model loaded successfully!")
    print(f"Device: {model.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    if args.interactive:
        # Interactive mode
        print("Enter interactive mode. Type 'quit', 'exit', or 'q' to exit.")
        print("-" * 60)

        while True:
            try:
                question = input("\nUser: ")
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    break

                if not question.strip():
                    continue

                print("\nAssistant: ", end="", flush=True)
                response = generate_response(
                    model, tokenizer, question,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                print(response)
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue
    else:
        # Single query mode
        print(f"Question: {args.query}\n")
        print("Answer:")
        print("-" * 60)

        response = generate_response(
            model, tokenizer, args.query,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(response)
        print("-" * 60)


if __name__ == "__main__":
    main()
