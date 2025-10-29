"""
Inference script for fine-tuned DeepSeek-Coder model on FastFormula data.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def load_prompt_template(template_path="inference_prompt_template.txt"):
    """Load prompt template from file"""
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def format_prompt(question, input_context="", use_template=True, template_path="inference_prompt_template.txt"):
    """Format the question for the model using template"""
    if use_template:
        template = load_prompt_template(template_path)
        if template:
            # Replace placeholders in template
            prompt = template.replace("{{instruction}}", question)
            prompt = prompt.replace("{{input}}", input_context if input_context else "N/A")
            return prompt
    
    # Fallback to simple format
    return f"User: {question}\n\nAssistant:"


def generate_response(model, tokenizer, question, input_context="", use_template=True, 
                      template_path="inference_prompt_template.txt", max_new_tokens=200, 
                      temperature=0.3, top_p=0.9):
    """Generate a response from the model"""
    # Format the prompt
    prompt = format_prompt(question, input_context, use_template, template_path)
    
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
    
    # Extract only the output part (remove the prompt)
    if use_template and "[Expected Output]" in response:
        # Try to extract content after the template
        parts = response.split("[Expected Output]")
        if len(parts) > 1:
            response = parts[-1].strip()
    elif "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned DeepSeek-Coder model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--query", type=str, default="What is the formula syntax?", help="Question to ask")
    parser.add_argument("--input", type=str, default="", help="Optional context/input")
    parser.add_argument("--template_path", type=str, default="inference_prompt_template.txt", help="Path to prompt template")
    parser.add_argument("--no_template", action="store_true", help="Don't use prompt template")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DeepSeek-Coder FastFormula Inference")
    print("=" * 60)
    
    # Load model and tokenizer
    print(f"\nLoading model from: {args.model_path}")
    
    # Check if this is a PEFT adapter
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        # This is a PEFT adapter, load base model first
        print("Detected PEFT adapter, loading base model first...")
        peft_config = PeftConfig.from_pretrained(args.model_path)
        base_model_path = peft_config.base_model_name_or_path
        
        print(f"Loading base model from: {base_model_path}")
        
        # Load tokenizer from the adapter path (which has the correct vocab size)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto",
        )
        
        # Resize model embeddings to match the tokenizer used during training
        print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        
        print(f"Loading adapter from: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        print("Merging adapter with base model...")
        model = model.merge_and_unload()  # Merge adapter to base model
    else:
        # Not a PEFT adapter, load directly
        print("Loading model directly...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("Model loaded successfully!\n")
    
    use_template = not args.no_template
    
    if args.interactive:
        # Interactive mode
        print("Enter interactive mode. Type 'quit' to exit.")
        if use_template:
            print(f"Using prompt template: {args.template_path}")
        print("-" * 60)
        
        while True:
            question = input("\nUser: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            # Optionally ask for context
            context = input("Context (optional, press Enter to skip): ").strip()
            
            print("\nAssistant: ", end="", flush=True)
            response = generate_response(
                model, tokenizer, question,
                input_context=context,
                use_template=use_template,
                template_path=args.template_path,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            print(response)
    else:
        # Single query mode
        print(f"Question: {args.query}")
        if args.input:
            print(f"Context: {args.input}")
        print()
        print("Answer:", end=" ")
        
        response = generate_response(
            model, tokenizer, args.query,
            input_context=args.input,
            use_template=use_template,
            template_path=args.template_path,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(response)


if __name__ == "__main__":
    main()
