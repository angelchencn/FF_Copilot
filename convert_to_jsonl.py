"""
Convert Q&A format to JSONL format for training.
"""

import json
import re

def convert_to_jsonl(input_file, output_file):
    """Convert User/Assistant Q&A format to JSONL format."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into Q&A pairs
    # Pattern: User: ... Assistant: ... (with optional blank lines)
    pattern = r'User: (.*?)\nAssistant: (.*?)(?=\nUser: |$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    jsonl_lines = []
    
    for question, answer in matches:
        # Clean up whitespace
        question = question.strip()
        answer = answer.strip()
        
        # Skip empty questions or answers
        if not question or not answer:
            continue
        
        # Create JSON object
        json_obj = {
            "prompt": question,
            "completion": answer
        }
        
        # Convert to JSON string
        json_str = json.dumps(json_obj, ensure_ascii=False)
        jsonl_lines.append(json_str)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in jsonl_lines:
            f.write(line + '\n')
    
    print(f"Converted {len(jsonl_lines)} Q&A pairs to JSONL format")
    print(f"Output written to: {output_file}")
    
    # Show first few examples
    if jsonl_lines:
        print("\nFirst 3 examples:")
        for i, line in enumerate(jsonl_lines[:3], 1):
            print(f"\n{i}. {line}")

if __name__ == "__main__":
    input_file = 'data/fastformula/input.txt'
    output_file = 'data/fastformula/input.jsonl'
    
    convert_to_jsonl(input_file, output_file)

