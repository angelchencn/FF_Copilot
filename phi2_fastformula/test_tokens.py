"""
Test script to verify custom tokens are added correctly
"""

from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Check padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Original vocabulary size: {len(tokenizer)}")

# Add custom tokens
new_tokens = ["WEEKDAY", "PAY_RATE", "OVERTIME_PAY", "FAST_FORMULA"]
num_added = tokenizer.add_tokens(new_tokens)

print(f"\nAdded {num_added} custom tokens: {new_tokens}")
print(f"New vocabulary size: {len(tokenizer)}")

# Test encoding
test_text = "This is a FAST_FORMULA for calculating PAY_RATE based on WEEKDAY"
encoded = tokenizer(test_text, return_tensors="pt")
print(f"\nTest text: '{test_text}'")
print(f"Encoded IDs: {encoded['input_ids'][0]}")
print(f"Decoded: {tokenizer.decode(encoded['input_ids'][0])}")

# Check if tokens exist in vocabulary
print("\nToken IDs for custom tokens:")
for token in new_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"  {token}: {token_id}")

