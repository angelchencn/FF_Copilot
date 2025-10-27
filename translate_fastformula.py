"""
Translate Chinese questions in FastFormula Q&A data to English.
"""

import re

# Read the file
with open('data/fastformula/input.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Translation mapping for common Chinese questions
translations = {
    "这里解释了哪些关键概念？": "What key concepts are explained here?",
    "公式写法是什么？": "What is the formula syntax?",
    "这段讲了什么？": "What does this paragraph talk about?",
    "它与 Payroll 有什么关系？": "What is its relationship with Payroll?",
    "举个示例说明？": "Can you give an example to explain?",
}

# Replace Chinese questions with English translations
for chinese, english in translations.items():
    # Replace with "User: " prefix for consistency
    content = content.replace(f"用户: {chinese}", f"User: {english}")
    # Also handle variations without trailing question mark
    chinese_alt = chinese.replace("？", "")
    if chinese_alt in translations:
        content = content.replace(f"用户: {chinese_alt}", f"User: {english}")
    else:
        content = content.replace(f"用户: {chinese_alt}？", f"User: {english}")

# Also replace "助手:" with "Assistant:" for consistency
content = content.replace("助手:", "Assistant:")

# Save the translated file
with open('data/fastformula/input_en.txt', 'w', encoding='utf-8') as f:
    f.write(content)

print("Translation complete!")
print(f"Original file: data/fastformula/input.txt")
print(f"Translated file: data/fastformula/input_en.txt")
print(f"\nTranslations made:")
for chinese, english in translations.items():
    print(f"  '{chinese}' → '{english}'")

