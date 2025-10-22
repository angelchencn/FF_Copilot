import os
import re
import json
import random
from tqdm import tqdm
from pypdf import PdfReader
import subprocess

# ========== 配置部分 ==========
PDF_PATH = "fastformula_ug.pdf"
DATA_DIR = "data/fastformula"
OUTPUT_TXT = os.path.join(DATA_DIR, "input.txt")

# ========== 步骤 1：提取 PDF 文本 ==========
def extract_text(pdf_path):
    print("📖 正在提取 PDF 文本...")
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() or "" for page in tqdm(reader.pages)])
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"Page\s*\d+", "", text)
    text = re.sub(r"Oracle FastFormula User’s Guide.*", "", text)
    return text

# ========== 步骤 2：分段 ==========
def chunk_text(text, max_chars=800):
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) > max_chars:
            chunks.append(current.strip())
            current = p
        else:
            current += " " + p
    if current:
        chunks.append(current.strip())
    return chunks

# ========== 步骤 3：生成问答 ==========
def make_qa_pairs(chunks):
    qa_list = []
    base_questions = [
        "这段讲了什么？",
        "这里解释了哪些关键概念？",
        "举个示例说明？",
        "它与 Payroll 有什么关系？",
        "公式写法是什么？"
    ]
    for c in chunks:
        q = random.choice(base_questions)
        qa_list.append((q, c))
    return qa_list

# ========== 步骤 4：写入 NanoGPT 格式 ==========
def save_to_txt(qa_list, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q, a in qa_list:
            f.write(f"用户: {q}\n助手: {a}\n\n")
    print(f"✅ 已生成 NanoGPT 格式文本: {out_path} (共 {len(qa_list)} 条问答)")

# ========== 步骤 5：调用 prepare.py ==========
def run_prepare():
    print("⚙️ 正在执行 NanoGPT 数据预处理...")
    cmd = [
        "python",
        "data/shakespeare_char/prepare.py",
        "--input",
        OUTPUT_TXT,
        "--output_dir",
        DATA_DIR
    ]
    subprocess.run(cmd, check=False)
    print("✅ 数据预处理完成，可直接进行训练。")

# ========== 主流程 ==========
if __name__ == "__main__":
    text = extract_text(PDF_PATH)
    chunks = chunk_text(text)
    qa_list = make_qa_pairs(chunks)
    save_to_txt(qa_list, OUTPUT_TXT)
    run_prepare()
