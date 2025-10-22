from pypdf import PdfReader

def extract_pdf(pdf_path, out_txt=None):
    if out_txt is None:
        out_txt = pdf_path.replace(".pdf", ".txt")
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages])
    open(out_txt, "w", encoding="utf-8").write(text)
    print(f"extract completed：{out_txt}")

#extract_pdf("fastformula_ug.pdf")
extract_pdf("administering-fast-formulas.pdf")
