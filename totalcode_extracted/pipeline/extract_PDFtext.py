import fitz
from pathlib import Path

INPUT_DIR = Path("data/raw/bulletins")
OUTPUT_DIR = Path("data/text/bulletins")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf_path in sorted(INPUT_DIR.glob("*.pdf")):
    txt_path = OUTPUT_DIR / (pdf_path.stem + ".txt")
    print(f"Processing {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += f"--- PAGE {page.number + 1} ---\n"
        text += page.get_text()
        text += "\n\n"

    txt_path.write_text(text, encoding="utf-8")
    print(f"{pdf_path.name}: {len(doc)} pages, {len(text)} chars")

print("Done.")