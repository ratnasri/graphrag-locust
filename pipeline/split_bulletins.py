"""
Split year-bundle bulletin texts into individual bulletin text files.
Individual monthly files (2026) are copied as-is.
Output: data/text/individual/ — one .txt per bulletin issue.
"""

import re
import shutil
from pathlib import Path
#Change directory to yours
INPUT_DIR = Path("G:/FAOLocustKG/data/text/bulletins")
OUTPUT_DIR = Path("G:/FAOLocustKG/data/text/individual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BULLETIN_HEADER = re.compile(
    r"(?:Desert Locust Bulletin\s*\n\s*)?No\.\s*(\d+)\s{3,}(\d{1,2}\s+\w+\s+\d{4})",
    re.IGNORECASE
)

SITUATION_HEADER = re.compile(
    r"General situation during (.+?)\n.*?Forecast until (.+?)\n",
    re.DOTALL
)


def split_year_bundle(text, source_name):
    """Split a year bundle text into individual bulletin chunks."""
    matches = list(BULLETIN_HEADER.finditer(text))
    if not matches:
        print(f"  No bulletin headers found in {source_name}")
        return []

    bulletins = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        bulletin_no = match.group(1)
        issue_date = match.group(2).strip()
        bulletins.append((bulletin_no, issue_date, chunk))

    return bulletins


def process_year_bundle(txt_path):
    text = txt_path.read_text(encoding="utf-8")
    bulletins = split_year_bundle(text, txt_path.name)

    print(f"{txt_path.name}: {len(bulletins)} bulletins found")
    for bulletin_no, issue_date, chunk in bulletins:
        safe_date = issue_date.replace(" ", "_").upper()
        out_name = f"bulletin_{bulletin_no}_{safe_date}.txt"
        out_path = OUTPUT_DIR / out_name
        out_path.write_text(chunk, encoding="utf-8")
        print(f"  -> {out_name} ({len(chunk)} chars)")


def process_individual(txt_path):
    """Copy individual monthly bulletin files as-is."""
    # Extract bulletin number from filename
    m = re.search(r"No_(\d+)", txt_path.name)
    if m:
        bulletin_no = m.group(1)
        text = txt_path.read_text(encoding="utf-8")

        # Get date from content
        match = BULLETIN_HEADER.search(text)
        if match:
            issue_date = match.group(2).strip().replace(" ", "_").upper()
            out_name = f"bulletin_{bulletin_no}_{issue_date}.txt"
        else:
            out_name = f"bulletin_{bulletin_no}.txt"

        out_path = OUTPUT_DIR / out_name
        shutil.copy(txt_path, out_path)
        print(f"{txt_path.name} -> {out_name}")
    else:
        shutil.copy(txt_path, OUTPUT_DIR / txt_path.name)
        print(f"{txt_path.name} -> copied as-is")


def run():
    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} text files\n")

    for txt_path in txt_files:
        # Year bundles: named 2023.txt, 2024.txt, 2025.txt
        if re.match(r"^\d{4}\.txt$", txt_path.name):
            process_year_bundle(txt_path)
        else:
            process_individual(txt_path)

    output_files = list(OUTPUT_DIR.glob("*.txt"))
    print(f"\nDone. {len(output_files)} individual bulletin files in {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
