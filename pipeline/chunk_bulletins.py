"""
Split individual bulletin text files into page-level chunks.
Reads from data/text/individual/, writes data/graph/chunks.json.
Page 1 (boilerplate header) is skipped.
"""
import json
import re
from pathlib import Path

import networkx as nx

TEXT_DIR = Path("data/text/individual")
GRAPHML  = Path("data/graph/locust_graph.graphml")
OUT_FILE = Path("data/graph/chunks.json")


def load_bulletin_meta(graphml_path):
    G = nx.read_graphml(graphml_path)
    meta = {}
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "Bulletin":
            bnum = int(data["bulletin_number"])
            meta[bnum] = {
                "year":            int(data.get("year", 0)),
                "month":           data.get("month", ""),
                "coverage_period": data.get("coverage_period", ""),
                "issue_date":      data.get("issue_date", ""),
            }
    return meta


def split_pages(text):
    """Return list of (page_num, page_text) by splitting on '--- PAGE N ---' markers."""
    if not re.search(r"---\s*PAGE\s+\d+", text):
        return [(1, text)]
    parts = re.split(r"---\s*PAGE\s+(\d+)\s*---", text)
    pages = [(1, parts[0])]
    for i in range(1, len(parts) - 1, 2):
        pages.append((int(parts[i]), parts[i + 1]))
    return pages


def main():
    meta = load_bulletin_meta(GRAPHML)

    files = sorted(TEXT_DIR.glob("bulletin_*.txt"))
    print(f"Found {len(files)} bulletin files")

    chunks  = []
    skipped = 0

    for fpath in files:
        m = re.search(r"bulletin_(\d+)_", fpath.name, re.IGNORECASE)
        if not m:
            print(f"  Skip (no bulletin number in filename): {fpath.name}")
            continue

        bnum = int(m.group(1))
        bm   = meta.get(bnum, {})
        text = fpath.read_text(encoding="utf-8", errors="replace")

        for page_num, page_text in split_pages(text):
            if page_num == 1:
                continue  # skip header/boilerplate

            clean = page_text.strip()
            if len(clean) < 80:
                skipped += 1
                continue

            chunks.append({
                "chunk_id":        f"{bnum}_p{page_num}",
                "bulletin_number": bnum,
                "page":            page_num,
                "year":            bm.get("year"),
                "month":           bm.get("month", ""),
                "coverage_period": bm.get("coverage_period", ""),
                "text":            clean,
            })

    OUT_FILE.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    from collections import Counter
    years = Counter(c["year"] for c in chunks if c["year"])
    print(f"Saved {len(chunks)} chunks ({skipped} near-empty skipped) -> {OUT_FILE}")
    print(f"By year: {dict(sorted(years.items()))}")


if __name__ == "__main__":
    main()
