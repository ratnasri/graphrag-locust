"""
One-shot script to build demo_cache.json from existing bulletin text files.
Run from the project root (G:/FAOLocustKG):
    python pipeline/build_demo.py
"""
import subprocess
import sys

steps = [
    ("Chunking bulletins",   ["python", "pipeline/chunk_bulletins.py"]),
    ("Embedding chunks",     ["python", "pipeline/embed_chunks.py"]),
    ("Pre-cooking demo",     ["python", "pipeline/precook_demo.py"]),
]

for label, cmd in steps:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: step failed with exit code {result.returncode}")
        sys.exit(result.returncode)

print("\nDone. Run: streamlit run app.py")
