"""Load raw novel .txt files from a folder into a pandas DataFrame."""
from pathlib import Path

import pandas as pd


def load_novels(folder):
    """Return DataFrame with columns: title, filename, raw_text (one row per .txt)."""
    folder = Path(folder)
    files = sorted(folder.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {folder.resolve()} - run download_books.py first."
        )
    rows = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        rows.append({"title": f.stem, "filename": f.name, "raw_text": text})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    df = load_novels(root / "data" / "raw")
    df["word_count"] = df["raw_text"].str.split().str.len()
    print(df[["title", "word_count"]].to_string(index=False))
    print(f"\nTotal documents: {len(df)} | Total words: {df['word_count'].sum():,}")