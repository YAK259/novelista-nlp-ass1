import pandas as pd
from pathlib import Path

df = pd.read_parquet("data/processed/cleaned_novels.parquet")
print("SHAPE:", df.shape)
print("COLUMNS:", list(df.columns))
print("\n--- head ---")
print(df.head(3).to_string()[:800])

text_col = next((c for c in ["clean_text", "cleaned_text", "text", "raw_text", "content"]
                 if c in df.columns), None)
if text_col is None:
    text_col = df.select_dtypes(include="object").columns[-1]
print("\nUSING TEXT COLUMN:", text_col)

wc = df[text_col].astype(str).str.split().str.len()
print("docs:", len(df), "| total words:", f"{wc.sum():,}", "| mean:", f"{wc.mean():,.0f}")
print("\n--- first 300 chars of doc 0 ---")
print(repr(str(df[text_col].iloc[0])[:300]))

for f in ["src/load_data.py", "src/clean_data.py"]:
    print(f"\n===== {f} =====")
    print(Path(f).read_text(encoding="utf-8"))