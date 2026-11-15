"""Build notebooks/03_count_based.ipynb (Assignment 2, Part A)."""
import json
from pathlib import Path

C1 = '''import sys
from pathlib import Path
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df = pd.read_parquet(ROOT / "data" / "processed" / "cleaned_novels.parquet")
texts  = df["cleaned_text"].tolist()
titles = df["novel_title"].tolist()

wc = df["cleaned_text"].str.split().str.len()
print("documents:", len(df))
print("titles:", titles)
print("total words:", f"{wc.sum():,}", "| mean:", f"{wc.mean():,.0f}")
print("unique tokens:", f"{len({w for t in texts for w in t.split()}):,}")'''

C2 = '''# A1 - Bag of Words
bow_vec = CountVectorizer()
X_bow = bow_vec.fit_transform(texts)
n_docs, n_terms = X_bow.shape
sparsity = 100 * (1 - X_bow.nnz / (n_docs * n_terms))

print(f"BoW shape: {X_bow.shape}   (documents x vocabulary)")
print(f"non-zero: {X_bow.nnz:,} of {n_docs*n_terms:,} cells")
print(f"SPARSITY: {sparsity:.2f}%")

terms = bow_vec.get_feature_names_out()
pd.DataFrame(X_bow[:5, :8].toarray(), columns=terms[:8], index=titles[:5])'''

C3 = '''# A2 - TF-IDF: high vs low weights
tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(texts)
terms = tfidf_vec.get_feature_names_out()
print("TF-IDF shape:", X_tfidf.shape,
      "| sparsity: %.2f%%" % (100 * (1 - X_tfidf.nnz / X_tfidf.size)))

row = X_tfidf[0].toarray().ravel()
top = row.argsort()[::-1][:8]
print()
print("Doc 0 =", titles[0])
print("HIGH weight:", [(terms[i], round(row[i], 4)) for i in top])
nz = np.flatnonzero(row)
low = nz[np.argsort(row[nz])[:8]]
print("LOW weight :", [(terms[i], round(row[i], 6)) for i in low])

idf = tfidf_vec.idf_; o = np.argsort(idf)
print()
print("In ALL docs (lowest IDF):", [(terms[i], round(idf[i], 4)) for i in o[:8]])
print("Most distinctive (highest IDF):", [(terms[i], round(idf[i], 4)) for i in o[-8:]])'''

C4 = '''# A2b - common words driven to EXACTLY zero (unsmoothed IDF)
tv0 = TfidfVectorizer(smooth_idf=False)   # idf = ln(N / df)
X0 = tv0.fit_transform(texts)
t0 = tv0.get_feature_names_out()
z = np.flatnonzero(tv0.idf_ == 0)
print(f"{len(z)} words occur in ALL {len(texts)} documents -> idf = ln(10/10) = 0")
print("examples:", [t0[i] for i in z[:15]])
print("their TF-IDF weight in doc 0:", [round(float(X0[0, i]), 6) for i in z[:5]])'''

C5 = '''# A3 - n-gram vocabulary growth and sparsity trade-off
rows = []
for name, rng in [("unigrams (1,1)", (1,1)), ("uni+bi (1,2)", (1,2)), ("uni+bi+tri (1,3)", (1,3))]:
    v = CountVectorizer(ngram_range=rng)
    M = v.fit_transform(texts)
    rows.append({"n-grams": name, "vocabulary": M.shape[1], "non-zero": M.nnz,
                 "sparsity %": round(100 * (1 - M.nnz / M.size), 3)})
growth = pd.DataFrame(rows)
growth["growth x"] = (growth["vocabulary"] / growth["vocabulary"].iloc[0]).round(2)
print(growth.to_string(index=False))'''

C7 = '''# A4 - verify the hand computation against the library
toy = ["love rule love", "love rule sufi", "rule sufi rumi"]

tv = TfidfVectorizer(); Xt = tv.fit_transform(toy)
lib = float(Xt[0, list(tv.get_feature_names_out()).index("love")])
print(f"TF-IDF  hand = 0.9321917 | sklearn = {lib:.7f} | MATCH: {abs(0.9321917 - lib) < 1e-7}")

bv = CountVectorizer(ngram_range=(2, 2)); B = bv.fit_transform(toy)
j = list(bv.get_feature_names_out()).index("love rule")
print(f"BIGRAM  hand = 2 | sklearn = {int(B[:, j].sum())} | MATCH: {int(B[:, j].sum()) == 2}")
print("bigram vocab:", list(bv.get_feature_names_out()))'''

C8 = '''# Hand-off to Part D: save count-based vectorizers + matrices
import joblib
from scipy import sparse

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

joblib.dump(bow_vec,   MODELS / "bow_vectorizer.joblib")
joblib.dump(tfidf_vec, MODELS / "tfidf_vectorizer.joblib")
sparse.save_npz(MODELS / "bow_matrix.npz",   X_bow)
sparse.save_npz(MODELS / "tfidf_matrix.npz", X_tfidf)
df[["novel_title"]].to_csv(MODELS / "doc_index.csv", index=False)

print("saved:", sorted(p.name for p in MODELS.iterdir()))'''


def code(src, i):
    return {"cell_type": "code", "id": f"c{i}", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src, i):
    return {"cell_type": "markdown", "id": f"m{i}", "metadata": {},
            "source": src.splitlines(keepends=True)}


# M6 is removed from the list below
nb = {"cells": [md("# Assignment 2 - Part A: Count-based representations", 0),
                code(C1, 1), code(C2, 2), code(C3, 3), code(C4, 4), code(C5, 5),
                code(C7, 7), code(C8, 8)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

Path("notebooks").mkdir(exist_ok=True)
out = Path("notebooks/03_count_based.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("created", out, "with", len(nb["cells"]), "cells")