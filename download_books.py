import ssl
import urllib.request
from pathlib import Path

try:
    import truststore
    CTX = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # use Windows cert store
except ImportError:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())

BOOKS = {
    "Pride and Prejudice": 1342,
    "Alice in Wonderland": 11,
    "A Tale of Two Cities": 98,
    "Frankenstein": 84,
    "The Adventures of Sherlock Holmes": 1661,
    "Moby Dick": 2701,
    "Jane Eyre": 1260,
    "Wuthering Heights": 768,
    "Great Expectations": 1400,
    "The Picture of Dorian Gray": 174,
}

out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)
for title, eid in BOOKS.items():
    dest = out / f"{title}.txt"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip (already there): {title}")
        continue
    url = f"https://www.gutenberg.org/cache/epub/{eid}/pg{eid}.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"saved {title} ({dest.stat().st_size/1024:.0f} KB)")
    except Exception as e:
        print(f"FAILED {title}: {e}")