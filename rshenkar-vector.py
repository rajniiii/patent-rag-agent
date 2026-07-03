"""
Build the HW11 vector store from AI/ML patent documents.

Downloads the same 10 Google Patents used in HW6/HW7, chunks by tokens
(using the inference model tokenizer, per HW7), computes OpenAI embeddings,
and saves to disk. Skip re-download/re-embed if cache already exists.

Run once before agent.py:
    uv run 
"""

import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY_PATH    = Path("./api_key.txt")
CACHE_DIR       = Path("./rag_cache")
EMBEDDINGS_FILE = CACHE_DIR / "embeddings.json"

EMBED_MODEL     = "text-embedding-3-small"
INFERENCE_MODEL = "gpt-4o-mini"
CHUNK_TOKENS    = 300
CHUNK_OVERLAP   = 50

PATENTS = [
    ("GPT-OpenAI",     "https://patents.google.com/patent/US10452978B2/en"),
    ("Transformer",    "https://patents.google.com/patent/US10452747B2/en"),
    ("GAN-Google",     "https://patents.google.com/patent/US10621481B2/en"),
    ("BERT-Google",    "https://patents.google.com/patent/US11556767B2/en"),
    ("FaceRecog-Meta", "https://patents.google.com/patent/US9123006B2/en"),
    ("NeuralNet-MS",   "https://patents.google.com/patent/US7747070B2/en"),
    ("PopBasedTrain",  "https://patents.google.com/patent/US11604985B2/en"),
    ("Siri-Apple",     "https://patents.google.com/patent/US8543407B1/en"),
    ("Alexa-Amazon",   "https://patents.google.com/patent/US9966073B2/en"),
    ("AutoML-Google",  "https://patents.google.com/patent/US11250328B2/en"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        html = r.read().decode("utf-8", errors="ignore")
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, enc: tiktoken.Encoding) -> list[str]:
    tokens = enc.encode(text)
    chunks, i = [], 0
    while i < len(tokens):
        chunks.append(enc.decode(tokens[i : i + CHUNK_TOKENS]))
        i += CHUNK_TOKENS - CHUNK_OVERLAP
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if EMBEDDINGS_FILE.exists():
        print("Vector store already exists — delete rag_cache/ to rebuild.")
        return

    from dotenv import load_dotenv
    import os
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    enc     = tiktoken.encoding_for_model(INFERENCE_MODEL)
    CACHE_DIR.mkdir(exist_ok=True)
    records = []

    for name, url in PATENTS:
        print(f"Fetching {name}…")
        try:
            chunks = chunk_text(fetch_text(url), enc)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue
        print(f"  {len(chunks)} chunks")

        for start in range(0, len(chunks), 100):
            batch = chunks[start : start + 100]
            resp  = client.embeddings.create(model=EMBED_MODEL, input=batch)
            for j, emb_obj in enumerate(resp.data):
                records.append({
                    "source":    name,
                    "chunk":     batch[j],
                    "embedding": emb_obj.embedding,
                })

    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(records, f)
    print(f"[done] saved {len(records)} chunks to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()