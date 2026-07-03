"""
HW11 Patent RAG Agent

This agent answers questions about AI/ML patents using three retrieval tools:
  1. vector_search       — semantic search over the local patent vector store
                           (built by rshenkar-vector.py, same corpus as HW6/HW7)
  2. google_patent_fetch — fetches live patent text from a Google Patents URL
                           using the requests + HTML-stripping approach from HW6
  3. uspto_search        — searches the official USPTO PatentsView API for patents
                           matching a keyword query, returning titles and abstracts

Implementation: bare OpenAI Responses API function-calling loop (no frameworks).
The agent loops until the model returns a final answer with no tool calls.

Example input: "How does BERT handle natural language understanding tasks?"
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

CACHE_DIR       = Path("./rag_cache")
EMBEDDINGS_FILE = CACHE_DIR / "embeddings.json"

EMBED_MODEL     = "text-embedding-3-small"
INFERENCE_MODEL = "gpt-4o-mini"
TOP_K           = 5

HEADERS = {"User-Agent": "Mozilla/5.0 (RAG-homework; educational use)"}

# ── Globals ───────────────────────────────────────────────────────────────────
_records: list[dict]    = []
_emb_matrix: np.ndarray = None
_client: OpenAI         = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _load_store() -> tuple[list[dict], np.ndarray]:
    global _records, _emb_matrix
    if not _records:
        with open(EMBEDDINGS_FILE) as f:
            _records = json.load(f)
        _emb_matrix = np.array([r["embedding"] for r in _records], dtype=np.float32)
    return _records, _emb_matrix


# ── Tool implementations ──────────────────────────────────────────────────────
def vector_search(query: str) -> str:
    """Semantic search over the local patent vector store."""
    client = _get_client()
    records, emb_matrix = _load_store()
    q_emb = np.array(
        client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding,
        dtype=np.float32,
    )
    sims    = emb_matrix @ q_emb / (
        np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_emb) + 1e-10
    )
    top_idx = np.argsort(sims)[-TOP_K:][::-1]
    results = [
        f"[{records[i]['source']}]\n{records[i]['chunk']}" for i in top_idx
    ]
    return "\n\n---\n\n".join(results)


def google_patent_fetch(url: str) -> str:
    """Fetch and extract readable text from a Google Patents URL."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s{2,}", " ", text)
    return text[:3000]


def uspto_search(query: str) -> str:
    """Search the USPTO PatentsView API for patents matching a keyword query."""
    url  = "https://search.patentsview.org/api/v1/patent/"
    body = {
        "q": {"_text_any": {"patent_abstract": query}},
        "f": ["patent_id", "patent_title", "patent_abstract", "patent_date"],
        "o": {"per_page": 5},
    }
    resp = requests.post(url, json=body, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    patents = resp.json().get("patents") or []
    if not patents:
        return "No USPTO results found."
    parts = []
    for p in patents:
        title    = p.get("patent_title", "N/A")
        date     = p.get("patent_date", "N/A")
        abstract = (p.get("patent_abstract") or "No abstract.")[:400]
        parts.append(f"Title: {title}\nDate: {date}\nAbstract: {abstract}")
    return "\n\n---\n\n".join(parts)


# ── Tool schemas ──────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "name": "vector_search",
        "description": (
            "Semantic search over a local vector store of 10 AI/ML patents: "
            "GPT-OpenAI, Transformer, GAN-Google, BERT-Google, FaceRecog-Meta, "
            "NeuralNet-MS, PopBasedTrain, Siri-Apple, Alexa-Amazon, AutoML-Google. "
            "Use this first for any question about these patents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query."},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "google_patent_fetch",
        "description": (
            "Fetch the full text of any patent directly from a Google Patents URL. "
            "Use this when you need deeper detail on a specific patent not covered "
            "well by vector_search, or for patents outside the local store."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full Google Patents URL, e.g. https://patents.google.com/patent/US10452978B2/en",
                },
            },
            "required": ["url"],
        },
    },
    {
        "type": "function",
        "name": "uspto_search",
        "description": (
            "Search the official USPTO PatentsView API for patents matching a keyword. "
            "Useful for finding related patents, checking prior art, or discovering "
            "patents outside the local vector store."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search for in patent abstracts, e.g. 'neural network image recognition'.",
                },
            },
            "required": ["query"],
        },
    },
]

TOOL_FN_MAP = {
    "vector_search":       vector_search,
    "google_patent_fetch": google_patent_fetch,
    "uspto_search":        uspto_search,
}


# ── Agent loop ────────────────────────────────────────────────────────────────
def run_agent(user_query: str) -> str:
    client = _get_client()
    print(f"\nUser: {user_query}\n")

    # First call — just the user message
    response = client.responses.create(
        model=INFERENCE_MODEL,
        tools=TOOLS,
        input=user_query,
    )

    while True:
        tool_calls = [item for item in response.output if item.type == "function_call"]

        if not tool_calls:
            # No tool calls — return final text answer
            for item in response.output:
                if item.type == "message":
                    return item.content[0].text
            return "(No text response)"

        # Execute each tool call and collect results
        tool_results = []
        for tc in tool_calls:
            fn_args = json.loads(tc.arguments)
            print(f"[tool] {tc.name}({fn_args})")
            try:
                result = TOOL_FN_MAP[tc.name](**fn_args)
            except Exception as e:
                result = f"Error: {e}"
            print(f"  → {result[:120]}…\n")
            tool_results.append({
                "type":    "function_call_output",
                "call_id": tc.call_id,
                "output":  result,
            })

        # Feed results back using previous_response_id (correct Responses API pattern)
        response = client.responses.create(
            model=INFERENCE_MODEL,
            tools=TOOLS,
            previous_response_id=response.id,
            input=tool_results,
        )


if __name__ == "__main__":
    if not EMBEDDINGS_FILE.exists():
        print("Vector store not found. Run rshenkar-vector.py first.")
        sys.exit(1)

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "How does BERT handle natural language understanding tasks?"
    )
    answer = run_agent(query)
    print(f"Agent: {answer}")