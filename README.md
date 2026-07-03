**Patent RAG Agent**

**Summary**: Retrieval-Augmented Generation (RAG) system and agent for AI/ML patents. This project contains two primary pieces:
- `rshenkar-vector.py` — build the local vector store: download patent HTML, chunk text, compute embeddings, and save a retrievable vector index in `rag_cache/`.
- `rshenkar.py` — the RAG agent that exposes three retrieval tools (semantic `vector_search`, live `google_patent_fetch`, and `uspto_search`) and uses them to answer patent questions.

**Quick Workflow**:
1. Build the vector store: run `python rshenkar-vector.py`. The script downloads the configured patent documents, chunks them, computes embeddings, and stores vectors + metadata in `rag_cache/`.
2. Start the agent: run `python rshenkar.py`. The agent uses `vector_search` first for questions about the local patent set; if more detail is required it falls back to `google_patent_fetch` (fetches full text from Google Patents) or `uspto_search` (PatentsView API).
3. Query flow: Agent receives question → runs `vector_search` to retrieve high-similarity patent chunks → if needed, fetches the full patent text or searches USPTO → assembles a concise, citation-aware answer.

**Accuracy & Behavior**:
- `vector_search` is optimized for retrieving the most relevant patent chunks from the local store; the agent prefers it for accuracy on covered patents.
- For deeper or missing coverage, `google_patent_fetch` retrieves the live patent page and `uspto_search` helps find related patents.
- `rag_cache/` is excluded from the repo and large cache files are ignored.

**Notes & Requirements**:
- Install dependencies listed in `rshenkar_requirements.txt`.
- `rag_cache/` and any `.env` files with API keys are intentionally excluded from commits.

**Commands**:
```
python rshenkar-vector.py
python rshenkar.py
```

**Files of interest**: `rshenkar-vector.py`, `rshenkar.py`, and the `rag_cache/` folder generated at runtime.
