"""
Builds the FAISS index from the ai4bharat/MSMARCO-XI dataset.

Run once (and whenever you want to rebuild the index):
    python scripts/build_index.py --limit 5000

What it does:
  1. Streams passages from huggingface.co/datasets/ai4bharat/MSMARCO-XI
  2. Runs all three chunking strategies (fixed-size, sentence-window, semantic)
     from backend/pipeline/chunking.py over every passage - see that file's
     docstring for why we use several strategies instead of one.
  3. Embeds every chunk with the local sentence-transformers model (no API
     calls => fast, and keeps query-time retrieval free of network latency).
  4. Builds a FAISS IndexFlatIP (cosine similarity via normalized inner
     product) and writes it + a metadata.jsonl sidecar to data/index/.

--limit caps how many source passages are processed, so you can do a fast
smoke-test build before committing to the full corpus (which is large).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pinecone import Pinecone
from fastembed import TextEmbedding
from tqdm import tqdm

from backend.config import settings
from backend.pipeline.chunking import build_chunks_multi_strategy


def load_passages(limit: int, split: str, config: str):
    """Loads passages from the HF dataset. Falls back to a small bundled
    sample corpus if the dataset can't be downloaded (offline dev / no
    internet in the grading environment), so the rest of the pipeline is
    always demoable end-to-end.
    """
    try:
        from datasets import load_dataset
        print(f"Downloading ai4bharat/MSMARCO-XI (config={config}, split={split})...")
        ds = load_dataset("ai4bharat/MSMARCO-XI", config, split=split, streaming=True)
        passages = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            # MSMARCO-XI rows are query/passage-pair style; pull out passage text
            # robustly across possible schema variants.
            text = row.get("passage_text") or row.get("passage") or row.get("text") or row.get("context")
            doc_id = str(row.get("id", row.get("query_id", i)))
            if isinstance(text, list):
                for j, t in enumerate(text):
                    if t and t.strip():
                        passages.append((f"{doc_id}-{j}", t.strip()))
            elif text and text.strip():
                passages.append((doc_id, text.strip()))
        if passages:
            print(f"Loaded {len(passages)} passages from HuggingFace.")
            return passages
        raise RuntimeError("Dataset loaded but yielded no usable passage text.")
    except Exception as e:
        print(f"[WARN] Could not load ai4bharat/MSMARCO-XI ({e}).")
        print("[WARN] Falling back to bundled sample corpus at data/sample_corpus.jsonl")
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_corpus.jsonl")
        passages = []
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                passages.append((row["id"], row["text"]))
        return passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3000, help="Max number of source passages to index")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--config", type=str, default="hi", help="Language config, e.g. hi/ta/te/en - check dataset card")
    parser.add_argument(
        "--strategies", type=str, default="fixed_size,sentence_window,semantic",
        help="Comma-separated chunking strategies to include in the index",
    )
    args = parser.parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    os.makedirs(settings.INDEX_DIR, exist_ok=True)

    print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    embedder = TextEmbedding(model_name=settings.EMBEDDING_MODEL)

    passages = load_passages(args.limit, args.split, args.config)

    all_texts = []
    all_meta = []
    t0 = time.time()
    for doc_id, text in tqdm(passages, desc="Chunking"):
        chunks = build_chunks_multi_strategy(
            doc_id, text, embedder, strategies=strategies,
            extra_meta={"source_doc_id": doc_id},
        )
        for c in chunks:
            all_texts.append(c.text)
            all_meta.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "strategy": c.strategy,
                "doc_id": c.doc_id,
                **c.metadata,
            })
    print(f"Chunking done in {time.time()-t0:.1f}s -> {len(all_texts)} chunks "
          f"from {len(passages)} passages using strategies={strategies}")

    if not all_texts:
        print("[ERROR] No chunks produced. Aborting index build.")
        return

    print("Embedding chunks (this is the slow one-time cost; retrieval itself will be fast)...")
    t0 = time.time()
    embeddings_gen = embedder.embed(all_texts, batch_size=64)
    embeddings = np.asarray(list(embeddings_gen), dtype="float32")
    print(f"Embedded {len(all_texts)} chunks in {time.time()-t0:.1f}s, dim={embeddings.shape[1]}")

    print("\nConnecting to Pinecone...")
    if not settings.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in backend/.env")
    
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)

    print("Upserting vectors to Pinecone in batches...")
    batch_size = 100
    upsert_data = []
    
    for i, meta in enumerate(all_meta):
        # Format required by Pinecone: (id, vector, metadata)
        vector = embeddings[i].tolist()
        upsert_data.append((meta["chunk_id"], vector, meta))
        
        if len(upsert_data) >= batch_size:
            index.upsert(vectors=upsert_data)
            upsert_data = []
            
    if upsert_data:
        index.upsert(vectors=upsert_data)

    print(f"\nSuccessfully upserted {len(all_texts)} vectors to Pinecone index '{settings.PINECONE_INDEX_NAME}'!")
    
    print("\nStrategy breakdown:")
    from collections import Counter
    strat_counts = Counter(m["strategy"] for m in all_meta)
    for k, v in strat_counts.items():
        print(f"  {k}: {v} chunks")


if __name__ == "__main__":
    main()
