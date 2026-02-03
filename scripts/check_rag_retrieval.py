#!/usr/bin/env python3
"""
Check RAG retrieval: list Redis indexes, show index info, run a search.
Usage (from project root):
  python scripts/check_rag_retrieval.py [index_name]
  uv run python scripts/check_rag_retrieval.py gideon
"""
import asyncio
import os
import sys

# Run from project root so app and services are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis
from app.config import settings
from services.redis_vectorstore import RedisVectorStore


def _get_embedding(model: str = "text-embedding-3-small"):
    from langchain_openai import OpenAIEmbeddings
    api_key = getattr(settings, "openai_api_key", None) or ""
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAIEmbeddings(model=model)


def fix_reindex(index_name: str) -> None:
    """Drop index without deleting keys, then recreate with prefix = index_name so keys index_name:* are indexed."""
    client = redis.Redis.from_url(settings.redis_url)
    try:
        client.execute_command("FT.DROPINDEX", index_name)  # no DD = keep keys
        print(f"Dropped index {index_name} (keys kept).")
    except redis.exceptions.ResponseError as e:
        print(f"Drop index: {e}")
        return
    emb = _get_embedding()
    store = RedisVectorStore(client=client, embedding_func=emb)
    store.create_index(index_name, prefix=index_name)
    print(f"Recreated index {index_name} with prefix {index_name!r}. Keys {index_name}:* should now be searchable.")


def main():
    argv = sys.argv[1:]
    if "--fix-reindex" in argv:
        idx = argv.index("--fix-reindex")
        index_name = (argv[idx + 1] if idx + 1 < len(argv) else "zazi").strip()
        print(f"Fix-reindex: {index_name}")
        fix_reindex(index_name)
        return
    args = [a for a in argv if a and not a.startswith("-")]
    index_name = (args[0] if args else "zazi").strip()
    print(f"Redis URL: {settings.redis_url[:50]}...")
    print(f"Index name: {index_name}\n")

    client = redis.Redis.from_url(settings.redis_url)

    # 1) List RediSearch indexes
    try:
        indexes = client.execute_command("FT._LIST")
        decoded = [x.decode() if isinstance(x, bytes) else x for x in indexes]
        print(f"FT._LIST: {decoded}")
    except Exception as e:
        print(f"FT._LIST error: {e}")
        return

    # 2) Info for this index
    try:
        info = client.ft(index_name).info()
        if isinstance(info, dict):
            print(f"\nFT.INFO {index_name}:")
            for k, v in sorted(info.items()):
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                print(f"  {key}: {val}")
        else:
            print(f"FT.INFO {index_name}: {info}")
    except redis.exceptions.ResponseError as e:
        print(f"\nFT.INFO {index_name} error: {e}")
        print("  -> Index may not exist or use a different name. Try listing keys with the prefix.")
        # Try to count keys that might belong to this index
        try:
            keys = client.keys(f"{index_name}:*")
            print(f"  Keys matching '{index_name}:*': {len(keys)}")
            if keys:
                print(f"  Sample key: {keys[0]}")
        except Exception as e2:
            print(f"  Keys scan error: {e2}")
        return

    # 3) Key counts: index has a prefix (e.g. "za") - docs with content may be under full name (e.g. "zazi:*")
    print("\n--- Key counts by prefix ---")
    try:
        for prefix in (index_name[:2], index_name):  # e.g. "za" and "zazi"
            keys = client.keys(f"{prefix}:*")
            n = len(keys)
            if n:
                sample = keys[0]
                sample_key = sample.decode() if isinstance(sample, bytes) else sample
                h = client.hgetall(sample_key)
                content_len = len(h.get(b"content") or h.get("content") or b"")
                vec_len = len(h.get(b"content_vector") or h.get("content_vector") or b"")
                print(f"  {prefix}:* -> {n} keys, sample key content={content_len} bytes content_vector={vec_len} bytes")
            else:
                print(f"  {prefix}:* -> 0 keys")
    except Exception as e:
        print(f"  Error: {e}")

    # 4) Raw FT.SEARCH * to see what the index actually returns
    print("\n--- Raw FT.SEARCH * LIMIT 0 2 (what the index returns) ---")
    try:
        from redis.commands.search.query import Query
        q = Query("*").return_fields("content", "filename", "content_vector").paging(0, 2)
        res = client.ft(index_name).search(q)
        print(f"  Total in index: {res.total}, docs: {len(res.docs)}")
        for i, d in enumerate(res.docs):
            key = getattr(d, "id", None)
            # redis-py Document stores RETURN fields as attributes; keys may be as in schema
            doc_dict = {k: v for k, v in vars(d).items() if k not in ("id", "payload", "score") and not k.startswith("_")}
            content = doc_dict.get("content") or getattr(d, "content", None) or b""
            cv = doc_dict.get("content_vector") or getattr(d, "content_vector", None)
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")[:80] if content else "(empty)"
            elif content and len(str(content)) > 80:
                content = str(content)[:80] + "..."
            has_vec = f"content_vector: {len(cv) if cv else 0} bytes"
            print(f"  doc {i+1} id={key} content={content!r} {has_vec}")
            if not content or not cv:
                print(f"    doc keys: {list(doc_dict.keys())}")
        # Raw FT.SEARCH to see response shape (Redis returns [total, id1, [k1,v1,k2,v2,...], id2, ...])
        if res.docs and not doc_dict:
            raw = client.execute_command("FT.SEARCH", index_name, "*", "RETURN", "3", "content", "filename", "content_vector", "LIMIT", "0", "1")
            print(f"  Raw reply: total={raw[0]}, len(res)={len(raw)}")
            if len(raw) >= 3:
                doc1_id, doc1_fields = raw[1], raw[2]
                print(f"  doc1_fields type={type(doc1_fields).__name__}, len={len(doc1_fields) if isinstance(doc1_fields, (list, dict)) else 'n/a'}")
                if isinstance(doc1_fields, list) and doc1_fields:
                    keys_in_reply = doc1_fields[::2]
                    print(f"  Field names in reply: {[k.decode() if isinstance(k, bytes) else k for k in keys_in_reply]}")
                # Try RETURN with $.attribute (identifier) in case RediSearch expects that
                raw2 = client.execute_command("FT.SEARCH", index_name, "*", "RETURN", "3", "$.content", "$.filename", "$.content_vector", "LIMIT", "0", "1")
                if len(raw2) >= 3 and isinstance(raw2[2], list) and raw2[2]:
                    keys2 = raw2[2][::2]
                    print(f"  With $. prefix - field names: {[k.decode() if isinstance(k, bytes) else k for k in keys2]}")
        try:
            idef = client.ft(index_name).info().get(b"index_definition") or []
            if isinstance(idef, list) and b"prefixes" in idef:
                idx = idef.index(b"prefixes") + 1
                if idx < len(idef):
                    prefs = idef[idx]
                    prefs = [x.decode() if isinstance(x, bytes) else x for x in prefs] if isinstance(prefs, list) else [prefs]
                    print(f"  Index prefix(es): {prefs} -> only these keys are in the index.")
                    if prefs and index_name and index_name not in prefs and index_name.startswith(prefs[0]):
                        print(f"  -> Your docs may be under '{index_name}:*' but index only has prefix {prefs[0]!r}. Run: python scripts/check_rag_retrieval.py --fix-reindex {index_name}")
        except Exception:
            pass
    except Exception as e:
        print(f"  Error: {e}")

    # 5) Run search using project RedisVectorStore
    print("\n--- Search (project RedisVectorStore) ---")
    try:
        emb = _get_embedding()
        store = RedisVectorStore(client=client, embedding_func=emb)
    except Exception as e:
        print(f"Embedding/Store init error: {e}")
        return

    async def run_search():
        docs = await store.search(index_name, "music", k=5)
        return docs

    docs = asyncio.run(run_search())
    print(f"Query 'music' -> {len(docs)} docs")
    for i, d in enumerate(docs[:5]):
        content_preview = (d.page_content or "")[:120].replace("\n", " ")
        print(f"  [{i+1}] {content_preview}...")
    if not docs:
        print("  (No results - check index prefix vs key prefix when indexing)")


if __name__ == "__main__":
    main()
