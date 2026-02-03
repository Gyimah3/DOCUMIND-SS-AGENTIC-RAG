# RAG / RediSearch: Empty Content in Retrieval Results

## Problem Summary

**Observed:** Data exists in Redis (confirmed via Redis Insight): hash documents have non-empty `content`, `content_vector`, `filename` (e.g. "Manus"), `page_number`, `type`, etc. Despite this, RAG retrieval returns documents with **empty `page_content`** (and effectively empty results to the user).

- **Redis Insight:** Hash fields show real values (e.g. content: "LoRA Fine-tuning python trainer.py ...", content_vector: binary blob, filename: "Manus").
- **FT.SEARCH / redis-py:** Query returns the correct number of documents, but the parsed `Document` objects have empty `content` and `content_vector` when read via `getattr(d, "content", None)` in our code.
- **Index:** The index in use (e.g. "zazi") may have a **prefix** that does not match the key prefix under which documents were stored (e.g. index prefix `za` vs keys `zazi:*`), so either docs are not found at all, or they are found but field values are not returned correctly.

## Technical Context

- **Stack:** FastAPI app (DocuMindSS), LangChain, custom `RedisVectorStore` (`services/redis_vectorstore.py`), RediSearch (FT.SEARCH with vector KNN).
- **Index schema:** HASH index with `$.content`, `$.content_vector`, etc. and `as_name="content"`, `as_name="content_vector"`.
- **Search:** Vector search uses `Query("(*)=>[KNN k @content_vector $query_vector AS vector_score]").return_fields("filename", "content", ...).dialect(2)`.
- **Result mapping:** We build LangChain `Document` from `results.docs` using `getattr(d, "content", None)` for page_content.

## Root cause (in codebase)

**HASH schema used JSON path `$.field` instead of plain field names.**  
RediSearch HASH indexes expect schema field names to match the hash keys exactly (e.g. `content`, `filename`). The code used `$.content`, `$.filename`, etc., which is JSON path syntax for JSON documents. That mismatch can cause indexing or RETURN to not bind to the actual hash fields, so FT.SEARCH returns docs but with empty content.

**Fix applied in codebase:**
- `services/redis_vectorstore.py`:
  - **create_index:** Schema changed from `name="$.content"` (and same for other fields) to **plain names** `name="content"`, `name="filename"`, etc., and removed `as_name` (not needed when name is the hash key).
  - **search:** Added `_get_doc_field()` so we read result fields from redis-py docs using both string and bytes keys and from `payload`/`vars(doc)`, and we normalize `page_number` to int.

**Important:** Existing indexes were created with the old schema. To get correct retrieval you must either:
1. **Recreate the index** (drop without deleting keys, then create again so it picks up the new schema), then re-index, or  
2. **Drop and re-upload** documents so the index is created with the new schema.  
Use `python scripts/check_rag_retrieval.py --fix-reindex <index_name>` to drop and recreate the index (keys are kept; you may need to re-add documents if the index was empty or you want a clean state).

## Other possible causes (if issue persists)

1. **Index prefix vs key prefix mismatch**
   - Index "zazi" may have been created with prefix **"za"** (so only keys `za:*` are indexed).
   - If documents were stored under **"zazi:*"**, they are **not** in the index; if under **"za:*"**, they are. Redis Insight shows the hash content but not necessarily which index covers that key.
   - **Check:** Run `FT.INFO <index_name>` and inspect `index_definition` → `prefixes`. Compare with actual key names (e.g. `za:*` vs `zazi:*`).

2. **redis-py mapping of RETURN fields** (mitigated by `_get_doc_field()` in search)
   - RediSearch returns `[total, doc_id, [field1, val1, field2, val2, ...], ...]`. redis-py maps these into a `Document`-like object.
   - If the client uses `decode_responses=False`, field names in the reply may be **bytes** (e.g. `b"content"`). Our code uses `getattr(d, "content", None)`; if the object stores attributes under byte keys or in a `payload` dict, we would get `None`.
   - **Check:** Log `vars(d)` or the raw `FT.SEARCH` reply (e.g. `FT.SEARCH index "*" RETURN 3 content filename content_vector LIMIT 0 1`) and see the exact key names and values returned.

3. **RETURN clause and schema attribute names**
   - Our schema uses **as_name** (e.g. `as_name="content"`). In FT.SEARCH, RETURN should use the **attribute name** (the name used in the index), which is the as_name. So `RETURN ... content ...` is correct for HASH fields indexed as `$.content` AS content.
   - Some sources suggest that for certain Redis/RediSearch versions or JSON path setups, RETURN might need to use the **path** (e.g. `$.content`) instead of the alias. If your Redis version behaves that way, returning by path could fix empty content.

4. **Redis / RediSearch version**
   - **Vector search:** Requires **DIALECT 2** (we already use `.dialect(2)`).
   - **RETURN:** The RETURN clause is standard across recent RediSearch versions; behavior should be consistent for HASH.
   - **Version compatibility:** RediSearch 2.6+ (Redis 6.0.16+) supports vector similarity; 2.8+ (Redis 7.2+) has RESP3 and other changes. In theory, RETURN field names could differ by version or by client (decode_responses, RESP2 vs RESP3). Worth confirming:
     - Redis server version: `redis-cli INFO server`
     - RediSearch module: `redis-cli MODULE LIST`
     - redis-py version and whether `decode_responses` is used when calling `ft().search()`.

## Does this depend on Redis / RediSearch version?

- **Partly.** Online findings suggest:
  - **Vector search** requires **DIALECT 2** (we use it); older dialect can yield no/wrong results.
  - **RETURN** behavior is standard in RediSearch; empty content is more often due to **schema path** (e.g. `$.` vs `$..`), **index prefix vs key prefix**, or **client parsing** (redis-py attribute names / decode_responses) than to server version alone.
  - **RediSearch 2.6+** (Redis 6.0.16+) supports vector search; **2.8+** (Redis 7.2+) has RESP3. Different Redis/redis-py versions could return response keys as bytes vs str, which would break `getattr(d, "content", None)` if the object only has `b"content"`.
- **Recommendation:** Confirm Redis server version (`redis-cli INFO server`), RediSearch module version (`MODULE LIST`), and redis-py version; if RETURN reply uses byte keys, normalize to string when building Documents.

## References (online)

- **redis-py FT.SEARCH not returning results / empty content:** Path/schema issues (e.g. `$..` vs `$.`) can lead to results with empty fields even when data exists in Redis. [SO: redis-py ft().search() not returning results](https://stackoverflow.com/questions/70399096/redis-py-ft-search-not-returning-results).
- **Vector search empty results:** Requires DIALECT 2; vector must be passed as binary in PARAMS; wrong encoding or dialect can yield no or wrong results. [SO: Unable to find results in vector search when using Redis as a vector database](https://stackoverflow.com/questions/78837936/unable-to-find-results-in-vector-search-when-using-redis-as-a-vector-database).
- **RediSearch RETURN:** RETURN uses field identifiers and optional AS alias; for HASH, specify the field name (or the alias used in the index). [Redis FT.SEARCH](https://redis.io/docs/latest/commands/ft.search/).
- **RediSearch version notes:** [RediSearch release notes](https://redis.io/docs/latest/operate/oss_and_stack/stack-with-enterprise/release-notes/redisearch) — ensure Redis/RediSearch versions meet minimum requirements for vector search and that client and server agree on response format.

## What to Do Next

1. **Verify index prefix vs keys**
   - Run: `python scripts/check_rag_retrieval.py <index_name>` and check "Index prefix(es)" and "Key counts by prefix".
   - If docs are under `zazi:*` but index has prefix `za`, either re-index under `za:*` or drop and recreate the index with prefix `zazi` (e.g. `python scripts/check_rag_retrieval.py --fix-reindex zazi`).

2. **Inspect raw FT.SEARCH reply and redis-py Document**
   - In `scripts/check_rag_retrieval.py`, the script already does a raw `FT.SEARCH` and prints field names in the reply. Confirm that "content" (or `b"content"`) is present and has non-empty values.
   - Log the full `vars(d)` for one result doc in `RedisVectorStore.search` to see where redis-py puts `content` (attribute name vs bytes, or inside `payload`).

3. **Fallback: read content by key if RETURN is empty**
   - If FT.SEARCH returns doc ids but empty RETURN fields, as a workaround we could **HGETALL** the document key (the doc id) inside `RedisVectorStore.search` and use that hash to build `page_content` and metadata. This would confirm that the issue is RETURN/parsing rather than missing data.

4. **Record Redis/RediSearch versions**
   - Add to this doc or to runbooks: Redis server version, RediSearch module version, redis-py version, and whether the Redis client uses `decode_responses=True` anywhere when executing search.

## Diagnostic Script

- **Location:** `scripts/check_rag_retrieval.py`
- **Usage:**
  - `python scripts/check_rag_retrieval.py [index_name]` — list indexes, FT.INFO, key counts by prefix, raw FT.SEARCH result, and project RedisVectorStore search.
  - `python scripts/check_rag_retrieval.py --fix-reindex <index_name>` — drop index (keep keys) and recreate with `prefix=index_name` so keys `index_name:*` are indexed.

## File References

- `services/redis_vectorstore.py` — `create_index` (schema with `$.content` / as_name), `search` (vector query, return_fields, mapping to Document).
- `rag/tools.py` — RAG tool using retriever from `load_vector_store` (in `app.utils`).
- `app/utils.py` — `load_vector_store` building RedisVectorStore/retriever for given index.
