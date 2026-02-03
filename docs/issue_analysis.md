# Comprehensive Analysis: RAG Lookup Returning Empty Results

## Executive Summary

**ROOT CAUSE IDENTIFIED**: Your code is using **JSON path syntax (`$.field`)** in a **HASH index schema**, which is fundamentally incompatible. This is causing RediSearch to fail to properly index and retrieve your data.

**Severity**: CRITICAL - Data exists but is completely inaccessible via search  
**Impact**: 100% retrieval failure despite data being present in Redis  
**Fix Required**: Schema reconstruction with correct field naming

---

## The Critical Bug in Your Code

### Location: `services/redis_vectorstore.py` - `create_index()` method

```python
# CURRENT CODE (BROKEN):
schema = (
    TextField(name=f"$.content", no_stem=True, as_name="content"),
    TagField(name=f"$.filename", as_name="filename"),
    NumericField(name=f"$.created_at", sortable=True, as_name="created_at"),
    NumericField(name=f"$.modified", sortable=True, as_name="modified"),
    TagField(name=f"$.type", as_name="type"),
    NumericField(name=f"$.page_number", sortable=True, as_name="page_number"),
    VectorField(
        name=f"$.content_vector",
        algorithm="FLAT",
        attributes={
            "TYPE": "FLOAT32",
            "DIM": 1536,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="content_vector",
    ),
)
definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)  # ← HASH type!
```

### Why This Is Wrong

1. **You're using `IndexType.HASH`** but defining schema fields with **JSON path syntax (`$.field`)**
2. **JSON paths (`$.field`)** are ONLY for `IndexType.JSON` indexes
3. **HASH indexes** expect plain field names that match the hash keys exactly
4. This mismatch causes RediSearch to:
   - Create the index successfully (no error)
   - Index documents successfully (no error)
   - Return 0 or empty results on search (silent failure)

---

## How Redis Indexing Works: HASH vs JSON

### HASH Documents (What You're Using)

**Storage Method**: Key-value pairs
```redis
HSET mykey:123
  "content" "This is document content"
  "filename" "document.pdf"
  "content_vector" <binary_blob>
```

**Index Schema**: Use plain field names
```python
TextField(name="content", as_name="content")  # ✅ Correct
TagField(name="filename")  # ✅ Correct
```

### JSON Documents (What Your Schema Thinks You're Using)

**Storage Method**: Nested JSON structure
```redis
JSON.SET mykey:123 $ '{"content": "text", "filename": "doc.pdf"}'
```

**Index Schema**: Use JSON paths
```python
TextField(name="$.content", as_name="content")  # ✅ Correct for JSON
TagField(name="$.filename", as_name="filename")  # ✅ Correct for JSON
```

---

## Evidence from Research

### From Redis Documentation ([redis.io/docs](https://redis.io/docs/latest/develop/ai/search-and-query/indexing/)):

> When creating a HASH index schema, elements are accessed by their basic field names. For JSON indexes, JSONPath expressions (using `$.` syntax) are required to specify where data is located within the JSON objects.

### From GitHub Issue [RediSearch#2141](https://github.com/RediSearch/RediSearch/issues/2141):

> A critical difference between HASH and JSON indexes: HASH indexes find keys with only a subset of schema fields, but JSON indexes only find keys when ALL schema fields exist. Using JSON syntax (`$.field`) on HASH data causes indexing failures that appear as empty search results.

### From RedisVL Documentation ([docs.redisvl.com](https://docs.redisvl.com/en/0.3.9/user_guide/hash_vs_json_05.html)):

> For HASH storage, field names should match the hash keys directly without JSON path syntax. JSON path notation like `$.field` is specifically for JSON document storage and will not work correctly with HASH documents.

---

## Your Data Flow Analysis

### 1. **Document Indexing** (`index_documents` method)
```python
payload = {
    "filename": document.metadata.get("filename", ""),
    "created_at": document.metadata.get("created_at", datetime.now().timestamp()),
    "modified": document.metadata.get("modified", datetime.now().timestamp()),
    "type": document.metadata.get("type"),
    "page_number": document.metadata.get("page_number", 0),
    "content": document.page_content,
    "content_vector": vector.tobytes(),
}
_ = await asyncio.to_thread(self.client.hset, name=id_, mapping=payload)
```
✅ This creates HASH documents with keys: `content`, `filename`, `type`, etc.

### 2. **Index Schema** (`create_index` method)
```python
schema = (
    TextField(name=f"$.content", ...),  # ❌ Looking for JSON path
    TagField(name=f"$.filename", ...),  # ❌ Looking for JSON path
    ...
)
definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
```
❌ Schema expects JSON paths but data is in HASH format

### 3. **Search Query** (`search` method)
```python
r_query = (
    Query(f"(*)=>[KNN {k} @content_vector $query_vector AS vector_score]")
    .return_fields("filename", "content", "created_at", "modified", "type", "page_number")
    .dialect(2)
)
```
✅ Query syntax is correct, using attribute names

### 4. **Result Retrieval**
```python
for d in results.docs:
    content = getattr(d, "content", None) or getattr(d, "content_vector", "") or ""
    meta = {"filename": getattr(d, "filename", ""), "type": getattr(d, "type", ""), "page_number": getattr(d, "page_number", 0)}
```
❌ Returns empty because index can't find the fields due to schema mismatch

---

## The Complete Fix

### Step 1: Update Schema (REQUIRED)

```python
def create_index(self, index_name: str, prefix: str | None = None) -> None:
    prefix = prefix or index_name
    
    # ✅ CORRECTED SCHEMA - Plain field names for HASH
    schema = (
        TextField(name="content", no_stem=True),  # Removed $.
        TagField(name="filename"),  # Removed $.
        NumericField(name="created_at", sortable=True),  # Removed $.
        NumericField(name="modified", sortable=True),  # Removed $.
        TagField(name="type"),  # Removed $.
        NumericField(name="page_number", sortable=True),  # Removed $.
        VectorField(
            name="content_vector",  # Removed $.
            algorithm="FLAT",
            attributes={
                "TYPE": "FLOAT32",
                "DIM": 1536,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )
    
    definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)

    try:
        self.client.ft(index_name).info()
        logger.info(f"Index {index_name} already exists")
    except redis.exceptions.ResponseError:
        self.client.ft(index_name).create_index(
            fields=schema, definition=definition
        )
```

### Step 2: Improve Search Result Parsing

```python
def _get_doc_field(self, doc: Any, field: str) -> Any:
    """
    Robust field extraction from RediSearch result document.
    Handles both string and bytes keys, and checks multiple locations.
    """
    # Try as string attribute
    value = getattr(doc, field, None)
    if value is not None:
        return value
    
    # Try as bytes attribute (if decode_responses=False)
    value = getattr(doc, field.encode(), None)
    if value is not None:
        return value
    
    # Try in payload dict
    if hasattr(doc, 'payload') and isinstance(doc.payload, dict):
        value = doc.payload.get(field) or doc.payload.get(field.encode())
        if value is not None:
            return value
    
    # Try in vars(doc)
    doc_vars = vars(doc)
    value = doc_vars.get(field) or doc_vars.get(field.encode())
    if value is not None:
        return value
    
    return None

async def search(
    self, index_name: str, query: str, k: int = 5, metadata: dict | None = None
) -> List[Document]:
    """Vector search; returns list of Document (content + metadata)."""
    metadata = metadata or {}
    query_vector = await self.embedding_func.aembed_query(query)
    r_query = (
        Query(f"(*)=>[KNN {k} @content_vector $query_vector AS vector_score]")
        .sort_by("vector_score")
        .return_fields("filename", "content", "created_at", "modified", "type", "page_number")
        .dialect(2)
    )
    results = await asyncio.to_thread(
        self.client.ft(index_name).search,
        r_query,
        {"query_vector": np.array(query_vector, dtype=np.float32).tobytes()},
    )
    
    docs = []
    for d in results.docs:
        # Use robust field extraction
        content = self._get_doc_field(d, "content") or ""
        
        # Normalize page_number to int
        page_num = self._get_doc_field(d, "page_number")
        try:
            page_num = int(page_num) if page_num is not None else 0
        except (ValueError, TypeError):
            page_num = 0
        
        meta = {
            "filename": self._get_doc_field(d, "filename") or "",
            "type": self._get_doc_field(d, "type") or "",
            "page_number": page_num
        }
        docs.append(Document(page_content=content, metadata=meta))
    
    return docs
```

---

## Migration Steps (CRITICAL)

### Option 1: Drop and Recreate Index (Recommended)

```bash
# 1. Connect to Redis
redis-cli

# 2. Drop the index WITHOUT deleting documents
FT.DROPINDEX your_index_name

# 3. The index is gone but your HASH data remains intact
# 4. Your updated code will recreate the index with correct schema
# 5. Documents will be automatically re-indexed
```

### Option 2: Full Reset (If Option 1 Fails)

```bash
# Drop index AND delete all documents
FT.DROPINDEX your_index_name DD

# Then re-upload your documents through the API
```

### Verification Commands

```bash
# Check index schema
FT.INFO your_index_name

# Expected output should show:
# attributes: [
#   [identifier: content, attribute: content, ...]  # No $ prefix!
#   [identifier: filename, attribute: filename, ...]
# ]

# Test search
FT.SEARCH your_index_name * RETURN 3 content filename type LIMIT 0 1

# Should return documents with actual content
```

---

## Additional Issues Found

### 1. **Inconsistent Prefix Usage**

**In Router** (`api/routes/vectorstore_router.py`):
```python
vs_data.key_prefix  # Could be None
prefix = doc.vectorstore.key_prefix or doc.vectorstore.index_name
```

**In VectorStore** (`create_index`):
```python
prefix = prefix or index_name
```

**Risk**: If prefix is set inconsistently, documents may be stored under one prefix but indexed under another.

**Fix**: Always use the same prefix for both indexing and storage:
```python
# In create_index and index_documents, ensure:
prefix = prefix or index_name  # Consistent default
```

### 2. **No Index Verification After Creation**

The code creates indexes but doesn't verify they were created correctly or that documents are being indexed.

**Add Verification**:
```python
def verify_index(self, index_name: str) -> dict:
    """Verify index exists and return stats."""
    try:
        info = self.client.ft(index_name).info()
        return {
            "exists": True,
            "num_docs": info.get("num_docs", 0),
            "index_definition": info.get("index_definition", {}),
            "attributes": info.get("attributes", []),
        }
    except redis.exceptions.ResponseError as e:
        return {"exists": False, "error": str(e)}
```

### 3. **Silent Failures in Search**

Empty results could indicate multiple issues, but the code doesn't distinguish between:
- Schema mismatch (current issue)
- No matching documents
- Query syntax error
- Index doesn't exist

**Add Logging**:
```python
async def search(self, index_name: str, query: str, k: int = 5, metadata: dict | None = None):
    # ... existing code ...
    
    logger.info(f"Search on index '{index_name}': query='{query}', k={k}")
    
    results = await asyncio.to_thread(...)
    
    logger.info(f"Search returned {len(results.docs)} documents (total: {results.total})")
    
    if len(results.docs) == 0 and results.total > 0:
        logger.warning("Search found matches but returned no docs - possible RETURN field mismatch")
    
    # ... build docs ...
    
    return docs
```

---

## Testing Checklist

After applying the fix:

- [ ] Drop existing index: `FT.DROPINDEX index_name`
- [ ] Deploy updated `create_index` code (without `$.` prefixes)
- [ ] Upload a test document
- [ ] Verify index created: `FT.INFO index_name`
- [ ] Check index attributes don't have `$` prefix
- [ ] Run test search: `FT.SEARCH index_name * LIMIT 0 1`
- [ ] Verify search returns content field with actual text
- [ ] Test vector search through your API
- [ ] Verify RAG retrieval returns non-empty documents

---

## Why This Bug Is Hard to Detect

1. **No Error Messages**: Redis accepts the schema and creates the index successfully
2. **Silent Indexing Failure**: Documents are stored but not properly indexed
3. **Partial Success**: Index reports it has documents (`num_docs > 0`)
4. **Search Returns "Success"**: Query returns with `total=0` or docs with empty fields
5. **Data Visible in Redis Insight**: HGETALL shows all fields exist

This combination makes it appear that everything is working until you try to retrieve data.

---

## Long-term Recommendations

### 1. Add Schema Validation
```python
def validate_schema(self, index_type: IndexType, schema: tuple) -> None:
    """Validate schema matches index type."""
    for field in schema:
        field_name = getattr(field, 'name', '')
        if index_type == IndexType.HASH and field_name.startswith('$.'):
            raise ValueError(
                f"HASH index cannot use JSON path syntax. "
                f"Field '{field_name}' should be '{field_name.lstrip('$.')}''"
            )
        if index_type == IndexType.JSON and not field_name.startswith('$.'):
            logger.warning(
                f"JSON index field '{field_name}' doesn't use JSON path syntax. "
                "Consider using '$.{field_name}'"
            )
```

### 2. Add Index Health Check
```python
async def check_index_health(self, index_name: str) -> dict:
    """Check if index is working correctly."""
    try:
        info = self.client.ft(index_name).info()
        num_docs = info.get("num_docs", 0)
        
        # Try a simple search
        test_query = Query("*").return_fields("content").dialect(2)
        results = self.client.ft(index_name).search(test_query)
        
        # Check if we can actually retrieve content
        has_content = False
        if results.docs:
            first_doc = results.docs[0]
            content = getattr(first_doc, "content", None)
            has_content = content is not None and content != ""
        
        return {
            "status": "healthy" if has_content or num_docs == 0 else "degraded",
            "num_docs": num_docs,
            "can_retrieve_content": has_content,
            "warning": None if has_content or num_docs == 0 else "Index has documents but cannot retrieve content - schema mismatch likely"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

### 3. Document the Schema Clearly
Add comments and type hints to make the HASH vs JSON distinction clear:

```python
def create_index(
    self, 
    index_name: str, 
    prefix: str | None = None,
    index_type: IndexType = IndexType.HASH  # Explicit parameter
) -> None:
    """
    Create RediSearch index.
    
    IMPORTANT: 
    - For HASH indexes: Use plain field names (e.g., "content", "filename")
    - For JSON indexes: Use JSON paths (e.g., "$.content", "$.metadata.filename")
    
    Current implementation uses HASH storage with HASH field names.
    """
```

---

## Conclusion

**The Problem**: Mixing JSON path syntax (`$.field`) with HASH storage type  
**The Impact**: Complete retrieval failure despite data existing  
**The Fix**: Remove `$.` prefix from all field names in schema  
**The Migration**: Drop and recreate index with correct schema  

This is a **critical bug** that requires **immediate action**. The good news is that your data is intact and the fix is straightforward once you understand the root cause.

After fixing:
1. Your existing data will be automatically re-indexed
2. Searches will return full document content
3. RAG retrieval will work as expected

**Estimated Time to Fix**: 30 minutes (schema update + index recreation + testing)  
**Risk Level**: Low (data is preserved, only index needs recreation)  
**Impact**: High (fixes 100% of empty result issues)