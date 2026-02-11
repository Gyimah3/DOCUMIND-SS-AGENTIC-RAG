# RAG System for Horo + Document Uploads

## Approach

Instead of requiring founders to attach a document to every message, this solution allows them to **upload all their documents once**, then:
- **Tag a specific document** to ask questions within that document alone
- **Tag all documents** (or clear the tag) to search across their entire knowledge base

Instead of stuffing document chunks into every LLM prompt, I built an **agentic RAG** system where the LLM decides when to search. The agent has a search tool and only retrieves documents when the question actually requires it — greetings, acknowledgments, and follow-ups skip retrieval entirely.

This reduces cost (fewer embedding calls, smaller prompts) and improves quality (the agent rephrases vague queries before searching).

---

## Document Processing Flow

```
┌──────────────┐
│  File Upload │  (PDF, DOCX, XLSX, CSV, PPTX, TXT)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ MIME Detect  │  Detect file type by extension/magic number
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                  Format-Specific Loaders                      │
├───────────────┬───────────────┬───────────────┬──────────────┤
│     PDF       │    DOCX       │  Excel/CSV    │    PPTX      │
├───────────────┼───────────────┼───────────────┼──────────────┤
│ • Page text   │ • Structural  │ • Row-level   │ • Slide text │
│   (PyMuPDF)   │   chunking    │   chunks      │ • Tables     │
│ • Tables      │   by title    │ • Sheet name  │ • Images     │
│   (pdfplumber)│ • Tables      │ • Row number  │              │
│ • Images*     │ • Images*     │   metadata    │              │
└───────────────┴───────────────┴───────────────┴──────────────┘
       │
       ▼
┌──────────────┐
│   Chunking   │  Max 4000 chars, 200 char overlap
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embedding   │  OpenAI text-embedding-3-small (1536 dims)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Redis Index  │  RediSearch FLAT index, cosine similarity
└──────────────┘

* Images: Extract → Filter (<30KB skipped) → Vision LLM summary → Searchable text chunk
```

**Metadata preserved per chunk:**
- PDFs → `filename`, `page_number`, `type`
- Excel → `filename`, `sheet_name`, `row_number`
- DOCX/PPTX → `filename`, `page_number`

---

## Agentic RAG Flow (LangGraph)

```
                         User Message
                              │
                              ▼
                    ┌─────────────────┐
                    │  Detect Intent  │
                    │  (Structured    │
                    │   Output LLM)   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     GENERAL_CONVERSATION          DOCUMENT_QUERY
     ("hello", "thanks")                    │
              │                             ▼
              │                  ┌─────────────────┐
              │                  │ Rephrase Query  │
              │                  │ (Structured     │
              │                  │  Output LLM)    │
              │                  └────────┬────────┘
              │                           │
              │                           ▼
              │                  ┌─────────────────┐
              │                  │   LLM + Tool    │
              │                  │  (Tool-bound    │
              │                  │   GPT-4)        │
              │                  └────────┬────────┘
              │                           │
              │              ┌────────────┴────────────┐
              │              │                         │
              │         Tool Called              No Tool Call
              │              │                         │
              │              ▼                         │
              │     ┌─────────────────┐               │
              │     │  Vector Search  │               │
              │     │  (Redis KNN)    │               │
              │     │  Top-k chunks   │               │
              │     └────────┬────────┘               │
              │              │                         │
              │              ▼                         │
              │     ┌─────────────────┐               │
              │     │ Generate Answer │◄──────────────┘
              │     │ with Citations  │
              │     └────────┬────────┘
              │              │
              ▼              ▼
        ┌─────────────────────────┐
        │   Streaming Response    │
        │   (NDJSON tokens +      │
        │    used_docs metadata)  │
        └─────────────────────────┘
```

**Why this flow?**
- Intent detection skips retrieval for ~30-40% of messages (conversational turns)
- Query rephrasing transforms "tell me about the doc" → "document summary main topics key points"
- Tool binding lets the LLM decide if search is needed based on context

---

## Key Differentiators

### 1. Multimodal Image Understanding

Current Horo chat **cannot see images** inside documents. My system extracts and understands them:

```
PDF/DOCX with charts, diagrams, infographics
        │
        ▼
Extract images per page (concurrent processing)
        │
        ▼
Filter: Skip if < 30KB (icons, decorations)
        │
        ▼
Vision LLM: "What is this image about?"
        │
        ▼
Store as searchable text chunk with base64 preserved
        │
        ▼
User asks: "What does the revenue chart show?"
        │
        ▼
System retrieves image chunk, answers with data from chart
```

**Example:**
> "What does the growth chart in my pitch deck show?"
> → "The chart shows MRR growing from GHS 2K to GHS 15K over 6 months." + **Pitch Deck.pdf (Page 7)**

### 2. Dedicated Table Extraction

Tables are extracted as **separate structured chunks**, not flattened text:

- **PDFs**: `pdfplumber` → pandas DataFrame → markdown table
- **DOCX**: `unstructured` with `infer_table_structure=True` → HTML + text
- **Excel/CSV**: Each row is a chunk with sheet/row metadata

**Example:**
> "What was Q3 revenue?"
> → Retrieves table chunk, reads exact cell value: "GHS 25K"

### 3. All-Documents Mode

Founders don't need to attach a document every time. They can:
- **Tag a document**: Search only that document's index
- **Clear tag**: Search across ALL uploaded documents simultaneously

```
Multi-index search:
  1. Get user's index names from DB
  2. Search each index concurrently (asyncio.gather)
  3. Merge results by vector_score
  4. Return global top-k
```

**Example:**
> "What's our CAC and how does it compare to targets?"
> → Searches Finance Sheet AND Growth Plan, cites both

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Intent Detection** | Skips search for conversational messages (~30-40% cost savings) |
| **Query Rephrasing** | Transforms vague queries into effective search terms before retrieval |
| **Multi-Index Search** | Search across all documents or tag a specific one |
| **Multimodal** | Extract images from PDFs/DOCX, generate LLM summaries, store as searchable chunks |
| **Table Extraction** | Dedicated pipeline preserves table structure (markdown format) |
| **Streaming** | Token-by-token NDJSON responses with source citations |

---

## Retrieved Documents Format

When the user asks a question, we retrieve **top-3 relevant chunks** with full metadata:

```json
{
  "used_docs": [
    {
      "page_content": "Maximum loan for first-time borrowers is GHS 5,000...",
      "metadata": {
        "filename": "Loan Policy.pdf",
        "page_number": 3,
        "type": "text",
        "vector_score": 0.87
      }
    }
  ]
}
```

The frontend displays: `Loan Policy.pdf (Page 3)` or for Excel: `Finance.xlsx (Sheet "Loans", Row 5)`

---

## Multi-Tenant Isolation

- JWT authentication with user_id scoping
- Each user's documents in separate Redis index prefixes
- Database queries always filter by current user
- S3 files stored under `{user_id}/{index_name}/`

---

## Tech Stack

- **Backend**: FastAPI, Python 3.12
- **Vector Store**: Redis Stack (RediSearch, FLAT index, cosine similarity)
- **Embeddings**: OpenAI text-embedding-3-small (1536 dims)
- **LLM**: GPT-4 with tool binding via LangGraph
- **Storage**: PostgreSQL (metadata), AWS S3 (files)
