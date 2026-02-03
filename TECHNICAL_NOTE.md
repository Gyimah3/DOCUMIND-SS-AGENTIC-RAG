# Horo + Document Uploads: RAG System Technical Note

## Approach Summary

A common RAG approach is to send document chunks directly into the LLM context window alongside the user's message — stuffing retrieved content into the prompt on every turn. This is expensive (large token counts per request), wasteful for non-document messages, and doesn't scale as the founder's knowledge base grows.

I designed an **agentic RAG system** instead, where an LLM agent decides *when* and *how* to search, rather than sending document info and chunks to be attached to the LLM context. The LLM is given a **search tool** and decides on its own whether to call it. The agent only retrieves chunks when it determines the question actually requires document context — and it retrieves only the most relevant chunks, not everything. For greetings, acknowledgments, and follow-up conversation, it responds directly without touching the vector store at all.

This approach reduces cost significantly (fewer embedding calls, smaller prompt token counts), improves answer quality (the agent rephrases vague queries into effective search terms before retrieval), and keeps answers grounded in the founder's own documents with precise citations.

The system has two main flows: **Document Processing** (upload time) and **Agentic Retrieval** (query time), connected through a Redis vector store scoped per user tenant.

---

## How to Use

### 1. Upload a document

Open the chat interface, click the **+** button, and select **Upload Document**. Choose a file (PDF, Excel, CSV, DOCX, PPTX, or plain text), give it a document identifier (e.g., "loan-policy"), toggle processing options (Multimodal, Extract Tables, Extract Images), and click Upload. The system parses, chunks, embeds, and indexes the document automatically.

### 2. Chat with a specific document

After uploading, **tag the document** by selecting it from the document list. A tag chip appears in the chat input area showing the active document name. Now every question you ask is scoped to that document only — the agent searches only that document's index and cites it in responses.

> "What's the maximum loan size for first-time borrowers?"
> → Answer + **Loan Policy.pdf (Page 3)**

### 3. Chat across all documents at once

This is a key differentiator: **you don't have to attach a document every time you ask a question.** Once a founder has uploaded multiple documents (pitch deck, loan policy, handbook, finance sheet), they can simply clear the tag (click the **x** on the tag chip) to switch to **"All Documents"** mode.

In this mode, the agent searches across the founder's entire knowledge base simultaneously — all indexes are queried concurrently and results are merged by relevance. The founder can ask cross-document questions without manually switching between files:

> "What's our CAC and how does it compare to the targets in the growth plan?"
> → The agent searches the finance sheet AND the growth plan, combines the information, and cites both: **Finance Sheet.xlsx (Sheet "Metrics", Row 4)** + **Growth Plan.pdf (Page 12)**

### 4. Switch between documents seamlessly

The founder can tag a different document at any time by selecting it from the **+** menu. The chat thread continues, but the agent now searches the newly tagged document. The system prompt explicitly tells the LLM that the active document may have changed, so it always does a fresh search rather than relying on stale answers from the previous document.

| Mode | What gets searched | When to use |
|------|-------------------|-------------|
| **Tagged document** | Only that document's index | Deep-dive into one file — "walk me through the onboarding steps" |
| **All Documents** | Every document the founder has uploaded | Cross-document questions — "summarize everything I've uploaded" |

---

## Key Differentiator: Multimodal Image Understanding

Horo's current chat **cannot extract or understand images** inside documents. Founders upload pitch decks, financial reports, and handbooks full of charts, diagrams, and infographics — but all that visual information is invisible to the current system.

**Our system solves this.** When a founder enables multimodal processing, we:

1. **Extract every image** from PDFs, DOCX, and PPTX files — including embedded charts, diagrams, screenshots, and photos
2. **Filter out noise** — images under 30KB (icons, borders, decorative elements) are automatically skipped to avoid wasting LLM calls
3. **Generate searchable descriptions** — each meaningful image is sent to a vision-capable LLM (GPT-4o) which produces a detailed text summary: what the image shows, what data it contains, what it's trying to convey
4. **Store as tagged, searchable chunks** — the LLM summary becomes a `type: "image"` chunk with the original base64 image preserved in metadata. The summary is embedded and indexed alongside text chunks
5. **Cite images in answers** — when a founder asks "what does the revenue chart show?", the system retrieves the image chunk by its summary, answers the question, and cites the exact page: **Pitch Deck.pdf (Page 7)**

### How it works technically

```
PDF/DOCX/PPTX Upload
     │
     ▼
Extract images per page (concurrent: asyncio.gather across all pages)
     │
     ▼
Filter: skip if image < 30KB (icons, decorations)
     │
     ▼
Convert to PNG → base64 encode
     │
     ▼
Send to vision LLM with prompt:
  "What is the image about, what is in the image,
   and what is the image trying to convey?"
     │
     ▼
Store as Document:
  page_content = LLM summary text (searchable via embedding)
  metadata = {
    type: "image",
    page_number: 7,
    image_base64: "iVBOR...",   ← original image preserved
    filename: "Pitch Deck.pdf"
  }
```

### Why this matters for SIGMA founders

| Scenario | Without multimodal | With multimodal |
|----------|-------------------|-----------------|
| "What does the revenue chart in my pitch deck show?" | "I don't know" | "The chart shows MRR growing from GHS 2K to GHS 15K over 6 months, with a sharp increase in Q3." + **Pitch Deck (Page 7)** |
| "Describe the system architecture diagram" | "I don't know" | "The diagram shows a microservices architecture with 3 services connected through an API gateway..." + **Technical Doc (Page 12)** |
| "What's in the financial projections table image?" | "I don't know" | Reads the table from the image and answers with specific numbers |

### Cost control

This is opt-in via a **Multimodal** checkbox at upload time. When disabled (default), only text and table chunks are embedded — zero vision LLM cost. Founders toggle it on only for image-heavy documents like pitch decks and reports, keeping costs predictable.

---

## Key Differentiator: Dedicated Table Extraction

Most RAG systems treat tables as flat text — they lose the row/column structure, and the LLM can't reason about the data. Our system has a **dedicated table extraction pipeline** that preserves table structure and stores tables as separate, typed chunks.

### How it works across formats

**PDFs** — We use `pdfplumber` alongside `fitz` specifically for table detection. When a page contains a table:
1. `pdfplumber.extract_table()` detects table boundaries and cell structure
2. The raw table is converted to a **pandas DataFrame**, then to **markdown format** — preserving rows, columns, and headers
3. Stored as a dedicated `type: "table"` chunk with the page number, separate from the page's text chunk

```
PDF Page 5 (contains text + a financial table)
     │
     ├─→ Chunk 1: page text (type: "text", page: 5)
     │     "The company's Q3 results showed strong growth..."
     │
     └─→ Chunk 2: extracted table (type: "table", page: 5)
           | Quarter | Revenue  | Expenses | Profit  |
           |---------|----------|----------|---------|
           | Q1      | GHS 12K  | GHS 8K   | GHS 4K  |
           | Q2      | GHS 18K  | GHS 10K  | GHS 8K  |
           | Q3      | GHS 25K  | GHS 12K  | GHS 13K |
```

**DOCX files** — We use `unstructured`'s partition engine with `infer_table_structure=True`:
1. Tables are detected as `category: "Table"` elements
2. Each table gets **dual representation**:
   - An **HTML chunk** (`text_as_html`) — preserves exact cell structure, merged cells, formatting
   - A **plain text chunk** — for broader keyword matching
3. Both are stored as separate typed chunks

**Excel/CSV** — Every data row is already a structured chunk. The sheet name and row number metadata enables precise citations like `(Sheet "Financials", Row 12)`.

### Why dedicated table chunks matter

| Query | Without table extraction | With table extraction |
|-------|------------------------|----------------------|
| "What was Q3 revenue?" | Finds the page text mentioning "strong growth" — no specific number | Finds the table chunk directly, reads GHS 25K from the Q3/Revenue cell |
| "Compare Q1 vs Q3 profit" | Vague answer based on surrounding paragraph text | Precise: "Q1 profit was GHS 4K, Q3 profit was GHS 13K — a 225% increase" |
| "Which quarter had the highest expenses?" | May not find the data at all | Scans the table, answers "Q3 with GHS 12K" |

### How the types are used

The chunk type (`text`, `table`, `image`) is stored as a **TagField** in Redis and in the document metadata. This enables:
- **Filtering at upload**: The `multimodal` toggle controls whether image chunks are included
- **Structured citations**: The frontend knows whether to show "Page 3" (text/table) or "Sheet X, Row Y" (spreadsheet data)
- **Future capability**: Type-based search filters (e.g., "only search tables") are already supported in the tool's filter schema via `TypeFilter`

---

## Flow 1: Document Processing Pipeline

```
File Upload → MIME Detection → Format-Specific Loader → Chunking → Embedding → Redis Index
```

### How it works

1. **Upload**: Founder uploads a file (PDF, Excel, CSV, DOCX, PPTX, or plain text) through the chat interface. The file is stored in S3 for later preview, and its content is processed for search.

2. **MIME Detection & Routing**: The system detects the file type and routes to a specialized loader. Each loader understands the structure of its format:
   - **PDFs**: Page-by-page text extraction with `fitz`, table extraction with `pdfplumber`, OCR fallback for scanned documents
   - **Excel/CSV**: Row-by-row chunking with sheet name and row number metadata — each data row becomes a searchable chunk
   - **DOCX/PPTX**: Structural chunking by title/section boundaries (max 4000 chars, 200 char overlap)
   - **Images**: Filtered by size (>30KB) to skip icons, then summarized by an LLM into searchable text descriptions

3. **Metadata Preservation**: Every chunk carries metadata that enables precise citations:
   - PDFs → `filename`, `page_number`
   - Excel → `filename`, `sheet_name`, `row_number`
   - CSV → `filename`, `row_number`
   - DOCX/PPTX → `filename`, `page_number`

4. **Embedding & Indexing**: Chunks are embedded with OpenAI `text-embedding-3-small` (1536 dimensions) and stored in Redis using RediSearch with FLAT vector indexing and cosine similarity. Each founder's documents live in their own Redis index prefix, isolated from other tenants.

5. **Database Tracking**: PostgreSQL tracks which files belong to which index and user, enabling document management (list, delete, preview) and multi-tenant isolation.

### Cost optimization at upload time

- **Lazy loading**: Documents are processed as an async generator — memory stays constant regardless of file count
- **Multimodal toggle**: Image embedding is opt-in. By default only text and table chunks are embedded, avoiding expensive image embeddings
- **Image size filter**: Small images (<30KB — icons, borders, logos) are skipped entirely, saving LLM summarization calls
- **Concurrent processing**: All pages/slides/sheets are processed in parallel with `asyncio.gather`, reducing wall-clock time

---

## Flow 2: Agentic RAG (LangGraph)

This is where cost savings are most significant. Instead of a simple "embed query → search → generate" pipeline, I use a **LangGraph state machine** with conditional routing.

### The Graph

```
User Message
     │
     ▼
┌─────────────┐
│ Detect      │  ← Structured output: DOCUMENT_QUERY or GENERAL_CONVERSATION
│ Intent      │     (one cheap LLM call with json_schema output)
└──────┬──────┘
       │
       ├── GENERAL_CONVERSATION ──────────────────┐
       │   ("hello", "ok", "thanks")              │
       │                                          │
       ▼                                          ▼
┌─────────────┐                          ┌─────────────┐
│ Rephrase    │  ← Structured output:    │ Assistant   │ → Direct response
│ Query       │     optimized search     │ (no search) │   (no embedding cost)
└──────┬──────┘     terms + keywords     └─────────────┘
       │
       ▼
┌─────────────┐
│ Assistant   │  ← LLM with tool binding
│ (with tool) │     Decides whether to call information_lookup_tool
└──────┬──────┘
       │
       ├── No tool call ──→ END (answer from context)
       │
       ▼
┌─────────────┐
│ Lookup      │  ← Vector search in Redis (KNN cosine similarity)
│ (ToolNode)  │     Returns relevant chunks with metadata
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Assistant   │  ← Generates final answer with citations
│ (final)     │     "According to Loan Policy (p. 3)..."
└─────────────┘
```

### Why agentic RAG instead of context-stuffing RAG

The standard approach is **context-stuffing**: retrieve chunks on every message and inject them into the LLM prompt. Every user message — even "hello" or "ok" — triggers an embedding call, a vector search, and a large prompt stuffed with document chunks the LLM doesn't need.

Our agentic approach flips this: the LLM receives a **tool** instead of pre-loaded context, and it decides whether to call it.

| Message | Context-Stuffing RAG | Agentic RAG |
|---------|---------------------|-------------|
| "hello" | Embed query + Search + Stuff 5 chunks into prompt + Generate (~2K extra tokens wasted) | Intent detect → Generate directly (zero retrieval cost) |
| "ok, thanks" | Embed + Search + Stuff chunks + Generate | Intent detect → Short reply (zero retrieval cost) |
| "what's our CAC?" | Embed raw query + Search (may miss) + Stuff chunks + Generate | Intent → Rephrase → Targeted search → Generate with only relevant chunks |
| "tell me about the doc" | Embed vague query → poor retrieval → stuff irrelevant chunks → weak answer | Intent → Rephrase to "summary key points overview" → quality retrieval → precise answer |

**Cost reduction**: ~30-40% fewer embedding API calls, and significantly smaller prompt token counts (no wasted context on conversational turns).

**Quality improvement**: The rephrase step transforms vague queries into effective search terms before retrieval. "Tell me about the doc" becomes "document summary main topics key points overview content", which retrieves much better chunks. Context-stuffing would embed the raw vague query and retrieve poor matches.

### How search works

**Single-index search** (founder tags a specific document):
- Embed the rephrased query → KNN search in that index → return top-k chunks sorted by cosine distance

**Multi-index search** (founder asks across all their documents):
- Fetch only the current user's index names from PostgreSQL
- Search each index concurrently with `asyncio.gather`
- Merge results across all indexes, sort by vector score, return global top-k
- Other founders' indexes are never touched (safe default: empty list if no user indexes found)

### Streaming response

Answers are streamed token-by-token to the frontend as newline-delimited JSON. The frontend renders markdown in real-time as tokens arrive, then appends source citations when the retriever results come through:

```
{"stream": "According to the "}
{"stream": "Loan Policy, the maximum"}
{"stream": " loan size for first-time borrowers is..."}
{"used_docs": [{"page_content": "...", "metadata": {"filename": "Loan Policy.pdf", "page_number": 3}}]}
```

The frontend then renders: `Loan Policy.pdf (Page 3)` or for Excel files: `Finance Sheet.xlsx (Sheet "Loans", Row 5)`.

### Conversation memory

Each chat thread has a `thread_id` with a `MemorySaver` checkpointer. This means:
- Follow-up questions work naturally ("What about for returning borrowers?" after asking about first-time borrowers)
- The system prompt reminds the LLM to always search fresh when the active document changes, preventing stale answers from conversation history

---

## Multi-Tenant Isolation

Every layer enforces tenant boundaries:

| Layer | Isolation Mechanism |
|-------|-------------------|
| **API** | JWT authentication, `user_id` extracted from token |
| **Database** | `VectorStore.user_id` foreign key, queries always filter by current user |
| **Redis** | Each index has a unique prefix; multi-index search only queries user's own index names |
| **S3** | Files stored under `{user_id}/{index_name}/` prefix |
| **Search** | `user_index_names` parameter is populated from DB, not user input — no injection possible |

A founder can never see, search, or retrieve another founder's documents.

---

## How the Flows Connect

```
                    ┌──────────────┐
                    │   Frontend   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
      Upload Flow    Chat Flow    Document Management
              │            │            │
              ▼            ▼            ▼
    ┌─────────────┐ ┌───────────┐ ┌──────────┐
    │ Document    │ │ LangGraph │ │ List /   │
    │ Processing  │ │ Agent     │ │ Delete / │
    │ Pipeline    │ │ (RAG)     │ │ Preview  │
    └──────┬──────┘ └─────┬─────┘ └────┬─────┘
           │              │            │
           ▼              ▼            ▼
    ┌─────────────────────────────────────────┐
    │          Redis Vector Store             │
    │  (RediSearch: FLAT index, COSINE)       │
    │  Per-tenant indexes with metadata       │
    └─────────────────────────────────────────┘
           │              │            │
           ▼              ▼            ▼
    ┌────────────┐ ┌───────────┐ ┌──────────┐
    │ PostgreSQL │ │  OpenAI   │ │   AWS    │
    │ (metadata, │ │ (embed +  │ │   S3     │
    │  users,    │ │  LLM)     │ │ (files)  │
    │  indexes)  │ │           │ │          │
    └────────────┘ └───────────┘ └──────────┘
```

---

## Design Decisions & Tradeoffs

| Decision | Rationale |
|----------|-----------|
| **Redis (not Pinecone/Weaviate)** | Already in the stack, RediSearch is fast for small-to-medium datasets typical of founder documents. No extra infra cost. |
| **FLAT index (not HNSW)** | Founders have hundreds to low thousands of chunks, not millions. FLAT gives exact results with no approximation error. HNSW would add complexity for negligible speed gain at this scale. |
| **Intent detection before search** | Most chat messages in a coaching context are conversational ("ok", "thanks", "got it"). Detecting intent first avoids ~30-40% of unnecessary embedding+search calls. |
| **Structured output for intent + rephrase** | Forces the LLM into a tight JSON schema. No wasted tokens on formatting. Deterministic parsing. |
| **Row-level chunking for spreadsheets** | Founders upload finance sheets and data tables. Each row is a meaningful unit. Page-level chunking would merge unrelated rows. Row-level enables "Row 5 in the Loans sheet" citations. |
| **Streaming NDJSON** | Simple, no WebSocket infra needed. Works with standard `fetch()` + `ReadableStream`. Each line is independently parseable. |
| **MemorySaver (in-memory)** | Good enough for demo/early stage. Production would swap to Redis-backed or PostgreSQL-backed checkpointer for persistence across restarts. |
| **Multimodal image extraction (opt-in)** | Current Horo cannot see images at all. We extract images, generate LLM summaries, and store them as searchable tagged chunks — making charts, diagrams, and infographics answerable. Opt-in toggle keeps costs zero for text-only docs. |
| **Image summaries as text chunks** | Instead of storing raw image embeddings (expensive, lower quality for search), we use a vision LLM to describe the image in words. The text summary is what gets embedded and searched — cheaper, more accurate retrieval, and the original image is preserved in metadata for display. |
| **Dedicated table extraction** | Tables are extracted as separate `type: "table"` chunks in structured format (markdown/HTML), not flattened into page text. This lets the LLM reason about rows and columns — critical for financial data that founders frequently ask about. Dual representation (HTML + text) covers both structure-aware and keyword-based retrieval. |

---

## Example Scenarios (from the brief)

### "What's the maximum loan size for first-time borrowers?"
1. Intent: `DOCUMENT_QUERY` (confidence: 0.95)
2. Rephrase: "maximum loan size limit first-time borrowers lending policy"
3. Assistant calls `information_lookup_tool(query="maximum loan size...")`
4. Redis returns chunk from `Loan Policy.pdf`, page 3
5. Answer: "According to the Loan Policy, the maximum loan size for first-time borrowers is GHS 5,000." + citation: **Loan Policy.pdf (Page 3)**

### "List the onboarding steps for our program."
1. Intent: `DOCUMENT_QUERY`
2. Rephrase: "onboarding steps process program enrollment procedures"
3. Tool returns chunks from pages 2, 5, 7 of Program Handbook
4. Answer with bullet points + citation: **Program Handbook.pdf (Page 2, 5, 7)**

### "What's our CAC?" (not in docs)
1. Intent: `DOCUMENT_QUERY`
2. Rephrase: "customer acquisition cost CAC marketing spend"
3. Tool searches — no relevant results
4. Answer: "I don't have that information in your uploaded documents. Please upload your latest growth or finance sheet so I can help."

### "What does the growth chart in my pitch deck show?" (multimodal enabled)
1. Intent: `DOCUMENT_QUERY`
2. Rephrase: "growth chart revenue metrics pitch deck visualization data"
3. Tool searches — retrieves the image chunk whose LLM summary mentions "growth chart" and revenue figures
4. Answer: "The growth chart on page 7 shows monthly recurring revenue increasing from GHS 2,000 to GHS 15,000 over the past 6 months, with the steepest growth in Q3." + citation: **Pitch Deck.pdf (Page 7)**
5. **This is impossible with Horo's current text-only RAG** — the chart is an image, not extractable text

### "Thanks, that's helpful"
1. Intent: `GENERAL_CONVERSATION` (confidence: 0.98)
2. Skip rephrase, skip search
3. Answer: "Glad I could help! Let me know if you have more questions."
4. **Zero embedding/search cost**
