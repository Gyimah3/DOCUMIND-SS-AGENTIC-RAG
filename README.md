# DocuMind

**Retrieval-augmented question answering over private documents, built so that a wrong answer is harder to produce than no answer.**

FastAPI, LangGraph, Redis vector search, PostgreSQL.

---

## The problem this is built around

Retrieval-augmented generation is usually evaluated on whether it answers. The
harder question is whether it should have. A system that reads a user's loan
policy, employment contract or audit file and invents a plausible clause is
worse than one that says it does not know, because the user has no way to tell
the difference.

Two things follow from that, and they shaped most of the design here:

1. **Retrieval has to be measured, not assumed.** Cosine similarity over
   embeddings approximates relevance; it does not establish it. So the pipeline
   over-retrieves and then re-scores with a cross-encoder, and it keeps both
   scores on every chunk so you can see where they disagree.
2. **Abstention has to be a first-class output.** The model is instructed to say
   "I don't know" and name which document would be needed, rather than answer
   from parametric memory. An answer and a refusal are both results.

Everything else is plumbing in service of those two.

---

## Pipeline

```
User query
  → Intent detection          DOCUMENT_QUERY vs GENERAL_CONVERSATION
  → Query rephrasing          rewrite into better search terms
  → Embedding                 OpenAI text-embedding-3-small
  → Redis KNN retrieval       over-retrieve k x 3 candidates
  → Cross-encoder re-ranking  FlashRank ms-marco-MiniLM-L-12-v2, ONNX, ~10-50ms
  → PII masking               regex redaction before the model sees the chunk
  → Generation                with source citations
  → Streamed response         NDJSON, token by token
```

The intent step exists so that "hello" does not trigger a vector search, and so
that a question the documents cannot answer is routed to a refusal rather than
to the model's own knowledge.

PII masking runs **before** retrieved chunks reach the model, not after
generation. Masking the output only hides leaks that have already happened.

---

## Retrieval evaluation

Re-ranking here is a **retrieval relevance evaluator built into the request
path**, not a quality nicety.

1. Over-retrieve `k x 3` candidates from Redis by cosine similarity
2. Score every candidate with FlashRank, which encodes query and passage
   *jointly* rather than comparing two independent embeddings
3. Keep the top `k` by cross-encoder score
4. Attach both `vector_score` and `rerank_score` to each chunk's metadata

Step 4 is the part worth keeping. When the two scores disagree sharply, that is
a signal the embedding space is wrong for this query, and it is visible per
request rather than hidden in an aggregate.

### What is not measured yet

Stated plainly, because an unmeasured dimension that is listed as "planned"
somewhere is easy to mistake for one that is handled.

| Dimension | Question it answers | Status |
|---|---|---|
| Retrieval relevance | Are the right chunks being found? | **Implemented** via cross-encoder re-ranking |
| Groundedness | Does the answer stay inside the retrieved context? | Not implemented |
| Answer relevance | Does the answer address what was asked? | Not implemented |
| Correctness | Is the answer right against a reference? | Not implemented |

The abstention behaviour is **prompt-level, not verified**. The system is
instructed to refuse when the context does not support an answer, and in use it
does, but there is no held-out set measuring how often it refuses when it should
have answered, or answers when it should have refused. That false-refusal and
false-answer rate is the obvious next thing to measure, and the honest position
is that it currently is not.

The intended approach for the remaining three is LLM-as-judge scoring, offline
on a labelled set first, then as a request-time guardrail only if the judge
itself survives a negative control.

---

## Design decisions worth explaining

**Why a cross-encoder rather than a bigger embedding model.** A bi-encoder has
to commit to a representation of the passage before it has seen the query.
FlashRank runs the pair together and costs 10 to 50ms on ONNX, which fits inside
a streaming response budget.

**Why over-retrieve by 3x.** Re-ranking can only reorder what retrieval returned.
Too small a candidate pool and the cross-encoder has nothing to fix.

**Why Redis rather than a dedicated vector database.** Multi-tenant isolation is
per-user index rather than metadata filtering, which removes a class of
cross-tenant leakage at the storage layer instead of relying on query-time
filters being written correctly every time.

**Why source citations are mandatory in the prompt.** A citation to a document
and page is checkable by the user. An uncited answer is not.

**Why images go through a vision model into text.** Keeping one retrieval path
rather than two avoids having to fuse rankings from two incomparable scoring
systems.

---

## Features

- Multi-format ingestion: PDF, DOCX, XLSX, CSV, PPTX, Markdown, HTML, plain text
- Agentic RAG with intent-aware routing (LangGraph)
- Cross-encoder re-ranking between retrieval and generation
- Source citations with document name and page, for example *Loan Policy.pdf (Page 3)*
- "I don't know" fallback that names the document that would be needed
- PII redaction of SSNs, card numbers, IBANs, account numbers, emails, phone numbers
- Multimodal: images extracted and indexed via generated descriptions
- Table extraction that preserves structure
- Multi-tenant isolation with per-user document indexes
- Token-by-token streaming over NDJSON

## Tech stack

FastAPI, Python 3.12 · PostgreSQL 16 for metadata · Redis Stack for vectors ·
OpenAI GPT-4 and text-embedding-3-small · LangChain and LangGraph ·
FlashRank ONNX re-ranker · AWS S3 · PyMuPDF, pdfplumber, python-docx, unstructured

---

## Quick start

Prerequisites: Docker and Docker Compose, an OpenAI API key, an S3 bucket.

```bash
git clone <your-repo-url>
cd DocuMindSS
cp .env.example .env            # add your credentials
docker-compose up -d --build
docker-compose exec app alembic upgrade head
```

App at http://localhost:8001 · RedisInsight at http://localhost:8002

### Without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

## Environment variables

| Variable | Description | Required |
|---|---|---|
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes |
| `POSTGRES_DB` | Database name, default `documindss` | No |
| `SECRET_KEY` | JWT signing key | Yes |
| `OPENAI_API_KEY` | Embeddings and chat | Yes |
| `AWS_ACCESS_KEY_ID` | AWS access key | Yes |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Yes |
| `AWS_BUCKET_NAME` | S3 bucket for documents | Yes |
| `AWS_REGION` | Default `us-east-1` | No |
| `LANGCHAIN_API_KEY` | LangSmith tracing | No |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | No |

## API

| Endpoint | Method | Description |
|---|---|---|
| `/auth/register` | POST | User registration |
| `/auth/login` | POST | Authentication |
| `/api/v1/vectorstore/add-document` | POST | Upload and index a document |
| `/api/v1/vectorstore/list-indexes` | GET | List the user's indexes |
| `/api/v1/documents/{index}/list` | GET | List documents in an index |
| `/api/v1/rag/chat` | POST | Chat with documents, streaming |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   FastAPI       │────▶│   PostgreSQL    │
│   (HTML/JS)     │     │   Backend       │     │   (Metadata)    │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            ┌───────────┐ ┌───────────┐ ┌───────────┐
            │  Redis    │ │  OpenAI   │ │  AWS S3   │
            │  (Vector) │ │  (LLM)    │ │  (Files)  │
            └───────────┘ └───────────┘ └───────────┘
```

## Where to look in the code

| Component | File | Purpose |
|---|---|---|
| RAG engine | `rag/engine.py` | LangGraph workflow: intent → rephrase → retrieve → generate |
| Prompts | `rag/prompts.py` | Citation format and the "I don't know" instruction |
| RAG tools | `rag/tools.py` | LangChain tool with metadata-aware formatting |
| Re-ranker | `services/reranker.py` | Singleton FlashRank cross-encoder |
| Retriever | `services/redis_vectorstore.py` | Redis vector search, re-ranking integration |
| PII masker | `services/pii_masker.py` | Regex redaction of sensitive identifiers |
| Loaders | `document/document_loaders/` | One loader per supported format |

Longer write-ups: [`RAG_APPROACH.md`](./RAG_APPROACH.md),
[`TECHNICAL_NOTE.md`](./TECHNICAL_NOTE.md), [`docs/RERANKING.md`](./docs/RERANKING.md).

```
DocuMindSS/
├── api/routes/     # API endpoints
├── app/            # Config and DI container
├── database/       # SQLAlchemy models and session
├── document/       # Loaders and processors
├── frontend/       # Static HTML/CSS/JS
├── middleware/     # Auth
├── rag/            # LangGraph RAG implementation
├── services/       # Redis vectorstore, re-ranker, PII masking, storage
└── main.py
```

## Usage

Upload through the **+** button in the chat input, set a document identifier,
and enable multimodal processing if the file has images worth indexing.

Tagged mode attaches one document to the conversation. All-documents mode
searches across everything the user has uploaded. Documents can be previewed and
deleted from the sidebar.

## License

MIT

---

Built by [Gideon Gyimah](https://gyimah3.github.io). Issues and pull requests welcome.
