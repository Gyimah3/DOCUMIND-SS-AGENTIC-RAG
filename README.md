# DocuMindSS

**DocuMind for Seedstars (SS)**

An intelligent document processing and RAG (Retrieval-Augmented Generation) system that enables natural language conversations with your documents.

## Features

- **Multi-format Document Support**: PDF, DOCX, XLSX, CSV, PPTX, Markdown, HTML, and plain text
- **Agentic RAG**: Intent-aware query routing using LangGraph for intelligent document retrieval
- **FlashRank Re-ranking**: Cross-encoder re-ranking layer between retrieval and generation for significantly improved answer relevance (see [Re-ranking & Evaluators](#re-ranking--evaluators))
- **Source Citations**: Answers include exact document name and page references (e.g., *"Loan Policy.pdf (Page 3)"*)
- **"I Don't Know" Fallback**: When the answer isn't in the documents, the system says so and suggests which file to upload
- **PII / Sensitive ID Masking**: Regex-based redaction of SSNs, card numbers, IBANs, account numbers, emails, and phone numbers in all responses and retrieved chunks before they reach the user
- **Multimodal Processing**: Extract and index images from documents with AI-generated descriptions
- **Dedicated Table Extraction**: Preserves table structure from PDFs and documents
- **Multi-tenant Architecture**: Secure user isolation with per-user document indexes
- **Vector Search**: Redis-powered semantic search with OpenAI embeddings
- **Real-time Streaming**: Token-by-token response streaming via NDJSON

## Tech Stack

- **Backend**: FastAPI, Python 3.12
- **Database**: PostgreSQL 16 (metadata), Redis Stack (vectors)
- **AI/ML**: OpenAI GPT-4, LangChain, LangGraph, FlashRank (ONNX re-ranker)
- **Storage**: AWS S3 for document storage
- **Document Processing**: PyMuPDF, pdfplumber, python-docx, unstructured

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key
- AWS S3 bucket and credentials

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd DocuMindSS
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Start the services**
   ```bash
   docker-compose up -d --build
   ```

4. **Run database migrations**
   ```bash
   docker-compose exec app alembic upgrade head
   ```

5. **Access the application**
   - App: http://localhost:8001
   - RedisInsight: http://localhost:8002

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes |
| `POSTGRES_DB` | Database name (default: documindss) | No |
| `SECRET_KEY` | JWT signing key | Yes |
| `OPENAI_API_KEY` | OpenAI API key for embeddings & chat | Yes |
| `AWS_ACCESS_KEY_ID` | AWS access key | Yes |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Yes |
| `AWS_BUCKET_NAME` | S3 bucket for documents | Yes |
| `AWS_REGION` | AWS region (default: us-east-1) | No |
| `LANGCHAIN_API_KEY` | LangSmith API key (optional) | No |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | No |

## Usage

### Upload Documents

1. Click the **+** button in the chat input
2. Select **Upload Document**
3. Choose a file and set a document identifier
4. Enable multimodal processing if needed for image extraction

### Chat with Documents

- **Tagged Mode**: Attach a specific document to focus the conversation
- **All Documents Mode**: Chat across all your uploaded documents without tagging

### Document Management

- Navigate to **Documents** in the sidebar to view, preview, and delete documents
- Each document is indexed in Redis for fast semantic search

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/login` | POST | User authentication |
| `/auth/register` | POST | User registration |
| `/api/v1/vectorstore/add-document` | POST | Upload and index document |
| `/api/v1/vectorstore/list-indexes` | GET | List user's document indexes |
| `/api/v1/documents/{index}/list` | GET | List documents in an index |
| `/api/v1/rag/chat` | POST | Chat with documents (streaming) |

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

## Development

### Local Development (without Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn main:app --reload --port 8001
```

### Project Structure

```
DocuMindSS/
├── api/routes/          # API endpoints
├── app/                  # Application config & container
├── database/            # SQLAlchemy models & session
├── document/            # Document loaders & processors
├── frontend/            # Static HTML/CSS/JS
├── middleware/          # Auth middleware
├── rag/                 # LangGraph RAG implementation
├── services/            # Redis vectorstore, storage
├── main.py              # Application entry point
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## RAG Pipeline Overview

```
User query
  → Intent detection (DOCUMENT_QUERY vs GENERAL_CONVERSATION)
  → Query rephrasing (better search terms)
  → Embedding (OpenAI text-embedding-3-small)
  → Redis KNN retrieval (k × 3 candidates)
  → FlashRank cross-encoder re-ranking → top k
  → PII masking (regex-based redaction)
  → LLM generation with source citations
  → Streamed response (NDJSON)
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Re-ranker | `services/reranker.py` | Singleton FlashRank cross-encoder for relevance scoring |
| PII Masker | `services/pii_masker.py` | Regex-based redaction of sensitive IDs, emails, phone numbers |
| Retriever | `services/redis_vectorstore.py` | Redis vector search with optional re-ranking integration |
| RAG Tools | `rag/tools.py` | LangChain tool with metadata-aware result formatting |
| RAG Engine | `rag/engine.py` | LangGraph workflow: intent → rephrase → retrieve → generate |
| Prompts | `rag/prompts.py` | System prompts for citation format and "I don't know" fallback |

## Re-ranking & Evaluators

### FlashRank Re-ranking (Implemented)

Re-ranking acts as a **retrieval relevance evaluator** built directly into the pipeline. Instead of only relying on cosine similarity (which approximates relevance at the embedding level), the FlashRank cross-encoder scores each candidate passage *jointly with the query* to produce a more accurate relevance ordering.

**How it works:**
1. Over-retrieve `k × 3` candidates from Redis via cosine similarity
2. Score all candidates with FlashRank (`ms-marco-MiniLM-L-12-v2`, ONNX, ~10-50ms)
3. Return the top `k` by re-rank score
4. Each document's metadata includes both `vector_score` (cosine) and `rerank_score` (cross-encoder)

This directly addresses the **Retrieval Relevance** evaluator — measuring *"how relevant are my retrieved results for this query"*.

### Evaluator Roadmap

Re-ranking is one of several evaluator dimensions for RAG quality. We plan to incorporate additional evaluators in future iterations:

| Evaluator | What it Measures | Status |
|-----------|-----------------|--------|
| **Retrieval Relevance** | Retrieved docs vs input — are the right chunks being found? | **Implemented** (FlashRank re-ranking) |
| **Groundedness** | Response vs retrieved docs — does the answer stay faithful to the context? | Planned |
| **Relevance** | Response vs input — does the answer actually address the user's question? | Planned |
| **Correctness** | Response vs reference answer — is the answer factually correct? | Planned |

**Planned approach:** Use LLM-as-judge evaluators (via LangSmith or custom) to assess each dimension. These can run offline on test datasets or as real-time guardrails.

## License

MIT License

## Support

For issues and feature requests, please open an issue on GitHub.
