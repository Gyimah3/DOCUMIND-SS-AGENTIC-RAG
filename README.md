# DocuMindSS

**DocuMind for Seedstars (SS)**

An intelligent document processing and RAG (Retrieval-Augmented Generation) system that enables natural language conversations with your documents.

## Features

- **Multi-format Document Support**: PDF, DOCX, XLSX, CSV, PPTX, Markdown, HTML, and plain text
- **Agentic RAG**: Intent-aware query routing using LangGraph for intelligent document retrieval
- **Multimodal Processing**: Extract and index images from documents with AI-generated descriptions
- **Dedicated Table Extraction**: Preserves table structure from PDFs and documents
- **Multi-tenant Architecture**: Secure user isolation with per-user document indexes
- **Vector Search**: Redis-powered semantic search with OpenAI embeddings
- **Real-time Streaming**: Token-by-token response streaming via NDJSON

## Tech Stack

- **Backend**: FastAPI, Python 3.12
- **Database**: PostgreSQL 16 (metadata), Redis Stack (vectors)
- **AI/ML**: OpenAI GPT-4, LangChain, LangGraph
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

## License

MIT License

## Support

For issues and feature requests, please open an issue on GitHub.
