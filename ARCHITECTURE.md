# Python Services - Architecture

## Overview

This repository contains two Python services that share common code but run as separate deployments:

1. **LlamaIndex Core Service** - HTTP API for queries and document processing
2. **Temporal Worker Service** - Background workers for async document indexing

## Core Service Architecture

### Design Principles

1. **Async-First**: All I/O operations use async/await for better concurrency
2. **Stateless**: Service can be scaled horizontally
3. **FastAPI**: Modern async web framework with automatic docs
4. **LlamaIndex**: RAG framework for retrieval-augmented generation

### Component Diagram

```
┌──────────────────────────────────────────────────────────┐
│           LlamaIndex Core Service                          │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │ FastAPI     │───►│  Services   │───►│  External    │  │
│  │ HTTP Layer  │    │  Layer      │    │  Services    │  │
│  └─────────────┘    └─────────────┘    └──────────────┘  │
│       │                  │                  │             │
│       │                  │                  │             │
│       ▼                  ▼                  ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │ Routes      │    │ RAG Engine  │    │ Qdrant       │  │
│  │ • Queries   │    │ • Retrieval │    │              │  │
│  │ • Convos    │    │ • Generation│    │ Postgres     │  │
│  │ • Docs      │    │             │    │ OpenAI       │  │
│  └─────────────┘    └─────────────┘    └──────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────┘
```

### Request Flow: Query

```
1. Gateway POST /query
   ↓
2. FastAPI route handler validates request
   ↓
3. RAGService.query() called
   ↓
4. Retrieve conversation history from Postgres
   ↓
5. Vector similarity search in Qdrant (top-k=5)
   ↓
6. Build RAG prompt with retrieved chunks
   ↓
7. Stream LLM response via SSE
   ↓
8. Save query/response to Postgres
   ↓
9. Return stream to Gateway
```

### Request Flow: Document Processing

```
1. Temporal Worker: Document downloaded from S3
   ↓
2. Parse document (CPU-only loader)
   ↓
3. Split into chunks (512 tokens, 50 overlap)
   ↓
4. Generate embeddings (OpenAI API)
   ↓
5. Upsert vectors to Qdrant
   ↓
6. Update document metadata in Postgres
   ↓
7. Return success to Temporal
```

## RAG Engine Design

### Components

```python
class RAGEngine:
    """Retrieval-Augmented Generation Engine"""

    def __init__(self, qdrant_client, openai_client, db):
        self.qdrant = qdrant_client
        self.openai = openai_client
        self.db = db
        self.vector_store = QdrantVectorStore(qdrant_client)
        self.index = VectorStoreIndex.from_vector_store(vector_store)

    async def query(
        self,
        query_text: str,
        conversation_id: str | None = None,
        stream: bool = True
    ) -> AsyncIterator[str] | str:
        """Perform RAG query with optional streaming"""

        # 1. Retrieve conversation history
        history = await self._get_conversation_history(conversation_id)

        # 2. Vector similarity search
        nodes = await self._retrieve_relevant_chunks(query_text)

        # 3. Build RAG prompt
        prompt = self._build_rag_prompt(query_text, nodes, history)

        # 4. Generate response
        if stream:
            return self._stream_response(prompt)
        else:
            return await self._generate_response(prompt)

    async def _retrieve_relevant_chunks(
        self, query: str, top_k: int = 5
    ) -> list[Node]:
        """Retrieve top-k relevant chunks from Qdrant"""

        query_embedding = await self.openai.embeddings.create(query)
        results = self.qdrant.search(
            collection_name="documents",
            query_vector=query_embedding,
            limit=top_k
        )
        return [self._node_from_qdrant_result(r) for r in results]
```

### Retrieval Strategy

1. **Vector Search**: Cosine similarity on OpenAI embeddings
2. **Top-K**: Retrieve 5 most relevant chunks
3. **Re-ranking**: Not implemented (future enhancement)
4. **Filters**: Can filter by document_id, upload_date

### Prompt Template

```
You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context_chunks}

Conversation History:
{conversation_history}

User: {query}

Assistant:
```

## Database Design

### Schema Overview

```sql
-- Conversations
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Messages
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Documents
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    s3_key TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'indexing', 'complete', 'failed')),
    error_message TEXT,
    indexed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
```

### Connection Pooling

```python
# Using asyncpg for async PostgreSQL
class Database:
    def __init__(self, url: str, pool_size: int = 10):
        self.url = url
        self.pool_size = pool_size
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            self.url,
            min_size=2,
            max_size=self.pool_size
        )

    async def execute(self, query: str, *args):
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args):
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)
```

## Qdrant Integration

### Collection Configuration

```python
COLLECTION_CONFIG = {
    "collection_name": "documents",
    "vectors_config": {
        "size": 1536,  # OpenAI text-embedding-3-small
        "distance": "Cosine"
    },
    "optimizers_config": {
        "indexing_threshold": 10000
    }
}
```

### Vector Upsert

```python
async def upsert_documents(
    self,
    document_id: str,
    chunks: list[DocumentChunk]
):
    """Upsert document chunks to Qdrant"""

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk.embedding,
            payload={
                "document_id": document_id,
                "content": chunk.content,
                "chunk_index": chunk.index,
                "filename": chunk.filename,
                "upload_timestamp": chunk.upload_timestamp
            }
        )
        for chunk in chunks
    ]

    await self.qdrant_client.upsert(
        collection_name="documents",
        points=points
    )
```

## Temporal Worker Architecture

### Workflow Design

```
UploadWorkflow (Parent)
    │
    ├── Activity: ValidateS3File
    │
    ├── Signal: VideoUploaded (1h timeout)
    │
    └── Child Workflow: IndexingWorkflow
            │
            ├── Activity: DownloadFromS3
            │
            ├── Activity: ParseDocument
            │
            ├── Activity: GenerateEmbeddings
            │
            ├── Activity: UpsertToQdrant
            │
            └── Activity: UpdatePostgresMetadata
```

### Activity Implementations

#### DownloadFromS3Activity
```python
@activity.defn
async def download_from_s3(document_id: str) -> bytes:
    """Download document from S3"""

    s3 = get_s3_client()
    document = await get_document_metadata(document_id)

    response = await s3.get_object(
        Bucket=S3_BUCKET,
        Key=document['s3_key']
    )

    return await response['Body'].read()
```

#### ParseDocumentActivity
```python
@activity.defn
async def parse_document(
    document_id: str,
    file_data: bytes,
    filename: str
) -> list[DocumentChunk]:
    """Parse document into chunks"""

    ext = Path(filename).suffix.lower()

    if ext == '.pdf':
        loader = PyPDFLoader(file_data)
    elif ext == '.docx':
        loader = DocxLoader(file_data)
    elif ext in ('.txt', '.md'):
        loader = TextLoader(file_data)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    documents = loader.load()

    # Split into chunks
    splitter = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)
```

#### GenerateEmbeddingsActivity
```python
@activity.defn
async def generate_embeddings(
    chunks: list[str]
) -> list[list[float]]:
    """Generate embeddings for chunks"""

    openai = get_openai_client()

    embeddings = []
    for chunk in chunks:
        response = await openai.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)

    return embeddings
```

### Error Handling

```python
# Retry policies
@activity.defn
async def upsert_to_qdrant(points: list[PointStruct]):
    """Upsert vectors to Qdrant with retry"""

    policy = RetryPolicy(
        maximum_attempts=3,
        initial_interval=timedelta(seconds=1),
        backoff_coefficient=2.0
    )

    with policy:
        await qdrant.upsert(collection_name="documents", points=points)
```

## Shared Code Design

### Modules in `shared/`

1. **database.py** - Async PostgreSQL connection pool
2. **qdrant.py** - Qdrant client wrapper
3. **s3.py** - S3 client wrapper
4. **models.py** - Shared data models

### Shared Models

```python
from dataclasses import dataclass
from enum import Enum

class DocumentStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    COMPLETE = "complete"
    FAILED = "failed"

@dataclass
class DocumentChunk:
    content: str
    embedding: list[float]
    chunk_index: int
    filename: str
    upload_timestamp: datetime

@dataclass
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    timestamp: datetime
    metadata: dict
```

## Performance Considerations

### Core Service

- **Connection Pooling**: Reuse database and Qdrant connections
- **Streaming**: SSE avoids buffering full responses
- **Async I/O**: Handle multiple concurrent requests
- **Caching**: Cache conversation history in memory (TBD)

### Worker Service

- **Batch Processing**: Process multiple documents in parallel (TBD)
- **Retry Logic**: Exponential backoff for resilience
- **Dead Letter Queue**: Failed workflows for manual review

### Resource Usage

- Core: ~512Mi RAM, ~500m CPU base
- Worker: ~256Mi RAM, ~200m CPU base
- Scales with document processing load

## Security Considerations

### API Security
- JWT validation via Gateway (service-to-service trust)
- No direct external access

### Data Security
- S3 bucket with encryption
- Qdrant collection with access controls
- PostgreSQL with user-level permissions

### Secrets Management
- Environment variables for API keys
- K8S Secrets in production

## Future Enhancements

### Core Service
1. Hybrid search (BM25 + semantic)
2. Re-ranking (Cross-Encoder)
3. Streaming response caching
4. Query analytics
5. A/B testing for prompts

### Worker Service
1. Batch document processing
2. Priority queues
3. Document versioning
4. Incremental updates
5. Failed document reprocessing
