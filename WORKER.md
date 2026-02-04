# Temporal Worker - Worker Specification

## Overview

The Temporal Worker service manages asynchronous document ingestion and indexing workflows using the Temporal platform.

## Technology Stack

- **Python 3.11+**
- **Temporal SDK** - Workflow orchestration
- **boto3** - S3 client
- **LlamaIndex** - Document loaders and embeddings
- **qdrant-client** - Vector database client

## Architecture

```
┌────────────────────────────────────────────────────┐
│          Temporal Worker Service                     │
│                                                        │
│  ┌──────────────┐     ┌──────────────┐             │
│  │   Workflows  │◄────┤   Activities  │             │
│  │              │     │              │             │
│  │ Upload       │     │ S3 Download  │             │
│  │ Indexing     │     │ Parse Doc    │             │
│  │ Retry Logic  │     │ Embeddings   │             │
│  │              │     │ Qdrant Upsert│             │
│  └──────────────┘     └──────────────┘             │
│           │                    │                      │
│           │                    │                      │
│           ▼                    ▼                      │
│  ┌──────────────┐     ┌──────────────┐             │
│  │ Temporal     │     │  External    │             │
│  │ Server       │     │  Services    │             │
│  └──────────────┘     └──────────────┘             │
└────────────────────────────────────────────────────┘
```

## Workflows

### UploadWorkflow

Parent workflow that manages the document upload and indexing process.

#### Workflow Definition

```python
@workflow.defn
class UploadWorkflow:
    """Manages document upload and triggers indexing"""

    @workflow.run
    async def run(self, document_id: str) -> DocumentStatus:
        """Wait for upload signal and trigger indexing"""

        # Wait for upload signal (1h timeout)
        await workflow.wait_for_condition(
            lambda: self._has_upload_signal(),
            timeout=timedelta(hours=1)
        )

        # Validate S3 file exists
        if not await workflow.execute_activity(
            validate_s3_file,
            document_id,
            start_to_close_timeout=timedelta(minutes=5)
        ):
            # Cleanup if file doesn't exist
            await workflow.execute_activity(
                cleanup_document,
                document_id,
                start_to_close_timeout=timedelta(minutes=5)
            )
            return DocumentStatus.FAILED

        # Start child indexing workflow
        status = await workflow.execute_child_workflow(
            IndexingWorkflow,
            document_id,
            task_queue="indexing-queue"
        )

        return status

    @workflow.signal
    def upload_complete(self, file_size: int):
        """Signal that upload is complete"""
        self.file_size = file_size
        self.upload_signal_received = True

    def _has_upload_signal(self) -> bool:
        return self.upload_signal_received
```

#### Signal

**Signal Name**: `upload_complete`

**Payload**:
```json
{
  "file_size": 1048576
}
```

#### Timeout
- **Signal Wait**: 1 hour (configurable)

### IndexingWorkflow

Child workflow that performs the actual document indexing.

#### Workflow Definition

```python
@workflow.defn
class IndexingWorkflow:
    """Indexes a document in Qdrant"""

    @workflow.run
    async def run(self, document_id: str) -> DocumentStatus:
        """Download, parse, embed, and index document"""

        # 1. Download from S3
        file_data = await workflow.execute_activity(
            download_from_s3,
            document_id,
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0
            ),
            start_to_close_timeout=timedelta(minutes=10)
        )

        # 2. Parse document
        try:
            chunks = await workflow.execute_activity(
                parse_document,
                file_data,
                document_id,
                retry_policy=RetryPolicy(maximum_attempts=1),  # No retry
                start_to_close_timeout=timedelta(minutes=5)
            )
        except Exception as e:
            # Update status to failed
            await workflow.execute_activity(
                update_document_status,
                document_id,
                DocumentStatus.FAILED,
                str(e),
                start_to_close_timeout=timedelta(minutes=5)
            )
            return DocumentStatus.FAILED

        # 3. Generate embeddings
        embeddings = await workflow.execute_activity(
            generate_embeddings,
            [chunk.content for chunk in chunks],
            retry_policy=RetryPolicy(
                maximum_attempts=5,
                initial_interval=timedelta(seconds=2),
                backoff_coefficient=2.0
            ),
            start_to_close_timeout=timedelta(minutes=10)
        )

        # 4. Upsert to Qdrant
        await workflow.execute_activity(
            upsert_to_qdrant,
            document_id,
            chunks,
            embeddings,
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0
            ),
            start_to_close_timeout=timedelta(minutes=5)
        )

        # 5. Update Postgres metadata
        await workflow.execute_activity(
            update_document_metadata,
            document_id,
            DocumentStatus.COMPLETE,
            len(chunks),
            start_to_close_timeout=timedelta(minutes=5)
        )

        return DocumentStatus.COMPLETE
```

## Activities

### DownloadFromS3Activity

Downloads a document from S3.

**Signature**:
```python
@activity.defn
async def download_from_s3(document_id: str) -> bytes
```

**Retry Policy**:
- Maximum attempts: 3
- Initial interval: 1s
- Backoff coefficient: 2.0

**Errors**:
- `S3DownloadError`: Failed to download file
- `DocumentNotFoundError`: Document metadata not found in Postgres

### ValidateS3FileActivity

Validates that the file exists in S3 after upload.

**Signature**:
```python
@activity.defn
async def validate_s3_file(document_id: str) -> bool
```

**Retry Policy**: None (run once)

**Returns**: `True` if file exists, `False` otherwise

### ParseDocumentActivity

Parses a document into text chunks.

**Signature**:
```python
@activity.defn
async def parse_document(
    file_data: bytes,
    document_id: str
) -> list[DocumentChunk]
```

**Retry Policy**: None (no retry on parse errors - invalid format)

**Supported Formats**:
- PDF (using `pypdf`)
- DOCX (using `python-docx`)
- TXT (plain text)
- MD (Markdown)
- HTML (using `BeautifulSoup`)

**Configuration**:
- Chunk size: 512 tokens
- Chunk overlap: 50 tokens

**Errors**:
- `UnsupportedFileTypeError`: File type not supported
- `ParseError`: Failed to parse document

### GenerateEmbeddingsActivity

Generates vector embeddings for document chunks.

**Signature**:
```python
@activity.defn
async def generate_embeddings(
    chunks: list[str]
) -> list[list[float]]
```

**Retry Policy**:
- Maximum attempts: 5
- Initial interval: 2s
- Backoff coefficient: 2.0

**Model**: OpenAI `text-embedding-3-small`
- Dimension: 1536
- Max tokens: 8191

**Errors**:
- `EmbeddingAPIError`: OpenAI API error
- `RateLimitError`: OpenAI rate limit exceeded

### UpsertToQdrantActivity

Upserts document vectors to Qdrant.

**Signature**:
```python
@activity.defn
async def upsert_to_qdrant(
    document_id: str,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]]
)
```

**Retry Policy**:
- Maximum attempts: 3
- Initial interval: 1s
- Backoff coefficient: 2.0

**Collection**: `documents`

**Point Structure**:
```python
PointStruct(
    id=str(uuid.uuid4()),
    vector=embedding,
    payload={
        "document_id": document_id,
        "content": chunk.content,
        "chunk_index": chunk.index,
        "filename": chunk.filename,
        "upload_timestamp": chunk.upload_timestamp
    }
)
```

**Errors**:
- `QdrantError`: Qdrant operation failed

### UpdateDocumentMetadataActivity

Updates document metadata in Postgres.

**Signature**:
```python
@activity.defn
async def update_document_metadata(
    document_id: str,
    status: DocumentStatus,
    chunk_count: int,
    error_message: str | None = None
)
```

**Retry Policy**: None (run once)

**Database Table**: `documents`

**Fields Updated**:
- `status`
- `indexed_at` (if complete)
- `error_message` (if failed)
- `metadata` (chunk_count)

### CleanupDocumentActivity

Cleans up a failed document (deletes from S3 and updates Postgres).

**Signature**:
```python
@activity.defn
async def cleanup_document(document_id: str)
```

**Retry Policy**: None (run once)

**Actions**:
1. Delete file from S3
2. Update document status to `failed`
3. Log error message

## Configuration

### Environment Variables

```bash
# Temporal
TEMPORAL_HOST=temporal
TEMPORAL_PORT=7233
TEMPORAL_NAMESPACE=default

# Task Queues
UPLOAD_QUEUE=upload-queue
INDEXING_QUEUE=indexing-queue

# Database
DATABASE_URL=postgresql://user:pass@postgres:5432/kb

# Qdrant
QDRANT_URL=http://qdrant:6333

# S3
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=kb-documents

# OpenAI
OPENAI_API_KEY=sk-...

# Worker Settings
MAX_CONCURRENT_WORKFLOWS=100
MAX_CONCURRENT_ACTIVITIES=100
```

## Deployment

### Running Locally

```bash
# Install dependencies
poetry install

# Start Temporal server locally
temporal server start-dev

# Start worker
cd workers
python worker.py
```

### Running in Docker

```bash
# Build image
docker build -f Dockerfile.worker -t kb-platform-temporal-worker .

# Run container
docker run \
  -e TEMPORAL_HOST=temporal \
  -e DATABASE_URL=postgresql://... \
  -e OPENAI_API_KEY=sk-... \
  kb-platform-temporal-worker
```

### Kubernetes

See `kb-platform-infra/k8s/services/temporal-worker.yaml` for deployment manifest.

### Worker Configuration

```python
from temporalio import worker

worker = worker.Worker(
    client=client,
    task_queue="upload-queue",
    workflows=[UploadWorkflow],
    activities=[
        download_from_s3,
        validate_s3_file,
        parse_document,
        generate_embeddings,
        upsert_to_qdrant,
        update_document_metadata,
        cleanup_document
    ],
    max_concurrent_workflow_tasks=100,
    max_concurrent_activities=100
)

await worker.run()
```

## Error Handling

### Error Categories

1. **Transient Errors** - Retry with backoff
   - S3 download failures
   - OpenAI API rate limits
   - Qdrant connection errors

2. **Permanent Errors** - No retry, mark as failed
   - Invalid file format
   - Document not found
   - Missing API keys

### Dead Letter Queue

Failed workflows are automatically sent to Temporal's Dead Letter Queue for manual review.

### Monitoring

Workers emit structured logs:
```json
{
  "timestamp": "2026-02-04T11:30:00Z",
  "level": "INFO",
  "workflow_id": "550e8400-e29b-41d4-a716-446655440000",
  "activity": "parse_document",
  "status": "completed",
  "duration_ms": 1234
}
```

## Performance Considerations

### Throughput

- Concurrent workflows: 100 (configurable)
- Concurrent activities: 100 (configurable)
- Documents per minute: Depends on OpenAI rate limits

### Resource Usage

- Base: ~256Mi RAM, ~200m CPU
- Scales with concurrent workflows

### Optimization

1. Batch embedding generation (multiple chunks in one API call)
2. Connection pooling for S3 and Qdrant
3. Async activity execution where possible

## Security

### Credentials

- S3 credentials from environment variables
- OpenAI API key from environment variables
- Database credentials from environment variables

### Access Control

- Worker assumes all requests are authorized (trusted from Gateway)
- Temporal namespace isolation

## Future Enhancements

1. Batch document processing workflows
2. Priority queues for urgent documents
3. Document re-indexing workflows
4. Document versioning support
5. Failed document auto-retry
6. Progress reporting during indexing
7. Cancellation support for long-running workflows
