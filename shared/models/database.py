"""Database connection and models."""

from typing import AsyncIterator
import asyncpg
from shared.config import get_settings
from shared.models.schemas import Document, Message, Conversation, DocumentStatus, MessageRole
from datetime import datetime


class Database:
    """Database connection pool."""

    def __init__(self):
        self.settings = get_settings()
        self.pool = None

    async def connect(self):
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(
            self.settings.database_url,
            min_size=2,
            max_size=10,
        )

    async def execute(self, query: str, *args):
        """Execute a query."""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args):
        """Fetch rows from query."""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        """Fetch single row from query."""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()


db = Database()


async def get_db() -> AsyncIterator[asyncpg.Connection]:
    """Get database connection for dependency injection."""
    if not db.pool:
        await db.connect()
    async with db.pool.acquire() as conn:
        yield conn


async def init_db():
    """Initialize database connection."""
    await db.connect()


async def get_document(conn: asyncpg.Connection, document_id: str) -> Document | None:
    """Get document by ID."""
    row = await conn.fetchrow(
        "SELECT * FROM documents WHERE id = $1",
        document_id
    )
    if not row:
        return None
    return Document(**dict(row))


async def create_document(
    conn: asyncpg.Connection,
    s3_key: str,
    filename: str,
    file_size: int,
    status: DocumentStatus = DocumentStatus.PENDING,
) -> Document:
    """Create a new document."""
    row = await conn.fetchrow(
        """
        INSERT INTO documents (s3_key, filename, file_size, status)
        VALUES ($1, $2, $3, $4)
        RETURNING *
        """,
        s3_key, filename, file_size, status.value
    )
    return Document(**dict(row))


async def update_document_status(
    conn: asyncpg.Connection,
    document_id: str,
    status: DocumentStatus,
    error_message: str | None = None,
    chunk_count: int | None = None,
):
    """Update document status."""
    await conn.execute(
        """
        UPDATE documents
        SET status = $1, error_message = $2,
            indexed_at = CASE WHEN $1 = 'complete' THEN NOW() ELSE indexed_at END,
            metadata = CASE WHEN $3 IS NOT NULL
                THEN jsonb_set(COALESCE(metadata, '{}'::jsonb), '{chunk_count}', to_jsonb($3::int))
                ELSE metadata
            END
        WHERE id = $4
        """,
        status.value, error_message, chunk_count, document_id
    )


async def get_conversation(
    conn: asyncpg.Connection,
    conversation_id: str,
) -> Conversation | None:
    """Get conversation by ID."""
    row = await conn.fetchrow(
        "SELECT * FROM conversations WHERE id = $1",
        conversation_id
    )
    if not row:
        return None
    return Conversation(**dict(row))


async def create_conversation(
    conn: asyncpg.Connection,
) -> Conversation:
    """Create a new conversation."""
    row = await conn.fetchrow(
        """
        INSERT INTO conversations DEFAULT VALUES RETURNING *
        """
    )
    return Conversation(**dict(row))


async def get_messages(
    conn: asyncpg.Connection,
    conversation_id: str,
) -> list[Message]:
    """Get messages for a conversation."""
    rows = await conn.fetch(
        """
        SELECT * FROM messages
        WHERE conversation_id = $1
        ORDER BY timestamp ASC
        """,
        conversation_id
    )
    return [Message(**dict(row)) for row in rows]


async def save_message(
    conn: asyncpg.Connection,
    conversation_id: str,
    role: MessageRole,
    content: str,
    metadata: dict | None = None,
) -> Message:
    """Save a message to conversation."""
    row = await conn.fetchrow(
        """
        INSERT INTO messages (conversation_id, role, content, metadata)
        VALUES ($1, $2, $3, $4)
        RETURNING *
        """,
        conversation_id, role.value, content, metadata or {}
    )
    return Message(**dict(row))
