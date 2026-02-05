"""Data models for KB platform."""

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import uuid4, UUID


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    INDEXING = "indexing"
    COMPLETE = "complete"
    FAILED = "failed"


class MessageRole(str, Enum):
    """Message role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"


class DocumentMetadata(BaseModel):
    """Document metadata."""
    chunk_count: Optional[int] = None


class Document(BaseModel):
    """Document model."""
    id: UUID = Field(default_factory=uuid4)
    s3_key: str
    filename: str
    file_size: int
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentListResponse(BaseModel):
    """Document list response."""
    documents: list[Document]
    total: int
    limit: int = 50
    offset: int = 0


class Message(BaseModel):
    """Message in conversation."""
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class MessageListResponse(BaseModel):
    """Message list response."""
    messages: list[Message]


class DocumentChunk(BaseModel):
    """Document chunk model."""
    content: str
    embedding: list[float]
    chunk_index: int
    filename: str
    upload_timestamp: datetime


class Conversation(BaseModel):
    """Conversation model."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: Optional[int] = None


class ConversationListResponse(BaseModel):
    """Conversation list response."""
    conversations: list[Conversation]
    total: int
    limit: int = 50
    offset: int = 0


class QueryRequest(BaseModel):
    """Query request."""
    query: str
    conversation_id: Optional[UUID] = None
    top_k: int = 5


class SSEEvent(BaseModel):
    """SSE event."""
    type: str
    id: Optional[str] = None
    content: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""
    error: ErrorDetail


class ErrorDetail(BaseModel):
    """Error detail."""
    code: str
    message: str
    details: Optional[Dict[str, str]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    status: str
    dependencies: Dict[str, str]
