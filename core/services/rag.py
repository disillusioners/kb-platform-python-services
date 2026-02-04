"""RAG query engine for KB platform."""

from typing import AsyncIterator, Optional
import openai
from .qdrant import QdrantClient


class RAGEngine:
    """Retrieval-Augmented Generation Engine."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_api_key: str,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
    ):
        self.qdrant = qdrant_client
        openai.api_key = openai_api_key
        self.llm_model = llm_model
        self.temperature = temperature

    async def query(
        self,
        query_text: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
    ) -> AsyncIterator[str]:
        """Perform RAG query with streaming response."""
        async for chunk in self._stream_response(query_text, top_k):
            yield chunk

    async def _stream_response(
        self,
        query: str,
        top_k: int,
    ) -> AsyncIterator[str]:
        """Stream LLM response."""
        prompt = self._build_rag_prompt(query, [], [])

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=True,
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.get("content"):
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise Exception(f"LLM generation failed: {e}")

    def _build_rag_prompt(
        self,
        query: str,
        context_chunks: list[str],
        conversation_history: list[dict],
    ) -> str:
        """Build RAG prompt."""
        context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."
        history = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])

        return f"""You are a helpful assistant for the Knowledge Base platform.

Context from documents:
{context}

Conversation history:
{history}

User: {query}

Assistant:"""
