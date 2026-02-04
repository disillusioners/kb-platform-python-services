"""RAG query engine for KB platform."""

from typing import AsyncIterator, Optional, List
from openai import AsyncOpenAI
from shared.services.qdrant import QdrantClient


class RAGEngine:
    """Retrieval-Augmented Generation Engine."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        openai_api_key: str,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.qdrant = qdrant_client
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
        self.temperature = temperature
        self.embedding_model = embedding_model

    async def query(
        self,
        query_text: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
        history: List[dict] = None,
    ) -> AsyncIterator[str]:
        """Perform RAG query with streaming response."""
        
        # 1. Retrieve relevant chunks
        context_chunks = await self._retrieve_relevant_chunks(query_text, top_k)
        
        # 2. Stream response
        async for chunk in self._stream_response(query_text, context_chunks, history or []):
            yield chunk

    async def _retrieve_relevant_chunks(
        self, query: str, top_k: int
    ) -> List[str]:
        """Retrieve relevant chunks from Qdrant."""
        
        # Generate embedding
        response = await self.client.embeddings.create(
            input=query,
            model=self.embedding_model
        )
        query_vector = response.data[0].embedding
        
        # Search Qdrant
        results = await self.qdrant.search(
            query_vector=query_vector,
            limit=top_k
        )
        
        # Extract content
        chunks = []
        for res in results:
            if "content" in res["payload"]:
                chunks.append(res["payload"]["content"])
                
        return chunks

    async def _stream_response(
        self,
        query: str,
        context_chunks: List[str],
        history: List[dict],
    ) -> AsyncIterator[str]:
        """Stream LLM response."""
        prompt = self._build_rag_prompt(query, context_chunks, history)

        try:
            # We use the construct prompt as the user message, or system + user.
            # Simple approach: System prompt + User prompt.
            # But here `_build_rag_prompt` returns a single string.
            # So we send it as a user message.
            
            messages = [{"role": "user", "content": prompt}]
            
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
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
        
        # Format history
        history_str = ""
        if conversation_history:
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])

        return f"""You are a helpful assistant for the Knowledge Base platform.

Context from documents:
{context}

Conversation history:
{history_str}

User: {query}

Assistant:"""
