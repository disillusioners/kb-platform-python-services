"""Qdrant client wrapper."""

from qdrant_client import QdrantClient as QdrantSDK
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List


class QdrantClient:
    """Qdrant vector database client."""

    def __init__(self, url: str, collection: str = "documents"):
        self.client = QdrantSDK(url=url)
        self.collection = collection

    async def ensure_collection(self, vector_size: int = 1536):
        """Ensure collection exists."""
        from qdrant_client.models import CreateCollection

        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def upsert_vectors(
        self,
        points: List[PointStruct],
    ):
        """Upsert vectors to collection."""
        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    async def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filter_dict: dict = None,
    ) -> List[dict]:
        """Search for similar vectors."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        search_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            search_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=search_filter,
        )

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    async def delete_document(self, document_id: str):
        """Delete all vectors for a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
