from typing import Any

from qdrant_client import AsyncQdrantClient, models

from .embeddings import Embedder


class CollectionError(RuntimeError):
    """The collection is missing or was built for a different embedding model."""


def vector_name_for(model_name: str) -> str:
    """Name the vector after the model, e.g. "BAAI/bge-m3" -> "bge-m3"."""
    return model_name.split("/")[-1].lower()


class QdrantStore:
    """Searches one Qdrant collection, embedding queries with the given embedder.

    The collection is filled by a separate pipeline, which must use the same model and store
    each point with a vector named `vector_name_for(model)` and the payload
    {"document": <text>, "metadata": {<key>: <value>, ...}}.
    """

    def __init__(self, client: AsyncQdrantClient, embedder: Embedder, collection: str):
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._vector_name = vector_name_for(embedder.model_name)

    async def check_collection(self) -> None:
        """Fail early if the collection is missing or does not fit the embedding model."""
        if not await self._client.collection_exists(self._collection):
            raise CollectionError(
                f"Collection '{self._collection}' does not exist. Load it with the ingestion pipeline first."
            )

        info = await self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        params = vectors.get(self._vector_name) if isinstance(vectors, dict) else None
        if params is None:
            names = list(vectors) if isinstance(vectors, dict) else ["<unnamed>"]
            raise CollectionError(
                f"Collection '{self._collection}' has vectors {names}, but model "
                f"'{self._embedder.model_name}' needs vector '{self._vector_name}'. "
                "Use the model the collection was built with."
            )

        dimension = len(await self._embedder.embed_query("dimension probe"))
        if params.size != dimension:
            raise CollectionError(
                f"Collection '{self._collection}' stores {params.size}-dimensional vectors, "
                f"but model '{self._embedder.model_name}' returns {dimension}."
            )

    async def search(
        self, query: str, limit: int = 5, metadata_filter: dict[str, str | int | bool] | None = None
    ) -> list[dict[str, Any]]:
        vector = await self._embedder.embed_query(query)
        query_filter = None
        if metadata_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
                    for key, value in metadata_filter.items()
                ]
            )
        response = await self._client.query_points(
            self._collection,
            query=vector,
            using=self._vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "id": str(point.id),
                "score": point.score,
                "document": point.payload.get("document"),
                "metadata": point.payload.get("metadata") or {},
            }
            for point in response.points
        ]

    async def collection_info(self) -> dict[str, Any]:
        info = await self._client.get_collection(self._collection)
        params = info.config.params.vectors[self._vector_name]
        return {
            "collection": self._collection,
            "status": str(info.status.value),
            "points_count": info.points_count,
            "embedding_model": self._embedder.model_name,
            "vector_name": self._vector_name,
            "dimension": params.size,
            "distance": params.distance.value,
        }

    async def list_collections(self) -> list[str]:
        response = await self._client.get_collections()
        return sorted(collection.name for collection in response.collections)
