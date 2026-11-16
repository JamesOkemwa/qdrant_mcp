from typing import Protocol


class Embedder(Protocol):
    """What the server needs from an embedding provider.

    Any class with these members can be used; the store never knows which
    provider or model is behind it.
    """

    model_name: str

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...

    async def close(self) -> None: ...
