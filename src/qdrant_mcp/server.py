import logging
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan
from mcp.types import ToolAnnotations
from pydantic import Field
from qdrant_client import AsyncQdrantClient

from .embeddings import create_embedder
from .settings import Settings
from .store import QdrantStore

logger = logging.getLogger(__name__)


def create_server(settings: Settings) -> FastMCP:
    @lifespan
    async def store_lifespan(server: FastMCP):
        # Runs once when the HTTP server starts: connect, then check the collection fits the model.
        embedder = create_embedder(settings)
        client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        try:
            store = QdrantStore(client, embedder, settings.qdrant_collection)
            await store.check_collection()
            logger.info("Using collection '%s' with model '%s'", settings.qdrant_collection, settings.embedding_model)
            yield {"store": store}
        finally:
            await embedder.close()
            await client.close()

    server = FastMCP(
        "qdrant-mcp",
        instructions=(
            f"Semantic search over the Qdrant collection '{settings.qdrant_collection}'. "
            "Use qdrant_find to look things up by meaning."
        ),
        lifespan=store_lifespan,
    )

    def get_store(ctx: Context) -> QdrantStore:
        return ctx.lifespan_context["store"]

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    async def qdrant_find(
        query: Annotated[str, Field(min_length=1, description="What to look for, in natural language.")],
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=50, description="Maximum number of results.")] = 5,
        metadata_filter: Annotated[
            dict[str, str | int | bool] | None,
            Field(
                description=(
                    "Only return entries whose metadata has these exact values, e.g. "
                    '{"project": "x", "year": 2024}. Values must be strings, integers or booleans.'
                )
            ),
        ] = None,
    ) -> list[dict[str, Any]]:
        """Search the collection by meaning. Returns the closest matches with their id, score, text and metadata."""
        return await get_store(ctx).search(query, limit, metadata_filter)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    async def qdrant_collection_info(ctx: Context) -> dict[str, Any]:
        """Show details of the collection this server searches: point count, embedding model, dimension and distance."""
        return await get_store(ctx).collection_info()

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    async def qdrant_list_collections(ctx: Context) -> list[str]:
        """List all collections in the Qdrant instance. This server only searches its own collection."""
        return await get_store(ctx).list_collections()

    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    create_server(settings).run(transport="http", host=settings.mcp_host, port=settings.mcp_port)


if __name__ == "__main__":
    main()
