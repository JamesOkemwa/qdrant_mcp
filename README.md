# qdrant-mcp

A read-only MCP server for semantic search over one Qdrant collection. Queries are embedded with
any OpenAI-compatible endpoint (OpenAI, vLLM, ...), so the model is a configuration choice.

The collection is created and filled by a separate ingestion pipeline; this server only searches it.

## Run

```bash
docker compose up -d     # Qdrant at http://localhost:6333 (skip if you already run Qdrant)
cp .env.sample .env      # then edit
uv run qdrant-mcp        # serves streamable HTTP at http://127.0.0.1:8765/mcp
```

Settings are read from environment variables and from a `.env` file in the **current working
directory**, so start the server from the repository root (or set the variables in the environment).

On startup the server checks that the collection exists and was built for the configured model
(matching vector name and dimension). If not, it refuses to start.

## Tools

| Tool | Purpose |
|---|---|
| `qdrant_find` | Search by meaning, optionally filtered by exact metadata values |
| `qdrant_collection_info` | Point count, model, dimension and distance of the collection |
| `qdrant_list_collections` | All collections in the Qdrant instance |

## Collection format (contract for the ingestion pipeline)

- **Model:** the same embedding model as `EMBEDDING_MODEL`.
- **Vector:** a named vector called after the model: the part after the last `/`, lowercased
  (`BAAI/bge-m3` -> `bge-m3`, `text-embedding-3-large` -> `text-embedding-3-large`). Use
  `qdrant_mcp.store.vector_name_for(model)`.
- **Payload:** `{"document": "<the text>", "metadata": {"<key>": <value>, ...}}`. `metadata` is
  optional; its keys are what `qdrant_find`'s `metadata_filter` matches against. Filter
  values must be strings, integers or booleans.
- **Distance:** any; the server reads it from the collection.

## Layout

```
src/qdrant_mcp/
  settings.py              # environment configuration
  server.py                # MCP tools and startup
  store.py                 # all Qdrant access
  embeddings/
    base.py                # Embedder protocol
    factory.py             # picks the provider
    openai_compat.py       # OpenAI-compatible provider
```

To add a provider, write a class matching `Embedder` in `embeddings/` and return it from
`create_embedder`.
