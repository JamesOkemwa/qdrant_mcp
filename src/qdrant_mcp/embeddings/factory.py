from ..settings import Settings
from .base import Embedder
from .openai_compat import OpenAICompatibleEmbedder


def create_embedder(settings: Settings) -> Embedder:
    """Build the embedder described by the settings.

    To add a provider, write a class that satisfies `Embedder` and choose it here
    (for example from a new `EMBEDDING_PROVIDER` setting).
    """
    return OpenAICompatibleEmbedder(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model_name=settings.embedding_model,
    )
