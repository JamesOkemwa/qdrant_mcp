from openai import AsyncOpenAI


class OpenAICompatibleEmbedder:
    """Embeds text through any server that implements the OpenAI /v1/embeddings API.

    Works with OpenAI itself and with self-hosted servers such as vLLM.
    """

    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.model_name = model_name
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def embed_query(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(model=self.model_name, input=text)
        return response.data[0].embedding

    async def close(self) -> None:
        await self._client.close()
