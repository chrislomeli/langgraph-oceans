from __future__ import annotations

from openai import OpenAI

_MODEL = "text-embedding-3-large"
_DIMENSIONS = 1536


class Embedder:
    """Singleton-friendly wrapper around OpenAI text embeddings.

    Share one instance across the app rather than creating per-call.
    """

    def __init__(self, api_key: str,  model: str = _MODEL, dimensions: int = _DIMENSIONS):
        self._client = OpenAI(api_key=api_key)
        self.dimensions = dimensions
        self.model = model

    @property
    def version(self) -> str:
        """Stable id for the embedding space → doc_chunks.embedder_ver (migration hook).

        Includes dimensions so the same model at a different size is distinguishable.
        """
        return f"{self.model}-{self.dimensions}-v1"

    def embed(self, text: str) -> list[float]:
        """Embed a single string; returns a list of floats of length dimensions."""
        response = self._client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings; returns one float list per text."""
        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]