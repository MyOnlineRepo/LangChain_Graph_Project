"""Lokale Embeddings ueber fastembed.

Anthropic bietet keinen Embeddings-Endpunkt an. Statt dafuer einen zweiten
API-Key zu besorgen, rechnet fastembed offline auf der CPU.

Der Adapter zeigt nebenbei, wie schmal das LangChain-Interface ist: zwei
Methoden, mehr verlangt `Embeddings` nicht.
"""

from langchain_core.embeddings import Embeddings

# Mehrsprachig, damit deutsche Fragen auch englische Texte finden.
# 384 Dimensionen, rund 220 MB - wird beim ersten Lauf heruntergeladen.
STANDARD_MODELL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedAdapter(Embeddings):
    """Bindet ein fastembed-Modell als LangChain-Embeddings ein."""

    def __init__(self, modell: str = STANDARD_MODELL) -> None:
        from fastembed import TextEmbedding

        self._modell = TextEmbedding(model_name=modell)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._modell.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._modell.embed([text]))).tolist()
