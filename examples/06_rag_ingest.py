"""Ebene 6a: Korpus indizieren - die Vorbereitung fuer den RAG-Graphen.

Bewusst getrennt vom Graphen: Indizieren passiert einmal, der Graph laeuft
oft. Und es haelt 07_rag_graph.py kurz.

Drei Entscheidungen, die hier festgelegt sind:

1. Korpus = das Markdown, das ohnehin im Repo liegt (Projektdokus und die
   25 Skill-Dateien unter .claude/skills). Kein Crawler, kein Download,
   reproduzierbar - und die Antwortqualitaet ist sofort pruefbar.
2. Embeddings = fastembed, lokal auf der CPU. Anthropic bietet keinen
   Embeddings-Endpunkt an, und so bleibt es bei einem einzigen API-Key.
   Mehrsprachiges Modell, damit deutsche Fragen auf englische Texte treffen.
3. Vector Store = Chroma, lokal in .chroma/ persistiert.

Ausdrueckliches Nicht-Ziel: Chunk-Tuning. Eine Konfiguration, fertig. Die
Zeit gehoert dem Graphen, nicht der Chunk-Groesse.

    uv run python examples/06_rag_ingest.py
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lcgraph.embeddings import FastEmbedAdapter

WURZEL = Path(__file__).resolve().parent.parent
INDEX_PFAD = WURZEL / ".chroma"
SAMMLUNG = "projekt-docs"

# Woher der Korpus kommt.
QUELLEN = ["*.md", ".claude/skills/*/SKILL.md"]


def dokumente_sammeln() -> list[Document]:
    """Liest alle Markdown-Dateien des Korpus ein."""
    docs: list[Document] = []
    for muster in QUELLEN:
        for pfad in sorted(WURZEL.glob(muster)):
            text = pfad.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"quelle": str(pfad.relative_to(WURZEL)).replace("\\", "/")},
                )
            )
    return docs


def main() -> None:
    docs = dokumente_sammeln()
    print(f"{len(docs)} Dateien eingelesen.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"{len(chunks)} Chunks erzeugt.")

    print("Embedding-Modell laden (beim ersten Mal ~220 MB Download) ...")
    embeddings = FastEmbedAdapter()

    print(f"Index schreiben nach {INDEX_PFAD} ...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=SAMMLUNG,
        persist_directory=str(INDEX_PFAD),
    )
    print("Fertig. Jetzt: uv run python examples/07_rag_graph.py")


if __name__ == "__main__":
    main()
