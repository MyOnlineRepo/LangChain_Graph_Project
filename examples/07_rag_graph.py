"""Ebene 6b: Selbstkorrigierendes RAG - Retrieval als Zyklus.

Eine RAG-Pipeline (suchen -> antworten) ist eine Gerade und braucht keinen
Graphen. Interessant wird es erst, wenn der Graph das Ergebnis *bewertet* und
bei schlechtem Ausgang einen zweiten Anlauf nimmt:

              +----------------------------+
              v                            |
    START -> suchen -> treffer_pruefen -> umformulieren      (Zyklus 1)
                            |
                     genug Treffer?
                            v
                        antworten <------+
                            |            |
                            v            |
                      antwort_pruefen ---+                   (Zyklus 2)
                            |
                        belegt?
                            v
                           END

Zyklus 1 repariert schlechtes Retrieval, Zyklus 2 faengt Antworten ab, die
nicht durch die Quellen gedeckt sind. `versuche` bremst beide.

Voraussetzung: einmal `uv run python examples/06_rag_ingest.py` laufen lassen.

    uv run python examples/07_rag_graph.py
"""

import operator
from pathlib import Path
from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from lcgraph import MODEL, MODEL_SCHNELL, load_env, nur_text
from lcgraph.embeddings import FastEmbedAdapter

load_env()

WURZEL = Path(__file__).resolve().parent.parent
INDEX_PFAD = WURZEL / ".chroma"
SAMMLUNG = "projekt-docs"

MAX_VERSUCHE = 2  # Bremse fuer beide Zyklen
TREFFER = 4


class State(TypedDict):
    frage: str
    suchanfrage: str
    treffer: list[Document]
    antwort: str
    versuche: int
    # Reducer: jeder Node haengt an, statt zu ueberschreiben.
    protokoll: Annotated[list[str], operator.add]


class Relevanz(BaseModel):
    brauchbar: bool = Field(description="Beantworten die Texte die Frage?")
    begruendung: str = Field(description="Ein knapper Satz.")


class Beleg(BaseModel):
    gedeckt: bool = Field(description="Steht alles in der Antwort so in den Quellen?")
    begruendung: str = Field(description="Ein knapper Satz.")


pruefer = init_chat_model(MODEL_SCHNELL)
schreiber = init_chat_model(MODEL)

_store = Chroma(
    collection_name=SAMMLUNG,
    embedding_function=FastEmbedAdapter(),
    persist_directory=str(INDEX_PFAD),
)


def _quellen(treffer: list[Document]) -> str:
    return "\n\n---\n\n".join(f"[{d.metadata['quelle']}]\n{d.page_content}" for d in treffer)


def suchen(state: State) -> dict:
    treffer = _store.similarity_search(state["suchanfrage"], k=TREFFER)
    namen = sorted({d.metadata["quelle"] for d in treffer})
    return {"treffer": treffer, "protokoll": [f"suchen: '{state['suchanfrage']}' -> {namen}"]}


def treffer_pruefen(state: State) -> dict:
    urteil = pruefer.with_structured_output(Relevanz).invoke(
        f"Frage: {state['frage']}\n\n"
        f"Gefundene Texte:\n{_quellen(state['treffer'])}\n\n"
        "Reichen diese Texte, um die Frage zu beantworten?"
    )
    return {
        "protokoll": [f"treffer_pruefen: brauchbar={urteil.brauchbar} ({urteil.begruendung})"],
        "antwort": "" if urteil.brauchbar else "UNBRAUCHBAR",
    }


def route_treffer(state: State) -> Literal["antworten", "umformulieren"]:
    if state["antwort"] != "UNBRAUCHBAR":
        return "antworten"
    # Bremse: lieber mit schwachen Treffern antworten als endlos suchen.
    return "umformulieren" if state["versuche"] < MAX_VERSUCHE else "antworten"


def umformulieren(state: State) -> dict:
    neu = pruefer.invoke(
        f"Die Suche nach '{state['suchanfrage']}' lieferte nichts Brauchbares.\n"
        f"Urspruengliche Frage: {state['frage']}\n"
        "Formuliere eine bessere Suchanfrage. Antworte NUR mit der Suchanfrage."
    )
    text = nur_text(neu.content)
    return {
        "suchanfrage": text.strip().strip('"'),
        "versuche": state["versuche"] + 1,
        "protokoll": [f"umformulieren: -> '{text.strip()[:60]}'"],
    }


def antworten(state: State) -> dict:
    antwort = schreiber.invoke(
        "Beantworte die Frage ausschliesslich aus den Quellen. Nenne die "
        "Dateinamen, auf die du dich stuetzt. Fehlt die Information, sage das.\n\n"
        f"Frage: {state['frage']}\n\nQuellen:\n{_quellen(state['treffer'])}"
    )
    return {
        "antwort": nur_text(antwort.content).strip(),
        "protokoll": ["antworten: Entwurf erstellt"],
    }


def antwort_pruefen(state: State) -> dict:
    urteil = pruefer.with_structured_output(Beleg).invoke(
        f"Quellen:\n{_quellen(state['treffer'])}\n\n"
        f"Antwort:\n{state['antwort']}\n\n"
        "Ist jede Aussage der Antwort durch die Quellen gedeckt?"
    )
    return {
        "protokoll": [f"antwort_pruefen: gedeckt={urteil.gedeckt} ({urteil.begruendung})"],
        "versuche": state["versuche"] + (0 if urteil.gedeckt else 1),
        "suchanfrage": state["suchanfrage"] if urteil.gedeckt else "NEU_SCHREIBEN",
    }


def route_antwort(state: State) -> Literal["antworten", "__end__"]:
    if state["suchanfrage"] != "NEU_SCHREIBEN":
        return END
    return "antworten" if state["versuche"] <= MAX_VERSUCHE else END


builder = StateGraph(State)
builder.add_node("suchen", suchen)
builder.add_node("treffer_pruefen", treffer_pruefen)
builder.add_node("umformulieren", umformulieren)
builder.add_node("antworten", antworten)
builder.add_node("antwort_pruefen", antwort_pruefen)

builder.add_edge(START, "suchen")
builder.add_edge("suchen", "treffer_pruefen")
builder.add_conditional_edges("treffer_pruefen", route_treffer)
builder.add_edge("umformulieren", "suchen")  # Zyklus 1
builder.add_edge("antworten", "antwort_pruefen")
builder.add_conditional_edges("antwort_pruefen", route_antwort)  # Zyklus 2

graph = builder.compile()


def frage_stellen(frage: str) -> None:
    print(f"\n{'=' * 70}\nFrage: {frage}\n{'=' * 70}")
    zustand = graph.invoke(
        {
            "frage": frage,
            "suchanfrage": frage,
            "treffer": [],
            "antwort": "",
            "versuche": 0,
            "protokoll": [],
        }
    )
    for zeile in zustand["protokoll"]:
        print(f"  . {zeile}")
    print(f"\n--- Antwort ({zustand['versuche']} Korrekturversuche) ---")
    print(zustand["antwort"])


def main() -> None:
    # Frage 1: steht so im Korpus - sollte ohne Korrektur durchgehen.
    frage_stellen("Welcher Paketmanager wird in diesem Projekt verwendet und warum nicht pip?")
    # Frage 2: nicht im Korpus - hier muss der Graph sich korrigieren oder passen.
    frage_stellen("Wie hoch ist die Schrittweite beim Adam-Optimizer in diesem Projekt?")


if __name__ == "__main__":
    main()
