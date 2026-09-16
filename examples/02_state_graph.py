"""Ebene 2: Eigener Graph mit der LangGraph StateGraph-API.

Zeigt die vier Bausteine: State (typisiert + Reducer), Nodes, Edges und
eine bedingte Kante (Routing).

    uv run python examples/02_state_graph.py
"""

from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from lcgraph import MODEL, load_env

load_env()


class State(TypedDict):
    # add_messages ist ein Reducer: Node-Rueckgaben werden angehaengt,
    # nicht ueberschrieben.
    messages: Annotated[list, add_messages]
    kategorie: str


llm = init_chat_model(MODEL)


def classify(state: State) -> dict:
    """Node 1: klassifiziert die Nutzeranfrage."""
    frage = state["messages"][-1].content
    antwort = llm.invoke(
        f"Klassifiziere die folgende Anfrage mit genau einem Wort "
        f"- 'technisch' oder 'allgemein':\n\n{frage}"
    )
    kategorie = "technisch" if "technisch" in antwort.content.lower() else "allgemein"
    return {"kategorie": kategorie}


def route(state: State) -> Literal["experte", "smalltalk"]:
    """Bedingte Kante: entscheidet ueber den naechsten Node."""
    return "experte" if state["kategorie"] == "technisch" else "smalltalk"


def experte(state: State) -> dict:
    antwort = llm.invoke(
        [{"role": "system", "content": "Antworte als Senior-Engineer, praezise und mit Code."}]
        + state["messages"]
    )
    return {"messages": [antwort]}


def smalltalk(state: State) -> dict:
    antwort = llm.invoke(
        [{"role": "system", "content": "Antworte locker und in maximal zwei Saetzen."}]
        + state["messages"]
    )
    return {"messages": [antwort]}


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("experte", experte)
builder.add_node("smalltalk", smalltalk)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route)
builder.add_edge("experte", END)
builder.add_edge("smalltalk", END)

graph = builder.compile()


def main() -> None:
    frage = "Wie funktioniert ein Reducer in LangGraph?"
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": frage}]},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            print(f"--- {node} ---")
            print(update)


if __name__ == "__main__":
    main()
