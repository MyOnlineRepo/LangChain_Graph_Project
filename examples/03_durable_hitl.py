"""Ebene 3: Persistenz (Checkpointer) + Human-in-the-Loop.

Der Checkpointer speichert den Graph-State nach jedem Superstep. Dadurch:
Gedaechtnis ueber Turns hinweg, Wiederaufnahme nach Absturz, Time Travel
und Unterbrechungen zur Freigabe durch einen Menschen.

    uv run python examples/03_durable_hitl.py
"""

from typing import Annotated

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from lcgraph import load_env

load_env()

DB_PATH = "checkpoints.sqlite"


class State(TypedDict):
    messages: Annotated[list, add_messages]
    entwurf: str
    freigegeben: bool


def entwurf_schreiben(state: State) -> dict:
    thema = state["messages"][-1].content
    return {"entwurf": f"[Entwurf zu: {thema}]"}


def freigabe(state: State) -> dict:
    """Haelt den Graph an und wartet auf eine menschliche Entscheidung."""
    antwort = interrupt({"frage": "Entwurf freigeben?", "entwurf": state["entwurf"]})
    return {"freigegeben": antwort == "ja"}


def veroeffentlichen(state: State) -> dict:
    text = state["entwurf"] if state["freigegeben"] else "(abgelehnt)"
    return {"messages": [{"role": "assistant", "content": f"Ergebnis: {text}"}]}


builder = StateGraph(State)
builder.add_node("entwurf", entwurf_schreiben)
builder.add_node("freigabe", freigabe)
builder.add_node("publish", veroeffentlichen)
builder.add_edge(START, "entwurf")
builder.add_edge("entwurf", "freigabe")
builder.add_edge("freigabe", "publish")
builder.add_edge("publish", END)


def main() -> None:
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "demo-1"}}

        # Lauf 1: stoppt beim interrupt in "freigabe"
        for chunk in graph.stream(
            {"messages": [{"role": "user", "content": "Blogpost ueber LangGraph"}]},
            config,
        ):
            print(chunk)

        # Zustand inspizieren - der Lauf liegt persistiert in der SQLite-DB
        snapshot = graph.get_state(config)
        print("wartet auf:", snapshot.next)

        # Lauf 2: Antwort des Menschen einspeisen, Graph laeuft weiter
        for chunk in graph.stream(Command(resume="ja"), config):
            print(chunk)


if __name__ == "__main__":
    main()
