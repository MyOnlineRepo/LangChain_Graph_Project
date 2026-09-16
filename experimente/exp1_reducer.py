"""Experiment 1: Was macht der Reducer eigentlich?

Zwei identische Graphen - der einzige Unterschied ist das Annotated.

    uv run python experimente/exp1_reducer.py
"""

from typing import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def baue_graph(mit_reducer: bool):
    if mit_reducer:

        class State(TypedDict):
            messages: Annotated[list, add_messages]  # anhaengen
    else:

        class State(TypedDict):
            messages: list  # ueberschreiben

    def node_a(state):
        return {"messages": [{"role": "assistant", "content": "A war hier"}]}

    def node_b(state):
        return {"messages": [{"role": "assistant", "content": "B war hier"}]}

    b = StateGraph(State)
    b.add_node("a", node_a)
    b.add_node("b", node_b)
    b.add_edge(START, "a")
    b.add_edge("a", "b")
    b.add_edge("b", END)
    return b.compile()


def inhalt(m):
    return m.content if hasattr(m, "content") else m["content"]


for mit in (True, False):
    graph = baue_graph(mit)
    ergebnis = graph.invoke({"messages": [{"role": "user", "content": "Frage des Nutzers"}]})
    msgs = ergebnis["messages"]
    label = "MIT add_messages" if mit else "OHNE Reducer   "
    print(f"{label} -> {len(msgs)} Nachricht(en): {[inhalt(m) for m in msgs]}")
