"""Experiment 3: stream_mode "updates" vs "values" - derselbe Graph.

uv run python experimente/exp3_stream.py
"""

import operator
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    log: Annotated[list, operator.add]
    zaehler: int
    status: str


def laden(state):
    return {"log": ["geladen"], "zaehler": 1, "status": "geladen"}


def pruefen(state):
    return {"log": ["geprueft"], "zaehler": state["zaehler"] + 1, "status": "geprueft"}


def senden(state):
    return {"log": ["gesendet"], "zaehler": state["zaehler"] + 1, "status": "fertig"}


b = StateGraph(State)
for name, fn in (("laden", laden), ("pruefen", pruefen), ("senden", senden)):
    b.add_node(name, fn)
b.add_edge(START, "laden")
b.add_edge("laden", "pruefen")
b.add_edge("pruefen", "senden")
b.add_edge("senden", END)
graph = b.compile()

start = {"log": [], "zaehler": 0, "status": "neu"}

print('=== stream_mode="updates"  -> WAS hat sich geaendert ===')
for chunk in graph.stream(dict(start), stream_mode="updates"):
    for node, delta in chunk.items():
        print(f"  {node:8} {delta}")

print()
print('=== stream_mode="values"   -> WIE sieht der State jetzt aus ===')
for i, snapshot in enumerate(graph.stream(dict(start), stream_mode="values")):
    print(f"  [{i}] {snapshot}")
