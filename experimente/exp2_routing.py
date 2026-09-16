"""Experiment 2: Kante ins Leere - Compile-Zeit oder Laufzeit?

Fall A: Routing-Funktion gibt einen Wert zurueck, der nicht im Literal steht.
Fall B: Das Literal nennt einen Node, den es gar nicht gibt.

    uv run python experimente/exp2_routing.py
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    wert: str


def start_node(state):
    return {"wert": "egal"}


def echter_node(state):
    return {"wert": "angekommen"}


def baue(route_fn):
    b = StateGraph(State)
    b.add_node("start", start_node)
    b.add_node("echter_node", echter_node)
    b.add_edge(START, "start")
    b.add_conditional_edges("start", route_fn)
    b.add_edge("echter_node", END)
    return b


# --- Fall A: Rueckgabe passt nicht zum Literal --------------------------
def route_a(state) -> Literal["echter_node"]:
    return "tippfehler_node"  # nicht im Literal, Node existiert nicht


print("Fall A: Rueckgabewert steht nicht im Literal")
try:
    graph = baue(route_a).compile()
    print("   compile()  -> OK, kein Fehler")
except Exception as e:
    print(f"   compile()  -> {type(e).__name__}: {e}")
else:
    try:
        graph.invoke({"wert": ""})
        print("   invoke()   -> OK")
    except Exception as e:
        print(f"   invoke()   -> {type(e).__name__}: {str(e)[:160]}")

print()


# --- Fall B: Literal nennt einen nicht existierenden Node ---------------
def route_b(state) -> Literal["gibt_es_nicht"]:
    return "gibt_es_nicht"


print("Fall B: Literal nennt einen Node, den es nicht gibt")
try:
    graph = baue(route_b).compile()
    print("   compile()  -> OK, kein Fehler")
except Exception as e:
    print(f"   compile()  -> {type(e).__name__}: {str(e)[:160]}")
else:
    try:
        graph.invoke({"wert": ""})
        print("   invoke()   -> OK")
    except Exception as e:
        print(f"   invoke()   -> {type(e).__name__}: {str(e)[:160]}")
