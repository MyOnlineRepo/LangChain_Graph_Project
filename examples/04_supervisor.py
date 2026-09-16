"""Ebene 4: Multi-Agent-Supervisor - Routing zwischen Spezialisten.

Ein Supervisor entscheidet wiederholt, welcher Spezialist als Naechstes
arbeitet, und beendet den Lauf, wenn die Aufgabe erledigt ist. Das ist ein
Zyklus: supervisor -> spezialist -> supervisor -> ... -> END.

Neu gegenueber 02: Dort war das Routing eine Einbahnstrasse (klassifizieren,
einmal abbiegen, fertig). Hier kehrt der Fluss immer wieder zum Supervisor
zurueck - deshalb braucht es eine Abbruchbremse.

Keine Daten noetig - reine Kontrollfluss-Topologie.

    uv run python examples/04_supervisor.py
"""

from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from lcgraph import MODEL, MODEL_SCHNELL, load_env, nur_text

load_env()

# Bremse: ohne sie koennte der Supervisor endlos im Kreis routen.
MAX_SCHRITTE = 6

SPEZIALISTEN = {
    "mathe": "Du bist Mathematiker. Rechne praezise und zeige den Rechenweg knapp.",
    "texter": "Du bist Texter. Formuliere knapp, konkret und ohne Floskeln.",
    "kritiker": "Du bist Kritiker. Benenne Schwachstellen in maximal drei Stichpunkten.",
}


class State(TypedDict):
    messages: Annotated[list, add_messages]
    naechster: str
    schritte: int


class Entscheidung(BaseModel):
    """Strukturierte Routing-Entscheidung des Supervisors."""

    naechster: Literal["mathe", "texter", "kritiker", "FERTIG"] = Field(
        description="Wer als Naechstes arbeitet, oder FERTIG wenn die Aufgabe erledigt ist."
    )
    begruendung: str = Field(description="Ein kurzer Satz zur Begruendung.")


# Guenstiges Modell fuer die Entscheidung, starkes fuer die eigentliche Arbeit.
supervisor_llm = init_chat_model(MODEL_SCHNELL).with_structured_output(Entscheidung)
arbeiter_llm = init_chat_model(MODEL)


def supervisor(state: State) -> dict:
    """Entscheidet, wer als Naechstes arbeitet - oder ob Schluss ist."""
    if state["schritte"] >= MAX_SCHRITTE:
        return {"naechster": "FERTIG", "schritte": state["schritte"]}

    verlauf = "\n".join(f"{m.type}: {nur_text(m.content)}" for m in state["messages"])
    entscheidung = supervisor_llm.invoke(
        "Du koordinierst ein Team aus 'mathe', 'texter' und 'kritiker'.\n"
        "Lies den bisherigen Verlauf und entscheide, wer als Naechstes arbeitet.\n"
        "Ist die Aufgabe vollstaendig beantwortet, antworte mit FERTIG.\n"
        "Rufe niemanden zweimal hintereinander ohne Grund auf.\n\n"
        f"Verlauf:\n{verlauf}"
    )
    print(f"    [supervisor] -> {entscheidung.naechster}: {entscheidung.begruendung}")
    return {"naechster": entscheidung.naechster, "schritte": state["schritte"] + 1}


def route(state: State) -> Literal["mathe", "texter", "kritiker", "__end__"]:
    """Bedingte Kante: uebersetzt die Entscheidung in den naechsten Node."""
    ziel = state["naechster"]
    return END if ziel == "FERTIG" else ziel


def baue_spezialist(name: str):
    """Erzeugt einen Worker-Node mit eigener Rolle."""

    def node(state: State) -> dict:
        # Wichtig: Die Nachrichtenliste endet nach dem ersten Worker auf einer
        # Assistant-Nachricht. Aktuelle Claude-Modelle lehnen das als "prefill"
        # mit HTTP 400 ab - der Verlauf muss auf einer User-Nachricht enden.
        # Deshalb bekommt jeder Spezialist seinen Auftrag explizit angehaengt.
        auftrag = {"role": "user", "content": f"Du bist dran als '{name}'. Liefere deinen Beitrag."}
        antwort = arbeiter_llm.invoke(
            [{"role": "system", "content": SPEZIALISTEN[name]}] + state["messages"] + [auftrag]
        )
        antwort.name = name
        return {"messages": [antwort]}

    return node


builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
for _name in SPEZIALISTEN:
    builder.add_node(_name, baue_spezialist(_name))

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route)
for _name in SPEZIALISTEN:
    # Jeder Spezialist meldet sich zurueck - das schliesst den Zyklus.
    builder.add_edge(_name, "supervisor")

graph = builder.compile()


def main() -> None:
    aufgabe = (
        "Ein Server kostet 90 Euro im Monat. Rechne die Jahreskosten aus "
        "und schreibe einen Zweizeiler, der die Ausgabe rechtfertigt."
    )
    zustand = graph.invoke(
        {"messages": [{"role": "user", "content": aufgabe}], "naechster": "", "schritte": 0}
    )

    print("\n--- Verlauf ---")
    for m in zustand["messages"]:
        wer = getattr(m, "name", None) or m.type
        print(f"\n[{wer}]\n{nur_text(m.content)}")
    print(f"\n--- {zustand['schritte']} Supervisor-Entscheidungen ---")


if __name__ == "__main__":
    main()
