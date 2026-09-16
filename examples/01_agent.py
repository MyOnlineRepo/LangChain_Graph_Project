"""Ebene 1: Fertiger Agent-Loop mit `create_agent` (LangChain 1.x).

Der schnellste Weg zu einem funktionierenden Agenten. Laeuft intern
bereits auf der LangGraph-Runtime.

    uv run python examples/01_agent.py
"""

from langchain.agents import create_agent
from langchain.tools import tool

from lcgraph import MODEL, load_env

load_env()


@tool
def get_weather(city: str) -> str:
    """Gibt das aktuelle Wetter fuer eine Stadt zurueck."""
    # Stub - hier wuerde ein echter API-Call stehen.
    return f"In {city} sind es 18 Grad und es ist bewoelkt."


@tool
def add(a: float, b: float) -> float:
    """Addiert zwei Zahlen."""
    return a + b


agent = create_agent(
    model=MODEL,
    tools=[get_weather, add],
    system_prompt="Du bist ein knapper, praeziser Assistent. Antworte auf Deutsch.",
)


def main() -> None:
    frage = "Wie ist das Wetter in Hamburg? Und was ist 17 + 25?"
    result = agent.invoke({"messages": [{"role": "user", "content": frage}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
