"""Ebene 5: MCP-Anbindung - externe Werkzeuge ueber den Standard.

In 01 waren die Tools Python-Funktionen im selben Modul. Hier kommen sie aus
einem MCP-Server: der Agent weiss nicht, wie sie implementiert sind, und der
Server weiss nichts von LangChain. Genau das ist der Punkt des Protokolls -
Werkzeuge und Agent sind entkoppelt.

`MCPAdapter` uebernimmt Verbindung und Uebersetzung; heraus fallen ganz
normale LangChain-Tools, die `create_agent` direkt frisst.

Die Tools sind asynchron, deshalb laeuft dieses Beispiel ueber asyncio.

    uv run python examples/05_mcp_tools.py
"""

import asyncio
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcp_projekt_server import mcp as mcp_server  # noqa: E402  - braucht den Pfad oben

from lcgraph import MODEL, load_env  # noqa: E402

load_env()


async def main() -> None:
    # Target ist hier der Server im selben Prozess - robust und ohne Subprozess.
    # Ein echter externer Server waere stattdessen eine URL ("https://.../mcp")
    # oder ein Pfad: MCPAdapter(Path("examples/05_mcp_server.py")).
    async with MCPAdapter(mcp_server) as adapter:
        tools = await adapter.list_tools()
        print("Vom Server gemeldete Tools:")
        for t in tools:
            print(f"  - {t.name}: {t.description.splitlines()[0]}")

        agent = create_agent(
            model=MODEL,
            tools=tools,
            system_prompt=(
                "Du erkundest ein Python-Projekt ausschliesslich mit den "
                "bereitgestellten Werkzeugen. Antworte knapp und auf Deutsch."
            ),
        )

        frage = (
            "Welche Modell-IDs sind in src/lcgraph/__init__.py hinterlegt "
            "und wofuer ist jede gedacht?"
        )
        print(f"\nFrage: {frage}\n")
        ergebnis = await agent.ainvoke({"messages": [{"role": "user", "content": frage}]})

        for m in ergebnis["messages"]:
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    print(f"  [Tool-Aufruf] {tc['name']}({tc['args']})")

        print("\n--- Antwort ---")
        print(ergebnis["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
