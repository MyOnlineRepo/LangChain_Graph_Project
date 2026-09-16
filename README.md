# LangChain / LangGraph Projekt

Lern- und Experimentierumgebung fuer die LangChain-1.x- und LangGraph-1.x-Stacks.

## Setup

```bash
uv sync
cp .env.example .env   # ANTHROPIC_API_KEY eintragen
```

## Beispiele

| Datei | Thema |
|---|---|
| `examples/01_agent.py` | `create_agent` - fertiger Agent-Loop mit Tools |
| `examples/02_state_graph.py` | `StateGraph` - State, Nodes, Edges, bedingtes Routing |
| `examples/03_durable_hitl.py` | Checkpointer (SQLite), `interrupt`, Resume via `Command` |

```bash
uv run python examples/01_agent.py
```

## Stack

- `langchain` 1.4 - Agent-Loop, Tools, Middleware, Model-Abstraktion, MCP
- `langgraph` 1.2 - Runtime: Graph, Persistenz, Streaming, Human-in-the-Loop
- `langgraph-checkpoint-sqlite` - lokale Persistenzschicht (Prod: Postgres/Redis)

Optional: LangSmith fuer Tracing (`LANGSMITH_TRACING=true` in `.env`).
