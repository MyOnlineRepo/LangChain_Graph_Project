# CLAUDE.md

## Projekt

Lern- und Experimentierprojekt fuer LangChain 1.x / LangGraph 1.x (Python).
Kein Produktivsystem - Beispiele sollen kurz und lesbar bleiben.

## Umgebung

- Paketmanager: **uv** (nicht pip). Abhaengigkeiten via `uv add`, Ausfuehrung via `uv run`.
- Python >= 3.12, src-Layout, Paket `lcgraph` unter `src/lcgraph/`.
- Secrets ausschliesslich in `.env` (gitignored), Vorlage in `.env.example`.

## Befehle

```bash
uv sync                              # Environment aufbauen
uv run python examples/01_agent.py   # Beispiel starten
uv run ruff check . --fix            # Lint
uv run ruff format .                 # Format
uv run pytest                        # Tests
```

## Konventionen

- Modell-ID zentral in `src/lcgraph/__init__.py` (`MODEL`), nicht in Beispielen hardcoden.
- Beispiele in `examples/` sind nummeriert und steigen in der Komplexitaet;
  jede Datei hat einen Docstring mit Thema und Aufrufbefehl.
- Neue LangGraph-Graphen: State als `TypedDict` mit Reducern (`Annotated[..., add_messages]`),
  Graph-Bau am Modulende, `main()` nur fuer den Demo-Lauf.
- Deutschsprachige Kommentare und Prompts, ASCII statt Umlaute im Code.
