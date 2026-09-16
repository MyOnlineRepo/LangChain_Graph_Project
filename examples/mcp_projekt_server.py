"""Ein kleiner MCP-Server - die Gegenstelle fuer 05_mcp_tools.py.

Stellt zwei Werkzeuge ueber das Model Context Protocol bereit, mit denen ein
Agent das Projektverzeichnis erkunden kann. Bewusst auf das Projekt-Root
eingesperrt: ein MCP-Server ist ein Prozess mit echtem Dateizugriff, und
"welche Pfade darf er sehen" ist keine Nebensache, sondern seine Kernfrage.

Diese Datei wird nicht direkt gestartet - 05_mcp_tools.py importiert sie.
Als eigenstaendiger Prozess liesse sie sich so betreiben:

    uv run fastmcp run examples/mcp_projekt_server.py:mcp
"""

from pathlib import Path

from fastmcp import FastMCP

# Projekt-Root = das Verzeichnis ueber examples/
WURZEL = Path(__file__).resolve().parent.parent

mcp = FastMCP("lcgraph-projekt")


def _sicherer_pfad(pfad: str) -> Path:
    """Loest einen Pfad auf und verweigert alles ausserhalb der Wurzel."""
    ziel = (WURZEL / pfad).resolve()
    if not ziel.is_relative_to(WURZEL):
        msg = f"Zugriff ausserhalb des Projekts verweigert: {pfad}"
        raise ValueError(msg)
    return ziel


@mcp.tool
def dateien_auflisten(muster: str = "*.py") -> list[str]:
    """Listet Dateien im Projekt auf, die zum Glob-Muster passen.

    Args:
        muster: Glob-Muster, z. B. '*.py', '*.md' oder 'examples/*.py'.
    """
    treffer = sorted(p for p in WURZEL.glob(muster) if p.is_file())
    return [str(p.relative_to(WURZEL)) for p in treffer[:50]]


@mcp.tool
def datei_lesen(pfad: str, max_zeichen: int = 4000) -> str:
    """Liest eine Textdatei aus dem Projekt.

    Args:
        pfad: Pfad relativ zum Projekt-Root, z. B. 'src/lcgraph/__init__.py'.
        max_zeichen: Obergrenze, damit grosse Dateien das Kontextfenster nicht fluten.
    """
    ziel = _sicherer_pfad(pfad)
    if not ziel.is_file():
        return f"Nicht gefunden: {pfad}"
    text = ziel.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_zeichen:
        return text[:max_zeichen] + f"\n... (gekuerzt, {len(text)} Zeichen gesamt)"
    return text
