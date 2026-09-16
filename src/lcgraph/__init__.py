"""LangChain / LangGraph Lernprojekt."""

__all__ = ["MODEL", "MODEL_SCHNELL", "load_env", "nur_text"]

# Hauptmodell fuer inhaltliche Arbeit (Generierung, Fachantworten).
MODEL = "anthropic:claude-sonnet-5"

# Guenstigeres Modell fuer einfache Entscheidungen: Routing, Klassifikation,
# Relevanz-Bewertung. Halber Preis, reicht fuer Ein-Wort-Urteile.
MODEL_SCHNELL = "anthropic:claude-haiku-4-5"


def nur_text(inhalt: str | list) -> str:
    """Holt den reinen Text aus einer Modellantwort.

    `AIMessage.content` ist bei denkenden Modellen keine Zeichenkette, sondern
    eine Liste von Bloecken ({"type": "thinking", ...}, {"type": "text", ...}).
    Wer sie ungefiltert weiterreicht, schickt Signatur-Blobs als Tokens ins
    naechste Prompt - teuer und nutzlos.
    """
    if isinstance(inhalt, str):
        return inhalt
    teile = [b.get("text", "") for b in inhalt if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(teile).strip()


def load_env() -> None:
    """Laedt .env aus dem Projekt-Root, falls vorhanden."""
    from dotenv import load_dotenv

    load_dotenv()
