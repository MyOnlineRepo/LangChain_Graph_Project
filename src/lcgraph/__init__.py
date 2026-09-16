"""LangChain / LangGraph Lernprojekt."""

__all__ = ["MODEL", "load_env"]

MODEL = "anthropic:claude-sonnet-5"


def load_env() -> None:
    """Laedt .env aus dem Projekt-Root, falls vorhanden."""
    from dotenv import load_dotenv

    load_dotenv()
