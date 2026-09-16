# LangChain / LangGraph — Überblick

Stand: 2026-09-17 · langchain 1.4.1 · langgraph 1.2.11 · Python 3.14.3

---

## 1. Das mentale Modell

Seit dem gemeinsamen **1.0-Release (22.10.2025)** ist die Arbeitsteilung sauber. Die alte
Frage „LangChain *oder* LangGraph?" ist hinfällig — es ist ein Stack:

- **LangGraph = Runtime.** Ein Graph aus *Nodes* (Funktionen, die State lesen und Updates
  zurückgeben) und *Edges* (was als Nächstes läuft). Kein Syntax-Zucker, sondern eine
  Ausführungsmaschine mit Persistenz, Streaming und Unterbrechbarkeit.
- **LangChain = Abstraktionsschicht darüber.** `create_agent()` ist der schnelle Weg zum
  klassischen Agent-Loop (Modell ↔ Tools ↔ Modell) und läuft intern auf der LangGraph-Runtime.
  Dazu Modell-Abstraktion (`"anthropic:claude-sonnet-5"` statt SDK-spezifischem Code),
  Tools, Middleware.

Die reale Entscheidung lautet also nicht „welches Framework", sondern:
**Reicht der vorgefertigte Agent-Loop, oder brauche ich expliziten Kontrollfluss?**

---

## 2. Die vier Kernkonzepte von LangGraph

| Konzept | Was es tut |
|---|---|
| **State** | Ein `TypedDict`, das durch den Graph fließt. Felder können *Reducer* haben — `Annotated[list, add_messages]` hängt an, statt zu überschreiben. Das ist der zentrale Trick. |
| **Nodes / Edges** | Nodes tun die Arbeit, Edges routen. `add_conditional_edges` verzweigt anhand einer Routing-Funktion → Zyklen, Branches, Multi-Agent-Topologien. |
| **Checkpointer** | Speichert den kompletten State nach *jedem Superstep*, key = `thread_id`. Daraus folgen Gedächtnis, Crash-Recovery, Time Travel. `InMemorySaver` zum Spielen, Postgres/Redis für Prod. |
| **`interrupt` / `Command`** | Der Graph hält an, wartet auf eine menschliche Entscheidung, läuft per `Command(resume=...)` weiter — auch Tage später, aus der DB heraus. |

### Der wichtigste Fallstrick

Beim Resume werden Teile eines Nodes **erneut ausgeführt**. Nichtdeterministisches und
Seiteneffekte (API-Calls, Schreibvorgänge) gehören in eigene Nodes bzw. müssen idempotent
sein. Die offiziellen Docs verschweigen das erfreulicherweise nicht.

---

## 3. Was 2026 dazugekommen ist

- **MCP nativ in LangChain** (v1.4, Sep 2026): Namespace `langchain.mcp` auf FastMCP-Basis,
  inkl. OAuth und Elicitation über LangGraph-Interrupts. Installation: `pip install "langchain[mcp]"`.
- **Streaming v3** (langgraph 1.2): typisierte, kanalweise Projektionen — `run.values`,
  `run.messages`, `run.lifecycle`, `run.subgraphs`.
- **Fault Tolerance** (langgraph 1.2): Timeouts und Error-Handler pro Node, Graceful Shutdown
  mit fortsetzbaren Checkpoints.
- **DeltaChannel** (Beta): speichert nur Deltas statt vollem State — deutlich weniger
  Checkpoint-Overhead bei langen Threads.
- **Middleware**: Guardrails, PII-Redaction, Memory als komponierbare Layer um den Agent-Loop.
- **Deploy-CLI**: Deployment nach LangSmith direkt aus dem Terminal.

---

## 4. Brauche ich Daten?

**Trainingsdaten nie** — hier wird nichts fine-getuned, das Modell kommt fertig über die API.
„Daten" heißt bei LangGraph immer nur: *woher holt sich der Agent zur Laufzeit seinen Kontext.*

| Richtung | Daten nötig? |
|---|---|
| Agent-Loop, Tools, Routing | **Nein.** Der State entsteht im Lauf, Tools holen alles live. |
| Multi-Agent-Supervisor | **Nein.** Reine Kontrollfluss-Topologie. |
| MCP-Anbindung | **Nein** — der MCP-Server bringt die Daten mit (Filesystem, GitHub, DB). |
| **RAG** | **Ja.** Die einzige Richtung, die einen Korpus voraussetzt. |

### Falls RAG: zwei Dinge fehlen dann

1. **Ein Korpus** — eigene PDFs/Markdown in einem Ordner reichen für den ersten Graph.
   Alternativ die LangChain-Docs selbst crawlen (angenehm selbstreferenziell, Qualität
   sofort prüfbar).
2. **Ein Embedding-Modell — nicht von Anthropic.** Anthropic bietet keinen Embeddings-Endpunkt
   an und verweist in den eigenen Docs auf Voyage AI. Also entweder ein zweiter Key (Voyage,
   OpenAI) oder lokal: `fastembed` / `sentence-transformers` laufen offline auf der CPU und
   reichen für Experimente. Dazu ein Vector Store — Chroma oder pgvector lokal.

**Empfehlung:** ohne Daten anfangen. Ein Multi-Agent-Supervisor oder ein Graph mit echtem
API-Tool zeigt die LangGraph-Konzepte klarer als RAG — bei RAG geht die erste Hälfte der Zeit
für Chunking und Embeddings drauf statt für den Graph.

---

## 5. Der Workspace

```
D:\Git\LangChain_Graph_Project
├─ pyproject.toml / uv.lock     uv, src-Layout, ruff + pytest
├─ CLAUDE.md                    Konventionen für künftige Sessions
├─ README.md                    Kurzeinstieg
├─ OVERVIEW.md                  dieses Dokument
├─ LICENSE                      MIT
├─ .env.example                 ANTHROPIC_API_KEY, optional LangSmith
├─ src/lcgraph/
│  ├─ __init__.py               MODEL, MODEL_SCHNELL, load_env(), nur_text()
│  └─ embeddings.py             fastembed als LangChain-Embeddings
├─ examples/
│  ├─ 01_agent.py               create_agent mit zwei Tools
│  ├─ 02_state_graph.py         State + Reducer + bedingtes Routing
│  ├─ 03_durable_hitl.py        SqliteSaver + interrupt + resume
│  ├─ 04_supervisor.py          Multi-Agent-Supervisor, Routing im Zyklus
│  ├─ 05_mcp_tools.py           MCPAdapter → Tools → create_agent
│  ├─ mcp_projekt_server.py     der MCP-Server dazu (kein eigener Lauf)
│  ├─ 06_rag_ingest.py          Korpus → Chunks → Chroma-Index
│  └─ 07_rag_graph.py           selbstkorrigierendes RAG, zwei Zyklen
├─ experimente/                 Wegwerf-Skripte zum Begreifen (ohne API-Key)
│  ├─ exp1_reducer.py           mit/ohne Reducer im Vergleich
│  ├─ exp2_routing.py           Kante ins Leere: Compile- vs. Laufzeit
│  └─ exp3_stream.py            stream_mode "updates" vs. "values"
└─ .claude/skills/              gevendorte Agent-Skills (MIT, mattpocock)
```

Die Beispiele sind bewusst aufsteigend: 01 zeigt, wie wenig man braucht; 02 zeigt, was
darunter liegt; 03 zeigt, was LangGraph von einer for-Schleife unterscheidet. 04 bis 07
sind die drei Ausbaurichtungen aus der vormals offenen Entscheidung (siehe Abschnitt 6).

Wichtig für 07: erst `06_rag_ingest.py` laufen lassen, sonst fehlt der Index. Der landet
unter `.chroma/` und ist gitignored — er entsteht in Sekunden neu.

### Setup

```bash
uv sync
cp .env.example .env            # ANTHROPIC_API_KEY eintragen
uv run python examples/01_agent.py
```

`uv sync` lief ohne Downgrade auf Python 3.14.3 durch — alle Wheels vorhanden.

### Verifikationsstand

Stand 2026-09-17: **alle Beispiele wurden ausgeführt**, `ruff check` und `ruff format` clean.

- 01–03 laufen. Bei 03 wurde zusätzlich aus einem *fremden* Prozess nachgewiesen, dass der
  State die Prozessgrenze überlebt: fünf Checkpoints in der SQLite-DB, einer pro Superstep.
- 04–07 laufen. Der RAG-Graph wurde gegen eine Frage getestet, deren Antwort **nicht** im
  Korpus steht: drei Umformulierungen, Bremse gegriffen, Grounding-Check auf `gedeckt=False`
  — und die Antwort sagt, dass die Information fehlt, statt zu halluzinieren.
- Gesamtkosten aller Läufe: deutlich unter einem Euro (Sonnet 5 + Haiku 4.5 gemischt).

Zwei Fallstricke, die dabei auffielen:

- **Prefill.** Reicht man in einem Multi-Agent-Graph die Nachrichtenliste an den nächsten
  Worker weiter, endet sie auf einer Assistant-Nachricht. Aktuelle Claude-Modelle lehnen das
  mit HTTP 400 ab. Jeder Worker braucht einen expliziten Auftrag als User-Nachricht.
- **`content` ist keine Zeichenkette.** Bei denkenden Modellen ist es eine Liste aus
  Thinking- und Text-Blöcken. Ungefiltert weitergereicht kostet das Tokens für Signatur-Blobs.
  Dafür gibt es `nur_text()`.

---

## 6. Entschieden — alle drei Richtungen gebaut

Die vormals offene Entscheidung ist erledigt: statt einer Richtung wurden alle drei gebaut.

| Richtung | Beispiel | Kern |
|---|---|---|
| **Multi-Agent-Supervisor** | `04_supervisor.py` | Supervisor routet zyklisch zwischen drei Spezialisten, `MAX_SCHRITTE` bremst. Haiku entscheidet, Sonnet arbeitet. |
| **MCP-Tool-Anbindung** | `05_mcp_tools.py` + `mcp_projekt_server.py` | `MCPAdapter` übersetzt MCP-Tools in LangChain-Tools. Agent und Werkzeuge wissen nichts voneinander. |
| **RAG-Graph** | `06_rag_ingest.py` + `07_rag_graph.py` | Zwei Zyklen: Retrieval-Korrektur (umformulieren → neu suchen) und Grounding-Check (Antwort nicht gedeckt → neu schreiben). |

### Die RAG-Entscheidungen aus Abschnitt 4 — so gefallen

- **Korpus:** das Markdown, das ohnehin im Repo liegt (Projektdokus + die 25 Skill-Dateien).
  Kein Crawler, kein Download, reproduzierbar — und jede Antwort ist gegenprüfbar.
- **Embedding-Modell:** `paraphrase-multilingual-MiniLM-L12-v2` über `fastembed`, lokal auf
  der CPU, 220 MB. **Kein zweiter API-Key.** Mehrsprachig, damit deutsche Fragen die
  englischen Skill-Texte finden.
- **Vector Store:** Chroma, lokal nach `.chroma/` persistiert.
- **Bewusstes Nicht-Ziel:** Chunk-Tuning. Eine Konfiguration (1200/150), fertig — die Zeit
  gehört dem Graphen, nicht der Chunk-Größe.

### Was als Nächstes offen ist

- **07 mit Checkpointer und HITL verbinden.** Scheitert der Grounding-Check zweimal, könnte
  der Graph per `interrupt` nachfragen, statt still aufzugeben — das wäre 03 und 07 vereint.
- **Tests.** `pytest` ist als Dev-Abhängigkeit installiert, aber es gibt kein `tests/`.
- **Konsolen-Encoding.** Deutsche Ausgaben brauchen unter Windows `PYTHONIOENCODING=utf-8`,
  sonst wird aus `bewölkt` ein `bew?lkt`. Ein Zweizeiler in `load_env()` würde das erledigen.

---

## Quellen

- [LangChain and LangGraph Agent Frameworks Reach v1.0](https://www.langchain.com/blog/langchain-langgraph-1dot0)
- [Changelog — Docs by LangChain](https://docs.langchain.com/oss/python/releases/changelog)
- [Persistence / Durable Execution](https://docs.langchain.com/oss/python/langgraph/durable-execution)
- [Agents — create_agent](https://docs.langchain.com/oss/python/langchain/agents)
- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Embeddings — Claude Docs](https://docs.claude.com/en/docs/build-with-claude/embeddings)
