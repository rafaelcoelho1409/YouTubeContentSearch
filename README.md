<p align="center"><img src="assets/YouTubeContentSearch_logo.png" alt="YouTube Content Search" width="220"></p>

<p align="center"><strong>GraphRAG over YouTube transcripts — four microservices, two Kubernetes flavors, and the research that became COELHO Nexus.</strong></p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white"></a>
  <a href="https://python.langchain.com/"><img alt="LangChain" src="https://img.shields.io/badge/LangChain-agents-1C3C3C"></a>
  <a href="https://langchain-ai.github.io/langgraph/"><img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-state%20machines-1C3C3C"></a>
  <a href="https://neo4j.com/"><img alt="Neo4j" src="https://img.shields.io/badge/Neo4j-knowledge%20graph-018BFF?logo=neo4j&logoColor=white"></a>
  <a href="https://fastapi.tiangolo.com/"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white"></a>
  <a href="https://mlflow.org/"><img alt="MLflow" src="https://img.shields.io/badge/MLflow-tracing-0194E2?logo=mlflow&logoColor=white"></a>
  <a href="https://kubernetes.io/"><img alt="Kubernetes" src="https://img.shields.io/badge/Kubernetes-k3d%20%2B%20minikube-326CE5?logo=kubernetes&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://rafaelcoelho.pages.dev/work/youtube-content-search">Portfolio Page</a> ·
  <a href="./YouTubeContentSearch.pdf">PDF Presentation</a>
</p>

---

## Table of Contents

- [What is this?](#what-is-this)
- [Architecture: four services, not one script](#architecture-four-services-not-one-script)
- [The GraphRAG pipeline](#the-graphrag-pipeline)
- [Multi-provider LLM routing](#multi-provider-llm-routing)
- [Observability](#observability)
- [Deployment: Compose, Minikube, or k3d](#deployment-compose-minikube-or-k3d)
- [Tech Stack](#tech-stack)
- [Notes on the current implementation](#notes-on-the-current-implementation)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Author](#author)

---

## What is this?

YouTube Content Search is the research project that became [COELHO Nexus](https://github.com/rafaelcoelho1409/COELHONexus) — it builds a **Knowledge Graph from YouTube video transcripts** (LangChain + LangGraph agents extract entities and relationships, Neo4j stores them) and answers questions through **GraphRAG**: hybrid retrieval that combines structured graph traversal with vector similarity search, rather than vector-similarity alone.

Four retrieval modes cover different starting points:

| Mode | What it does |
|---|---|
| **Search** | An agent turns free-text context into a YouTube search query, then builds the graph from the results |
| **Video** | Builds the graph from one specified video |
| **Channel** | Builds the graph from a channel's video corpus |
| **Playlist** | Builds the graph from a playlist's video corpus |

Every mode ends in the same place: a follow-up chat where a second agent answers questions by querying the Knowledge Graph built from that run's transcripts.

## Architecture: four services, not one script

This isn't a single Streamlit app — it's four containers wired together (`docker-compose.yml`):

| Service | Role | Port |
|---|---|---|
| **Streamlit** | UI, and — unusually — where the LangGraph state machines actually run | 8501 |
| **FastAPI** | All the real work: LLM calls, YouTube/transcript fetching, entity extraction, Neo4j reads/writes | 8000 |
| **Neo4j** | Knowledge Graph storage, ships with the APOC and Graph Data Science plugin bundles enabled | 7474 (HTTP) / 7687 (Bolt) |
| **MLflow** | Autologs every LangChain call — prompts, completions, timings — with zero manual instrumentation | 5000 |

The split is deliberate but unusual: LangGraph's `StateGraph` orchestration (`streamlit_app/models/youtube_content_search.py`) lives in the **Streamlit** container, while every node in that graph is a thin wrapper that calls out to a **FastAPI** endpoint to do the actual work (`fastapi_app/app.py`). FastAPI owns the LLM client, the Neo4j driver, the graph transformer — Streamlit owns only the state machine shape and the chat UI.

Because Streamlit reruns its whole script on every interaction, chat history can't just live in local variables — so FastAPI also acts as a **shared UI-state broker**: every rendering instruction (which chat bubble, which expander, which Streamlit call with which arguments) is PUT to `/streamlit_actions` as it happens, and the Streamlit script GETs and replays the full list on every rerun. It's a working solution to Streamlit's stateless-rerun model without a separate database.

## The GraphRAG pipeline

**Building the graph** (`set_knowledge_graph` node → several FastAPI endpoints):
1. Fetch each video's transcript (`youtube_transcript_api`)
2. Split into 512-token chunks, 24-token overlap (`TokenTextSplitter`)
3. Run each chunk through `LLMGraphTransformer` — the LLM itself extracts nodes and relationships
4. Write to Neo4j (`Neo4jGraph.add_graph_documents`) and build a fulltext index on entity IDs
5. The graph is cleared (`MATCH (n) DETACH DELETE n`) at the start of every new run — this is a fresh, session-scoped graph each time, not one that accumulates across searches

**Answering a question** (`rag_chain` endpoint) retrieves two ways in parallel and merges both into context:
- **Structured**: extract entities from the question (a dedicated LLM call), fuzzy full-text match them against the graph (`~2` edit-distance tolerance for misspellings), then traverse one hop in both directions to collect `entity - RELATION -> entity` triples
- **Unstructured**: `Neo4jVector` hybrid search (vector + keyword) over the same chunks, embedded with `sentence-transformers/all-MiniLM-L6-v2`

A `RunnableBranch` condenses multi-turn chat history into a standalone question before retrieval when there's prior conversation, so follow-up questions ("what about the costs?") resolve correctly against the graph.

## Multi-provider LLM routing

| Framework | Status | Notes |
|---|---|---|
| **Groq** | Wired, active | 14 models incl. Llama 3.3 70B, DeepSeek-R1-distill variants |
| **OpenAI** | Wired, active | Incl. o1 / o1-mini / o3-mini reasoning models |
| **SambaNova** | Wired, active | Incl. full DeepSeek-R1, Llama 3.1 405B |
| **Scaleway** | Wired, active | Routed through `ChatOpenAI` against Scaleway's OpenAI-compatible endpoint |
| **Google Generative AI** | Built, disabled in UI | `ChatGoogleGenerativeAI` + Gemini 1.5 Pro are wired into the backend's provider dict but commented out of the Streamlit settings selectbox |
| **Ollama** | Partial | Full model-selection UI (start/stop, active-model detection via `ollama.ps()`), but the `/youtube_content_search` endpoint's provider-dispatch dict doesn't currently include an Ollama branch |

The presentation deck shows this tested against Meta Llama 3.1/3.2/3.3 and Google Gemma 2 via Ollama, alongside the four production providers above.

## Observability

- **MLflow autologging** — every LangChain call is traced automatically (`mlflow.langchain.autolog()`); the deck's own Traces tab shows real request/response pairs with execution time, e.g. a cost-comparison question resolved in 1.6s with the actual figures ("15,000 Euros per year... 1,600 to 1,900 Euros for a one-bedroom apartment...") pulled from the graph.
- **LangGraph introspection** — a "View application graphs" button renders both state machines (the 3-node ingestion graph and the 1-node chatbot graph) as Mermaid diagrams, live, from the running graph objects.
- **Neo4j graph explorer** — a "View Neo4j context graph" button queries the live graph, builds a NetworkX `DiGraph`, and renders it as an interactive Pyvis network embedded directly in Streamlit; the same graph is also reachable directly through the Neo4j Browser at `:7474`.
- **FastAPI's own `/docs`** — every endpoint above is independently callable and testable through FastAPI's generated Swagger UI.

## Deployment: Compose, Minikube, or k3d

All three paths build the same four images from `docker-compose.yml`.

**Docker Compose** — `docker compose up --build`, then Streamlit :8501 / FastAPI :8000 / MLflow :5000 / Neo4j :7474.

**Minikube** — `k8s-manifest-minikube.yml` maps each service to a fixed NodePort (Streamlit 30003, FastAPI 30002, MLflow 30004, Neo4j 30001 HTTP / 31132 Bolt on its own port, since Bolt and HTTP need separately advertised addresses to be reachable from outside the pod network).

**k3d** — `k8s-manifest-k3d.yml` follows the same layout; because k3d doesn't share the host's Docker daemon, each image has to be explicitly imported into the cluster first (`k3d image import ... -c mycluster`) before applying the manifest. Both sets of exact commands are recorded in `k8s_deploy_commands.txt`, including the Kubernetes Dashboard bootstrap via Helm.

## Tech Stack

| Technology | Role |
|---|---|
| **LangChain / LangGraph** | Agent orchestration, structured-output extraction, state machines |
| **Neo4j** (+ APOC, Graph Data Science) | Knowledge Graph storage, Cypher traversal, fulltext + vector hybrid search |
| **FastAPI** | Backend service — LLM calls, YouTube/transcript fetching, Neo4j access |
| **Streamlit** | UI shell and LangGraph execution host |
| **MLflow** | Automatic LangChain call tracing |
| **PyTubeFix** | YouTube search, video/channel/playlist metadata (actively-maintained `pytube` fork) |
| **HuggingFace `sentence-transformers`** | `all-MiniLM-L6-v2` embeddings for the vector half of retrieval |
| **Docker Compose / Kubernetes (k3d, Minikube)** | Local multi-service orchestration |
| **uv** | Per-service dependency management (`pyproject.toml` + lockfile in each of the four apps) |

## Notes on the current implementation

A few honest details, since accuracy matters more than polish:

- Each Dockerfile runs `uv add -r requirements.txt` in its `CMD` rather than at image build time — dependencies install fresh on every container start, trading build-layer caching for the live-reload bind-mounts already in `docker-compose.yml`.
- Neo4j ships with the Graph Data Science plugin enabled, but no query in the current codebase calls a `gds.*` procedure yet — retrieval uses plain Cypher traversal and fulltext search.
- Three `mlflow.langchain.log_model(...)` calls (search agent, entity chain, RAG chain) are commented out — autologging/tracing is fully active, but registering these chains as versioned MLflow Models isn't currently exercised.
- `fastapi_app/app.py` imports `streamlit` and instantiates a `StreamlitCallbackHandler` inside the FastAPI process — a carryover from the Streamlit-side code that has no active Streamlit script context to attach to in that container.

## Prerequisites

| Requirement | Notes |
|---|---|
| **Docker** + **Docker Compose** | Primary path |
| **Minikube** or **k3d** | Only needed for the Kubernetes paths |
| API keys | For whichever of Groq / OpenAI / SambaNova / Scaleway you intend to use — entered live in the Settings dialog, or via `.env` |

## Installation

```bash
git clone https://github.com/rafaelcoelho1409/YouTubeContentSearch
cd YouTubeContentSearch

docker compose up --build
```

Then open Streamlit (`:8501`), pick a framework and model in **Settings**, and choose a search mode. See [`k8s_deploy_commands.txt`](./k8s_deploy_commands.txt) for the Minikube/k3d command sequences.

## Project Structure

```
YouTubeContentSearch/
├── streamlit_app/
│   ├── app.py                       # Sidebar, search-mode forms, chat loop
│   ├── functions.py                  # Settings dialog, graph viewers, Neo4j visualization
│   └── models/
│       └── youtube_content_search.py   # The two LangGraph StateGraphs
│
├── fastapi_app/
│   └── app.py                       # LLM routing, YouTube/transcript fetching, GraphRAG, Neo4j
│
├── mlflow_app/                      # MLflow tracking server (autolog target)
├── neo4j_app/                       # Neo4j + APOC + GDS Dockerfile
│
├── docker-compose.yml
├── k8s-manifest-minikube.yml
├── k8s-manifest-k3d.yml
├── k8s_deploy_commands.txt           # Exact command sequences for both K8s paths
│
├── YouTubeContentSearch.pdf           # Deployment record / demo deck
└── requirements.txt
```

## Author

**Rafael Coelho** — [rafaelcoelho.pages.dev](https://rafaelcoelho.pages.dev/)
