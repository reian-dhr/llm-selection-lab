# Agent instructions

Guidance for coding agents (and human contributors) working on this repository.

## What this project is
A tool to choose the best LLM for a given task. The web app sends one request to models from
several providers, a judge model picks the best answer (the others stay one click away), and
the user's preferences teach the judge. The command line and an MCP server run a fuller
check of a kind of task: a drafted test, cost and speed per model, and a recommendation.
It is a point-in-time check ("which model is best today for this task"), not a production
router.

## Layout
- `src/model_selection_lab/`: the package
  - `models/`: one unified call per provider (output, tokens, latency, cost), model registry, retries
  - `runs/`: task drafting, task loading, batch runner, results storage
  - `judge/`: the user's picks and the learning judge
  - `report/`: per-model summary, recommendation, cost of one real job
  - `ask.py`: one request on every standard model, then the judge's pick (used by the web app)
  - `check.py`: the fuller check of a kind of task; `mcp_server.py`: the same check as MCP tools
  - `remote_store.py`: copy of saved files in a private GitHub repository, for online hosting
  - `settings.py`: every run setting and its default, in one place
- `app/`: Streamlit web app
- `scripts/`: command-line entry points
- `examples/`: generic prompts and an example task, used when no private version exists
- `tests/`: pytest tests, fully offline (fake model clients in `tests/fakes.py`)

## Commands
- Install: `uv sync` (use uv only, never pip or conda; add a dependency with `uv add <package>`)
- Tests: `uv run pytest`
- Web app: `uv run streamlit run app/main.py`
- One check from the terminal: `uv run python scripts/check_task.py "describe the task"`
- MCP server: `uv run python -m model_selection_lab.mcp_server`

## Coding style
- Simple, readable Python: clear names, short functions, no clever abstractions.
- Official provider SDKs only (Anthropic, OpenAI, Mistral, Groq). No LangChain or LlamaIndex.
- New settings go in `settings.py` with a default, not as new parameters the user must set.
- Prompts and judge rules are loaded from files at run time, never written into `src/`.
- Every new module gets offline tests; tests must never call a real API.
- No long dashes (em dashes) in code, comments, docs or commit messages.

## Cost and safety
- Anything that calls paid models must show a cost estimate first and respect `MAX_RUN_COST_USD`.
- Premium (expensive) models run only when explicitly requested.
- API keys live in `.env` only: never print, log or commit them, and never change `.gitignore` to allow `.env`.
- Public or synthetic data only.
- Model names, prices and SDK syntax change often: check the providers' official docs
  instead of relying on memory, and cite the source.
