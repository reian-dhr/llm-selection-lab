# LLM Selection Lab

[![tests](https://github.com/reian-dhr/llm-selection-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/reian-dhr/llm-selection-lab/actions/workflows/tests.yml)

Which LLM does your task best, today? Type a request once. Models from Anthropic, OpenAI
and Mistral all answer it, a judge model picks the best answer, and the others stay one
click away. When you prefer another answer, one click says so, and the judge learns your
standard from it.

For a fuller check of a kind of task, the command line and an MCP server draft a small test
from a one-sentence description and compare cost, speed and answers across models.

It is a point-in-time check for one decision, not a production router: models and prices
change every few months, so the useful question is "which model is best for this kind of
task right now".

## How it works

```
web app                                     command line / MCP
-------                                     ------------------
request (one box)                           "Turn meeting notes into a follow-up email"
   |                                            |
   v                                            v
cost estimate, refused above the cap        helper drafts a test (3 inputs, one with a trap)
   |                                            |
   v                                            v
every standard model answers                cost preview (free), then run on every model
   |                                            |
   v                                            v
judge picks the best (sees A-D, no names)   summary: cost, speed, failures, recommendation
   |
   v
best answer shown, others one click away
   |
   +--> "I prefer this one" --> saved as a pick --> an example for the judge next time
```

Both flows share the same runner (`runs/`), unified model client (`models/`) and
storage. Only the web app uses the judge (`judge/`) so far: a check compares cost and
speed, and saves every answer in `results/` to read.

## Design decisions

- **The judge cannot favour a brand.** It sees the answers as A to D in a shuffled order,
  without model names, and comes from a model family that is not being compared. Its letters
  are turned back into model names only for display.
- **The judge learns in context, not by training.** It gets the user's written rules and
  their most recent picks with reasons as few-shot examples, so every preference improves
  the next judgement at no extra cost.
- **Best answer first, feedback in one click.** The point is a good answer fast; disagreeing
  costs one click and becomes training data for the judge, instead of a form to fill in.
- **Spending is always previewed and capped.** Every run is estimated before any call and
  refused or stopped at a cap; premium models are opt-in only. The MCP server splits
  `preview_check` and `run_check` into two tools, so an agent cannot spend money before the
  user has seen the cost.
- **Errors are data.** Every call returns a result with an error field instead of raising;
  temporary errors are retried following the provider's `retry-after`, permanent ones are
  not, and cut-off answers are flagged. A busy judge still leaves the user with the answers.
- **One unified call, official SDKs only.** `run_model()` hides the differences between
  providers. No LangChain or LlamaIndex: the primitives stay visible.
- **Online without a database.** Local JSONL files are the source of truth; online, each new
  file is copied to a private GitHub repository, and past picks are downloaded at start.
- **Prompts and rules are data, not code.** They are loaded from files at run time, with
  generic fallbacks in `examples/`, so the mechanism can be shared while personal rubrics
  stay private.

## Quick start

```
git clone https://github.com/reian-dhr/llm-selection-lab.git
cd llm-selection-lab
uv sync                              # install the dependencies
cp .env.example .env                 # then add your API keys (see below)
uv run pytest                        # tests: no network, no cost
uv run streamlit run app/main.py     # the web app, at http://localhost:8501
```

Without the web app: `uv run python scripts/check_task.py "describe the task"`.

Keys: one per compared provider (Anthropic, OpenAI, Mistral; a provider without a key
just shows failed answers), and a free Groq key, which the helper and the judge need.
The Python package is named `model_selection_lab`.

## Use it from Claude (MCP)

Register the server with Claude Code (run from anywhere, with the path to this folder):

```
claude mcp add llm-selection-lab -- uv run --directory /path/to/llm-selection-lab python -m model_selection_lab.mcp_server
```

Then ask, for example, "preview a check for: draft a cold email to a recruiter". Claude
calls `preview_check`, shows the drafted test and its maximum cost, and calls `run_check`
only once you agree. `list_checks` shows past checks. Premium models are never used
from MCP.

## Tests

`uv run pytest` runs about 120 tests in a few seconds. Every function that calls a model
accepts a stand-in, so tests use fakes: no network, no keys, no cost. GitHub Actions runs
them on every push.

## Put it online

The web app runs on Streamlit Community Cloud from a GitHub repository, behind a username and
password (`uv run python scripts/make_password_hash.py`). Secrets to set on the host: the
provider API keys, `username`, `password_hash`, and for the backup `github_token` (write
access to one private data repository) and `data_repo`. See
[docs/how_it_works.md](docs/how_it_works.md), section 5, for both.

## Limits and next steps

- The judge's agreement with human picks is measured but on a small sample so far; a
  proper reliability study (agreement rate, Cohen's kappa, position-bias check) is next.
- The helper and judge run on a free tier with tight rate limits: when busy, the web app
  shows the answers unranked.
- Rules are written by hand. Next: when the judge disagrees with a pick, propose a rule
  change for the user to approve.
- The check flow (command line and MCP) does not judge quality yet: it compares cost
  and speed, and the answers are saved for you to read.
- The sign-in lock counts wrong tries per browser session; it is not a rate limit.
  Each request still has its spending cap.

## Learn more

[docs/how_it_works.md](docs/how_it_works.md) explains the flow, every file and the design
choices. [AGENTS.md](AGENTS.md) has the conventions for coding agents and contributors.

## License

MIT, see [LICENSE](LICENSE).
