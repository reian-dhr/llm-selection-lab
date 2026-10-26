# How it works

A plain-language guide to the flow, the files, and the design choices.

## 1. The flow

The web app answers one real request:

```
request --> every standard model answers --> judge picks the best --> best answer shown
(one box)   (cost estimated first, capped)   (sees A to D, no names)   others one click away
                                                                            |
                         "I prefer this one" or "Good pick" (optional) <----+
                                    |
                                    +--> saved as a pick: an example for the judge next time
```

The command line (`scripts/check_task.py`) and the MCP server (`mcp_server.py`) run a fuller
check of a kind of task: a helper drafts a test with three inputs (one with a trap), the
models answer each, and a summary compares cost and speed. Both flows reuse the same runner
and storage; only the web app runs the judge so far, so a check's answers are saved in
`results/` for you to read.

## 2. The web app

`uv run streamlit run app/main.py`, then http://localhost:8501.

One page, like a chat: a greeting, an input bar, and a switch at the top to add the premium
(expensive) models. A request goes as is to every compared model, the judge's pick appears as the best answer with its reason, cost and time, and the
other answers sit behind "Compare with the other answers", each with an "I prefer this one"
button. Only the latest request is shown; every request is saved in the background.

## 3. Files

```
app/                     the web app
  main.py                the page: input bar, best answer, other answers, preferences, backup
  components.py          style, login, greeting, cost formatting
src/model_selection_lab/
  ask.py                 one request on every standard model, then the judge's pick
  check.py               the fuller check of a kind of task; past checks listed and reloaded
  mcp_server.py          the check as MCP tools: preview (free), run (capped), list
  remote_store.py        a copy of saved files in a private GitHub repository (online only)
  settings.py            every default value, in one place
  auth.py                password hashing for the login
  text_files.py          reads a private file, or the generic example if it is missing
  runs/
    task_drafter.py      the helper turns a description into a task
    task_loader.py       reads and checks a task file
    batch_runner.py      every input on every model, cost estimate, spending cap, reuse
    results_storage.py   one saved line per answer; finds answers that can be reused
  models/
    model_registry.py    every model: provider, ID, prices, tier, display name; cost formula
    model_client.py      run_model(): one call to any provider
    retry.py             retries on temporary errors, following the provider's retry-after
    call_result.py       RunResult: what one call returns
  judge/
    picks.py             picks (the preferred answer and why), saved per request or check
    judge.py             the judge: prompt, verdicts, agreement with the picks
  report/
    summary.py           one row per model; recommendation on cost and speed
    job_cost.py          estimated cost of one larger, real job per model
scripts/
  check_task.py          a whole check from the terminal
  run_task.py            run a hand-written task file
  estimate_job_cost.py   cost of a larger job, from a run's results
  make_password_hash.py  create the login's username and password hash
examples/                generic prompts and an example task, used when private ones are missing
tests/                   one test file per module, fakes only: no network, no cost
```

Created locally when the tool runs, not part of the repository: `data/` (picks, verdicts,
your own judge rules and prompts), `tasks/` and `results/`. `AGENTS.md` holds the
instructions for coding agents.

## 4. Design choices

**A test from one sentence.** The helper model writes what it understood, open questions,
facts it is missing, what a good answer looks like, the instruction, and three inputs (one
hard, with a trap). The same checks as for hand-written tasks apply; an invalid answer is
retried once, with the problem explained. It answers in JSON, which quotes every text.

**Helper and judge from outside the comparison.** Both are Qwen 3.8 (Alibaba, open weights)
on Groq's free tier, with the tier `helper`, so they never compete and cannot favour their
own family. The free tier is tight (about 8,000 tokens a minute): calls are kept short, and
retries wait as long as the provider asks, up to 60 seconds.

**How the judge learns.** It does not train. For each input it gets the task, the user's
rules (`data/judge/*.md`: a file named `slide_*.md` applies to slide tasks, `general_*.md`
to all), the 3 most recent picks with their reasons as examples, and the answers A to D. It
writes a critique per answer, then a verdict. Each new pick becomes an example next time.
In the web app its pick is shown first, so a preference is given after seeing it: the
agreement between judge and user can be anchored by the judge, and should be read with
that in mind.

**Best answer first, feedback in one click.** The web app shows the judge's pick first,
because the point is to get a good answer quickly. Disagreeing costs one click ("I prefer
this one", with an optional reason), and that click is saved as a pick, so the judge learns
from it. The judge still sees the answers labelled A to D in a shuffled order, without model
names; its letters are turned back into names for display.

**Online without a database.** The online host wipes local files on restart. Local files
stay the source of truth, and `remote_store.py` copies each new file to a private GitHub
repository; at start, past picks and their answers are downloaded again for the judge.

**Budget guards.** Standard models by default, premium ones on request; a maximum cost
shown before every run; a spending cap per run; answers reused for the same input, model and
exact prompt (for 30 days, since models change).

**Robust calls.** Errors become data, not crashes. Temporary errors (429, 5xx, timeouts)
are retried; permanent ones (400, 401, 404) are not. Cut-off answers are flagged. Latency
counts the last attempt only.

**Model facts in one registry.** IDs, prices with their source and date, tiers, quirks
(for example which word turns thinking off). Premium models cannot switch thinking off, so
they get extra room and their thinking is billed as output.

**Tests without network or cost.** Functions accept a stand-in for the model call
(dependency injection); tests pass fakes.

## 5. Login (for an online version)

Without configuration the app is open, which is fine on your own computer. To protect it:

1. `uv run python scripts/make_password_hash.py`, then type a username and a password
   (12 characters or more).
2. Paste the two printed lines (`username` and `password_hash`) into
   `.streamlit/secrets.toml` (never committed) or the hosting service's secrets settings.
   Environment variables work too: `APP_USERNAME` and `APP_PASSWORD_HASH`.

Only a salted hash of the password is stored. Both fields are always checked and compared in
constant time. Five wrong tries block the form for that browser session; reloading the page
starts a new session, so this slows guessing down but is not a rate limit. Each request
still has its spending cap. An optional `display_name` secret adds a name to the greeting.

**Backup (optional).** Online hosts wipe local files on restart. To keep requests and
picks, create a private GitHub repository for the data and a fine-grained token with
"Contents: read and write" on that repository only, then add two secrets: `github_token`
(the token) and `data_repo` (`owner/name`). Without them nothing is copied.

## 6. Adding things

- **A model:** an entry in `MODELS` (`model_registry.py`), with ID and prices from the
  official docs, `docs_url` and `date_checked`. Expensive: `tier="premium"`. Cannot switch
  thinking off: `reasoning=True`.
- **A provider:** a `call_<provider>` function and one line in `PROVIDER_FUNCTIONS`
  (`model_client.py`), its errors in `retry.py`, a fake in `tests/fakes.py`, and tests.
- **Rules for the judge:** a Markdown file in `data/judge/`, named after the word that
  identifies the kind of task.

## 7. Open items

- The helper and judge run on a preview model on a free tier: either may change. Switch
  `TASK_DRAFTER_MODEL` and `JUDGE_MODEL` in `settings.py` if needed.
- Later: daily spending cap, rate limits, judge notes ("the judge was wrong because"),
  agreement over time, Google login, the job cost in the app.
- The check flow (command line and MCP) does not run the judge yet.
- Refusals by a model are saved as empty answers, without a warning.
