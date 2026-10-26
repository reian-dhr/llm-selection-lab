"""LLM Selection Lab, the web app: one box, every model answers, the best answer first.

Run with: uv run streamlit run app/main.py
Then open http://localhost:8501 in the browser.

The page shows a greeting and an input bar. A request is sent as is to every standard model,
and to the premium ones too when the switch at the top is on (see
src/model_selection_lab/ask.py). The judge's pick is shown as the best answer, and the
other answers are one click away, each with an "I prefer this one" button that teaches the
judge. Only the latest request is shown: each request is independent, and past requests are
saved in the background, not displayed.

When a password hash is set, every visitor must sign in first: see docs/how_it_works.md.
"""

from pathlib import Path

import streamlit as st
from components import (
    apply_style,
    as_html_text,
    as_markdown,
    format_cost,
    greeting,
    require_login,
    secret,
)

from model_selection_lab import remote_store
from model_selection_lab.ask import ask, prefer
from model_selection_lab.judge.judge import verdicts_path_for
from model_selection_lab.models.model_registry import display_name, select_models
from model_selection_lab.settings import MAX_RUN_COST_USD

AGREE_REASON = "Agreed with the judge's pick"


def remote() -> tuple[str, str] | None:
    """The GitHub token and data repository, if the remote copy is configured."""
    token, repo = secret("github_token"), secret("data_repo")
    return (token, repo) if token and repo else None


@st.cache_resource
def pull_past_picks() -> int:
    """Once per app start: bring back past preferences, so the judge still learns from them."""
    if remote() is None:
        return 0
    try:
        return remote_store.pull_picks(*remote())
    except Exception as problem:  # never block the page on the backup
        print(f"Could not download past picks: {problem}")
        return 0


def back_up(*paths: str) -> None:
    """Copy saved files to the private data repository, if configured. Never fails the page."""
    if remote() is None:
        return
    for path in paths:
        if not Path(path).exists():
            continue
        try:
            remote_store.push_file(path, *remote())
        except Exception as problem:
            print(f"Could not back up {path}: {problem}")
            st.toast("Saved on this server, but the online backup failed.")


def premium_switch(disabled: bool) -> bool:
    """The switch for the premium (expensive) models, at the top right of every page."""
    _, right = st.columns([3, 1])
    return right.toggle("Premium models", key="premium", disabled=disabled, help=(
        "Also ask the most capable models. They cost much more per answer; each request "
        f"still stops at ${MAX_RUN_COST_USD:.0f}."
    ))


def show_welcome(include_premium: bool) -> None:
    """The empty page: a greeting and one line on what happens."""
    st.markdown("<div style='height: 10vh'></div>", unsafe_allow_html=True)
    st.title(greeting())
    names = ", ".join(display_name(key) for key in select_models(include_premium))
    which = "Standard and premium models" if include_premium else "Standard models only"
    st.markdown(
        "<p class='muted' style='text-align:center'>Ask anything. "
        f"{names} answer, and a judge picks the best.<br>"
        f"{which}, each request stops at ${MAX_RUN_COST_USD:.0f}.</p>",
        unsafe_allow_html=True,
    )


def answer_details(record: dict) -> str:
    """'0.04¢ · 2.1 s' for one answer."""
    return f"{format_cost(record['cost_usd'])} · {record['latency_s']:.1f} s"


def save_preference(outcome, model_key: str, reason: str, message: str) -> None:
    """Save a preference, remember it for this page, and thank the user."""
    back_up(prefer(outcome, model_key, reason))
    st.session_state.preferred = model_key
    st.toast(message)


def show_best(outcome) -> None:
    """The judge's pick, its reason, and a button to agree with it."""
    record = outcome.answers[outcome.best_model]
    st.markdown(
        f"<span class='label'>★ BEST ANSWER · {display_name(outcome.best_model)}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(as_markdown(record["text"]))
    st.markdown(
        f"<p class='muted'>Why: {as_html_text(outcome.judge_reason)} · "
        f"{answer_details(record)}</p>",
        unsafe_allow_html=True,
    )
    if st.session_state.get("preferred") == outcome.best_model:
        st.caption("✓ You agreed with this pick.")
    elif st.button("👍 Good pick", key=f"agree_{outcome.task_name}"):
        save_preference(outcome, outcome.best_model, AGREE_REASON, "Noted, thank you.")
        st.rerun()


def show_one_alternative(outcome, model_key: str) -> None:
    """One other answer, with its cost and an 'I prefer this one' button."""
    record = outcome.answers[model_key]
    with st.container(border=True):
        st.markdown(f"**{display_name(model_key)}**")
        st.markdown(as_markdown(record["text"]))
        st.markdown(f"<p class='muted'>{answer_details(record)}</p>", unsafe_allow_html=True)
        if st.session_state.get("preferred") == model_key:
            st.caption("✓ You preferred this one. The judge will learn from it.")
            return
        key = f"{outcome.task_name}_{model_key}"
        left, right = st.columns([3, 1], vertical_alignment="center")
        reason = left.text_input(
            "Why (optional)", key=f"reason_{key}", placeholder="Why do you prefer it? (optional)",
            label_visibility="collapsed",
        )
        if right.button("I prefer this one", key=f"prefer_{key}", use_container_width=True):
            save_preference(outcome, model_key, reason, "Thanks: the judge will learn from it.")
            st.rerun()


def show_outcome(outcome) -> None:
    """The request, then the best answer and the others behind an expander."""
    with st.chat_message("user"):
        st.markdown(as_markdown(outcome.request))
    with st.chat_message("assistant"):
        if not outcome.answers:
            st.error("No model could answer. Check the API keys, or try again in a minute.")
            return
        others = [key for key in outcome.answers if key != outcome.best_model]
        if outcome.best_model is not None:
            show_best(outcome)
            label = f"Compare with the other {len(others)} answers"
        else:
            if outcome.judge_error:
                st.info(
                    "The judge could not answer (free tier busy, or GROQ_API_KEY missing). "
                    "Here are all the answers, unranked."
                )
            else:
                st.info("The judge found no answer clearly good. Here are all of them.")
            label = f"See the {len(others)} answers"
        with st.expander(label, expanded=outcome.best_model is None):
            for model_key in others:
                show_one_alternative(outcome, model_key)


def run_request(request: str, include_premium: bool) -> None:
    """Ask every model and keep the outcome for the page, or the problem to show.

    It runs on a page where the input bar and the switch are locked: clicking a widget
    restarts a Streamlit page at once, which would stop a request that is already paid for.
    """
    with st.chat_message("user"):
        st.markdown(as_markdown(request))
    with st.chat_message("assistant"):
        with st.spinner("The models are answering, then the judge reads them..."):
            try:
                outcome = ask(request, include_premium=include_premium)
            except ValueError as problem:
                st.session_state.problem = str(problem)
                outcome = None
            if outcome is not None:
                back_up(outcome.results_path, verdicts_path_for(outcome.task_name))
    st.session_state.outcome = outcome
    del st.session_state.pending
    st.rerun()  # show the result, with the input bar unlocked


st.set_page_config(page_title="LLM Selection Lab", page_icon=":material/balance:")
apply_style()
require_login()  # does nothing unless a password hash is configured
pull_past_picks()  # does nothing unless the remote copy is configured

# A new request is first stored, then run on a fresh page with the input bar locked.
pending = st.session_state.get("pending")
include_premium = premium_switch(disabled=pending is not None)
request = st.chat_input("Ask anything...", disabled=pending is not None)
if request:
    st.session_state.pending = (request, include_premium)
    st.session_state.pop("preferred", None)
    st.rerun()
elif pending is not None:
    run_request(*pending)
else:
    if "problem" in st.session_state:
        st.error(st.session_state.pop("problem"))
    if st.session_state.get("outcome") is None:
        show_welcome(include_premium)
    else:
        show_outcome(st.session_state.outcome)
