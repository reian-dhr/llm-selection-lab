"""Building blocks of the web app.

- apply_style(): layout touches on top of the theme in .streamlit/config.toml.
- secret(): read one optional setting from Streamlit secrets or the environment.
- require_login(): asks for the username and password when the app is protected.
- greeting(): "Good afternoon, <name>", the name coming from the optional display_name secret.
- format_cost(): a cost in US dollars, written in cents when it is small.
- as_markdown(), as_html_text(): make model text safe to show (dollar signs, HTML).
"""

import hmac
import html
import os
import re
from datetime import datetime

import streamlit as st

from model_selection_lab.auth import check_password

# A short stylesheet on top of the theme: a narrow reading column, no Streamlit chrome,
# and chat bubbles.
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&display=swap');
/* A narrow centred column, like a chat. */
.block-container {max-width: 760px; padding-top: 2.5rem; padding-bottom: 8rem;}
/* No Streamlit header, toolbar or decoration line. */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stAppDeployButton"] {display: none;}
/* Headings: serif, light, a little tighter. */
h1 {font-family: 'Source Serif 4', Georgia, serif !important; font-weight: 400 !important;
    letter-spacing: -0.01em; text-align: center;}
/* Chat messages: no avatars; the request in a soft bubble, the answer on the page. */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarCustom"] {display: none;}
[data-testid="stChatMessage"] {background: transparent; padding: 0.25rem 0; gap: 0;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: #2E2F34; border-radius: 1rem; padding: 0.75rem 1.1rem; margin-bottom: 1.5rem;
}
/* The input bar: rounded, a shade lighter than the page. */
[data-testid="stChatInput"] {border-radius: 1.25rem; box-shadow: 0 2px 14px rgba(0,0,0,0.35);
  background: #2A2B2F !important;}
[data-testid="stChatInput"] * {background-color: transparent;}
[data-testid="stChatInput"] button {background-color: #5FB3A1 !important; color: #202124;}
[data-testid="stChatInput"] button:disabled {background-color: #3A3B40 !important;}
/* The bar that holds the input: same colour as the page. */
[data-testid="stBottomBlockContainer"] {background: #202124;}
/* The sign-in card: narrow and centred. */
[data-testid="stForm"] {max-width: 420px; margin: 0 auto; background: #2A2B2F;}
/* Wide tables in answers scroll sideways inside their box instead of overflowing it. */
[data-testid="stMarkdownContainer"] table {display: block; max-width: 100%; overflow-x: auto;
  font-size: 0.9rem;}
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {min-width: 8rem;}
/* A thin scrollbar that stays visible, so it is clear when a table has more columns. */
[data-testid="stMarkdownContainer"] table::-webkit-scrollbar {height: 8px;}
[data-testid="stMarkdownContainer"] table::-webkit-scrollbar-thumb {
  background: #4A4B51; border-radius: 4px;
}
[data-testid="stMarkdownContainer"] table::-webkit-scrollbar-track {background: #2A2B2F;}
/* Long unbroken words or links wrap instead of widening the page. */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
  overflow-wrap: anywhere;
}
/* Small muted text under an answer. */
.muted {color: #9C9A94; font-size: 0.85rem;}
.label {color: #5FB3A1; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.02em;}
</style>
"""

MAX_SIGN_IN_TRIES = 5


def apply_style() -> None:
    """Add the stylesheet above to the page."""
    st.markdown(STYLE, unsafe_allow_html=True)


def secret(name: str) -> str:
    """One optional setting: from Streamlit secrets, else the environment variable APP_<NAME>,
    else empty. The prefix avoids clashes with system variables such as USERNAME."""
    try:
        value = st.secrets.get(name, "")
    except Exception:  # no secrets file at all
        value = ""
    return str(value or os.environ.get(f"APP_{name.upper()}", ""))


def require_login() -> None:
    """Stop the page until the right username and password are given, if the app is protected.

    The app is protected when the password_hash secret is set (see docs/how_it_works.md).
    """
    stored_hash = secret("password_hash")
    if not stored_hash or st.session_state.get("signed_in"):
        return
    st.title("Model Selection Lab")
    tries = st.session_state.get("sign_in_tries", 0)
    if tries >= MAX_SIGN_IN_TRIES:
        st.error("Too many wrong tries. Reload the page later.")
        st.stop()
    with st.form("sign_in"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary", use_container_width=True):
            # Both are always checked, and compared in constant time, so a wrong try does
            # not reveal which of the two was wrong.
            right_name = hmac.compare_digest(
                username.strip().encode(), secret("username").encode()
            )
            right_password = check_password(password, stored_hash)
            if right_name and right_password:
                st.session_state.signed_in = True
                st.rerun()
            st.session_state.sign_in_tries = tries + 1
            st.error("Wrong username or password.")
    st.stop()


def greeting(now: datetime | None = None) -> str:
    """'Good morning', 'Good afternoon' or 'Good evening', with the name if one is set."""
    hour = (now or datetime.now()).hour
    part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
    name = secret("display_name")
    return f"Good {part}, {name}" if name else f"Good {part}"


def format_cost(cost_usd: float) -> str:
    """Write a cost as '$1.20' from a dollar, '0.42¢' under it, '0.006¢' under a tenth of a cent."""
    if cost_usd >= 1:
        return f"${cost_usd:.2f}"
    cents = cost_usd * 100
    return f"{cents:.2f}¢" if cents >= 0.1 else f"{cents:.3f}¢"


# Code blocks (```...```) and inline code (`...`): shown as is, so never escaped.
CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)


def as_markdown(text: str) -> str:
    """Model or user text, ready for st.markdown. Streamlit reads text between two dollar
    signs as a maths formula, so "$12 and $30" would lose its dollars: escape them, except
    inside code, where text is shown as is."""
    parts = CODE.split(text)  # odd positions are the code parts
    return "".join(
        part if number % 2 else part.replace("$", "\\$") for number, part in enumerate(parts)
    )


def as_html_text(text: str) -> str:
    """Model text placed inside our own HTML: shown as plain text, never run as HTML. Inside
    HTML a backslash does not escape a dollar sign, so the dollar's HTML code is used."""
    return html.escape(text).replace("$", "&#36;")
