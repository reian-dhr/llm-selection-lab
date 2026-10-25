"""Remote store: a copy of the saved files in a private GitHub repository.

Online hosting (for example Streamlit Community Cloud) wipes local files when the app
restarts. Local files stay the source of truth, and this module keeps a copy on GitHub:
- push_file(): after each request or preference, upload that file (same path in the repo).
- pull_picks(): when the app starts, download the picks and the answers they refer to, so
  the judge still learns from past preferences after a restart.

It uses GitHub's REST API ("contents" endpoints) with a fine-grained token that can write to
that one repository only. Nothing here runs unless a token and a repository are configured.
"""

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
PICKS_FOLDER = "data/picks"
RESULTS_FOLDER = "results"


def github_request(method: str, url: str, token: str, body: dict | None = None):
    """One call to the GitHub API. Returns (status code, parsed JSON or raw text)."""
    request = urllib.request.Request(
        url,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as problem:
        text = problem.read().decode("utf-8", errors="replace")
        status = problem.code
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, text


def contents_url(repo: str, path: str) -> str:
    """The API address of one file in the repository."""
    return f"{API}/repos/{repo}/contents/{Path(path).as_posix()}"


def push_file(path: str, token: str, repo: str, http=github_request) -> None:
    """Upload a local file to the same path in the repository, creating or replacing it.

    http: only for tests, to use a fake instead of the network. Raises RuntimeError.
    """
    url = contents_url(repo, path)
    status, existing = http("GET", url, token)
    body = {
        "message": f"Save {Path(path).name}",
        "content": base64.b64encode(Path(path).read_bytes()).decode(),
    }
    if status == 200:
        body["sha"] = existing["sha"]  # GitHub needs it to replace a file
    status, answer = http("PUT", url, token, body)
    if status not in (200, 201):
        raise RuntimeError(f"GitHub refused to save {path}: {status} {answer}")


def download(path: str, token: str, repo: str, http=github_request) -> bool:
    """Download one file from the repository to the same local path. False if missing."""
    status, answer = http("GET", contents_url(repo, path), token)
    if status != 200:
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(base64.b64decode(answer["content"]))
    return True


def pull_picks(token: str, repo: str, http=github_request) -> int:
    """Download every picks file and the results file of the same request. Returns how many
    picks files were downloaded (0 if the repository is still empty)."""
    status, listing = http("GET", contents_url(repo, PICKS_FOLDER), token)
    if status != 200:
        return 0
    pulled = 0
    for item in listing:
        if item.get("type") != "file" or not item["name"].endswith(".jsonl"):
            continue
        download(f"{PICKS_FOLDER}/{item['name']}", token, repo, http)
        download(f"{RESULTS_FOLDER}/{item['name']}", token, repo, http)
        pulled += 1
    return pulled
