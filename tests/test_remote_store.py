"""Tests for remote_store.py, with a fake GitHub kept in a dictionary. No network."""

import base64

import pytest

from model_selection_lab.remote_store import contents_url, pull_picks, push_file

REPO = "someone/data"


class FakeGitHub:
    """Answers GET and PUT on contents URLs like GitHub does, from a dict path -> bytes."""

    def __init__(self, files=None, refuse_writes=False):
        self.files = dict(files or {})
        self.refuse_writes = refuse_writes
        self.puts = []

    def __call__(self, method, url, token, body=None):
        path = url.split("/contents/", 1)[1]
        if method == "PUT":
            self.puts.append(body)
            if self.refuse_writes:
                return 403, {"message": "Resource not accessible"}
            self.files[path] = base64.b64decode(body["content"])
            return 201, {}
        if path in self.files:
            content = base64.b64encode(self.files[path]).decode()
            return 200, {"sha": f"sha-{path}", "content": content}
        listing = [
            {"type": "file", "name": name.split("/")[-1]}
            for name in self.files
            if name.startswith(path + "/")
        ]
        return (200, listing) if listing else (404, {"message": "Not Found"})


def test_push_creates_a_new_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "ask_1.jsonl").write_text('{"a": 1}\n')
    github = FakeGitHub()

    push_file("results/ask_1.jsonl", "token", REPO, http=github)

    assert github.files["results/ask_1.jsonl"] == b'{"a": 1}\n'
    assert "sha" not in github.puts[0]


def test_push_replaces_an_existing_file_with_its_sha(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "picks.jsonl").write_text("new\n")
    github = FakeGitHub({"picks.jsonl": b"old\n"})

    push_file("picks.jsonl", "token", REPO, http=github)

    assert github.files["picks.jsonl"] == b"new\n"
    assert github.puts[0]["sha"] == "sha-picks.jsonl"


def test_push_raises_when_github_refuses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "picks.jsonl").write_text("new\n")

    with pytest.raises(RuntimeError, match="403"):
        push_file("picks.jsonl", "token", REPO, http=FakeGitHub(refuse_writes=True))


def test_pull_downloads_picks_and_their_answers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    github = FakeGitHub({
        "data/picks/ask_1.jsonl": b"pick\n",
        "results/ask_1.jsonl": b"answers\n",
        "results/ask_2.jsonl": b"answers without a pick\n",
    })

    pulled = pull_picks("token", REPO, http=github)

    assert pulled == 1
    assert (tmp_path / "data/picks/ask_1.jsonl").read_text() == "pick\n"
    assert (tmp_path / "results/ask_1.jsonl").read_text() == "answers\n"
    assert not (tmp_path / "results/ask_2.jsonl").exists()  # not needed by the judge


def test_pull_from_an_empty_repository_does_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert pull_picks("token", REPO, http=FakeGitHub()) == 0


def test_contents_url():
    assert contents_url(REPO, "data/picks/x.jsonl") == (
        "https://api.github.com/repos/someone/data/contents/data/picks/x.jsonl"
    )
