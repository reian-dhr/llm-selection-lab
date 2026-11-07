"""Tests for auth.py: the password hash."""

from model_selection_lab.auth import SignInGuard, check_password, make_hash


def test_the_right_password_matches_and_a_wrong_one_does_not():
    stored = make_hash("correct horse battery")

    assert check_password("correct horse battery", stored)
    assert not check_password("wrong password", stored)


def test_the_hash_holds_no_password_and_is_salted():
    first, second = make_hash("same password"), make_hash("same password")

    assert "same password" not in first
    assert first != second  # a new salt each time


def test_a_malformed_hash_never_matches():
    assert not check_password("anything", "not-a-hash")


class Clock:
    """A clock the test moves by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_the_guard_locks_after_too_many_failures_and_unlocks_later():
    clock = Clock()
    guard = SignInGuard(max_failures=3, lock_s=60, clock=clock)

    for _ in range(2):
        guard.record_failure()
    assert not guard.locked()
    guard.record_failure()
    assert guard.locked()

    clock.now += 61
    assert not guard.locked()


def test_a_success_clears_the_failures():
    guard = SignInGuard(max_failures=2, lock_s=60, clock=Clock())
    guard.record_failure()

    guard.record_success()
    guard.record_failure()

    assert not guard.locked()
