"""Tests for auth.py: the password hash."""

from model_selection_lab.auth import check_password, make_hash


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
