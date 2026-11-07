"""Password checking for the online app: stores only a salted hash, never the password.

A hash looks like "pbkdf2_sha256$<iterations>$<salt>$<digest>". It is made once with
scripts/make_password_hash.py and saved in .streamlit/secrets.toml (never committed) or in
the APP_PASSWORD_HASH environment variable on the hosting service.
"""

import hashlib
import hmac
import secrets
import time

from model_selection_lab.settings import SIGN_IN_LOCK_S, SIGN_IN_MAX_FAILURES

ITERATIONS = 200_000


def make_hash(password: str, salt: str | None = None) -> str:
    """Return the salted hash of a password, in the format above."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt}${digest.hex()}"


def check_password(password: str, stored_hash: str) -> bool:
    """True if the password matches the stored hash. Compares in constant time."""
    try:
        _, iterations, salt, expected = stored_hash.split("$")
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
    return hmac.compare_digest(digest.hex(), expected)


class SignInGuard:
    """Counts wrong sign-in tries for the whole running app, not per browser session, so a
    script that opens a fresh session for every try is stopped too. After max_failures wrong
    tries the app is locked until lock_s seconds have passed since the last one."""

    def __init__(
        self, max_failures=SIGN_IN_MAX_FAILURES, lock_s=SIGN_IN_LOCK_S, clock=time.monotonic
    ):
        self.max_failures = max_failures
        self.lock_s = lock_s
        self.clock = clock
        self.failures = 0
        self.last_failure = 0.0

    def locked(self) -> bool:
        """True while too many wrong tries are recent."""
        if self.failures and self.clock() - self.last_failure > self.lock_s:
            self.failures = 0  # the lock has run out
        return self.failures >= self.max_failures

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure = self.clock()

    def record_success(self) -> None:
        self.failures = 0
