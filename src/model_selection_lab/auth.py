"""Password checking for the online app: stores only a salted hash, never the password.

A hash looks like "pbkdf2_sha256$<iterations>$<salt>$<digest>". It is made once with
scripts/make_password_hash.py and saved in .streamlit/secrets.toml (never committed) or in
the APP_PASSWORD_HASH environment variable on the hosting service.
"""

import hashlib
import hmac
import secrets

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
