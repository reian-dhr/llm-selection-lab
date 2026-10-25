"""Create the username and password hash that protect the online app.

Run with: uv run python scripts/make_password_hash.py
It asks for a username and a password (not shown as you type) and prints two lines to paste
into .streamlit/secrets.toml or the hosting service's secrets settings.
"""

from getpass import getpass

from model_selection_lab.auth import make_hash

if __name__ == "__main__":
    username = input("Username for the app: ").strip()
    if not username:
        raise SystemExit("The username cannot be empty.")
    password = getpass("Password for the app: ")
    if len(password) < 12:
        raise SystemExit("Use at least 12 characters.")
    if getpass("Same password again: ") != password:
        raise SystemExit("The two passwords differ.")
    print(f'\nusername = "{username}"\npassword_hash = "{make_hash(password)}"')
