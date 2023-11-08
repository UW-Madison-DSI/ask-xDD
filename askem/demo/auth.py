import os

import bcrypt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def store_demo_auth(password: str) -> None:
    """Securely store a password for the demo app."""

    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode(), salt)

    with open(".env", "a") as f:
        f.write(f"DEMO_SALT={salt.decode()}\n")
        f.write(f"DEMO_HASHED_PASSWORD={hashed_password.decode()}\n")


def authenticate_user(password: str) -> bool:
    """Returns `True` if the user had the correct password."""

    stored_hashed_password = os.getenv("DEMO_HASHED_PASSWORD")
    stored_salt = os.getenv("DEMO_SALT")

    if stored_hashed_password is None:
        raise ValueError("No password set for demo app.")

    if stored_salt is None:
        raise ValueError("No salt set for demo app.")

    hashed_password_attempt = bcrypt.hashpw(password.encode(), stored_salt.encode())

    # Compare the stored hashed password with the hashed password attempt
    if hashed_password_attempt.decode() == stored_hashed_password:
        return True

    return False


def st_check_password() -> bool:
    """Returns `True` if the user had the correct password.

    Usage:
    ```python
    if st_check_password():
        # Protected content
        st.title("You are logged in!")
    ```

    """

    def password_entered():
        """Checks whether a password entered by the user is correct."""

        if authenticate_user(st.session_state["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True
