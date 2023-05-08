import secrets
import string


def generate_api_key(length=32):
    characters = string.ascii_letters + string.digits
    api_key = "".join(secrets.choice(characters) for _ in range(length))
    return api_key
