"""Wire format shared by the Colab server and the client.

Every request and response body is a Fernet token wrapping JSON. Fernet gives
confidentiality and integrity (AES-128-CBC + HMAC-SHA256), so whoever relays the
traffic (Cloudflare) sees only ciphertext and cannot forge or alter a request. The
key is the only credential: there is no separate bearer token to leak.

Fernet tokens carry their creation time. The server rejects requests older than
MAX_AGE_S and remembers every token it accepted within that window, so a captured
request cannot be replayed.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

from cryptography.fernet import Fernet, InvalidToken

SCHEME = "colab-bridge"
MAX_AGE_S = 120
# Upload/download payloads travel base64-encoded inside one JSON body.
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_BODY_BYTES = MAX_FILE_BYTES * 2


class BridgeError(Exception):
    """A request failed or could not be authenticated."""


def new_key() -> str:
    return Fernet.generate_key().decode()


def seal(key: str, payload: dict[str, Any]) -> bytes:
    return Fernet(key.encode()).encrypt(json.dumps(payload).encode())


def open_sealed(key: str, token: bytes, max_age_s: int | None = MAX_AGE_S) -> dict[str, Any]:
    try:
        plain = Fernet(key.encode()).decrypt(token, ttl=max_age_s)
    except InvalidToken as e:
        raise BridgeError("bad or expired message") from e
    data = json.loads(plain)
    if not isinstance(data, dict):
        raise BridgeError("message is not an object")
    return data


def make_uri(key: str, host: str) -> str:
    return f"{SCHEME}://{key}@{host}"


def parse_uri(uri: str) -> tuple[str, str]:
    """Split `colab-bridge://KEY@host` into (https base url, key)."""
    parts = urlsplit(uri.strip())
    if parts.scheme != SCHEME or not parts.username or not parts.hostname:
        raise BridgeError(f"connection string must look like {SCHEME}://KEY@host")
    key = parts.username
    try:
        Fernet(key.encode())
    except ValueError as e:
        raise BridgeError("connection string has an invalid key") from e
    return f"https://{parts.hostname}", key
