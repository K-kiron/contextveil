"""Authenticated recovery of the normalized text, never the source container."""

import hashlib
import hmac
import json
import os
import re

from .models import ContextVeilError

MAGIC = b"CONTEXTVEIL\x01"
MAX_VAULT_BYTES = 128 * 1024 * 1024


def _crypto(passphrase: str, salt: bytes):
    if not isinstance(passphrase, str) or len(passphrase) < 16:
        raise ContextVeilError("Use a recovery passphrase of at least 16 characters.")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    except ImportError:
        raise ContextVeilError("Encrypted recovery requires the vault extra (cryptography).") from None
    key = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(passphrase.encode("utf-8"))
    return AESGCM(key)


def _document_aad(salt: bytes, document_id: str) -> bytes:
    if not isinstance(document_id, str) or not re.fullmatch(r"[0-9a-f]{32}", document_id):
        raise ContextVeilError("A valid document_id from the matching report is required for recovery.")
    return MAGIC + salt + document_id.encode("ascii")


def seal(original: str, output: str, passphrase: str, *, document_id: str) -> bytes:
    salt, nonce = os.urandom(16), os.urandom(12)
    # Full normalized text preserves alias spelling and structured scalar types on recovery.
    payload = json.dumps({"original": original, "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest()},
                         ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_VAULT_BYTES - 64:
        raise ContextVeilError("Recovery data exceeds the size limit.")
    return MAGIC + salt + nonce + _crypto(passphrase, salt).encrypt(nonce, payload, _document_aad(salt, document_id))


def restore(output: str, vault: bytes, passphrase: str, *, document_id: str) -> str:
    if not isinstance(vault, bytes) or not len(MAGIC) + 44 <= len(vault) <= MAX_VAULT_BYTES or not vault.startswith(MAGIC):
        raise ContextVeilError("Invalid recovery vault.")
    salt = vault[len(MAGIC):len(MAGIC) + 16]
    nonce = vault[len(MAGIC) + 16:len(MAGIC) + 28]
    cipher = _crypto(passphrase, salt)
    try:
        from cryptography.exceptions import InvalidTag
        payload = json.loads(cipher.decrypt(nonce, vault[len(MAGIC) + 28:], _document_aad(salt, document_id)))
        expected = hashlib.sha256(output.encode("utf-8")).hexdigest()
        if (not isinstance(payload, dict) or set(payload) != {"original", "output_sha256"}
                or not isinstance(payload["original"], str) or not isinstance(payload["output_sha256"], str)
                or not hmac.compare_digest(payload["output_sha256"], expected)):
            raise ValueError
        return payload["original"]
    except (InvalidTag, ValueError, TypeError, KeyError, UnicodeError):
        raise ContextVeilError("Recovery authentication failed or the output has changed.") from None
