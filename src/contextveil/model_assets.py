"""Integrity check for the single evaluated offline GLiNER bundle."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import ContextVeilError

GLINER_REFERENCE = {
    "model": {
        "repo": "urchade/gliner_multi_pii-v1",
        "revision": "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d",
        "files": {
            "gliner_config.json": "2088bee85f50849dea2ab4cf411ab5cf3aaaf7c8a45c23412035978b2bc02801",
            "pytorch_model.bin": "3003753fba99e40645cf088c7367a2c6211fc174897dc64f1f9c147c29d18d2d",
        },
    },
    "encoder": {
        "repo": "microsoft/mdeberta-v3-base",
        "revision": "a0484667b22365f84929a935b5e50a51f71f159d",
        "files": {
            "config.json": "bcffcd343dc5efa5ef2d5a58d2b405eed108f01cc45b48d0a907b333ec41801f",
            "tokenizer_config.json": "3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d",
            "spm.model": "13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3",
        },
    },
}


def verify_gliner_bundle(root: str | Path) -> dict:
    """Verify pinned file bytes, then return reference metadata."""
    bundle = Path(root)
    try:
        for part, expected in GLINER_REFERENCE.items():
            directory = bundle if part == "model" else bundle / "encoder"
            for name, expected_hash in expected["files"].items():
                digest = hashlib.sha256()
                with (directory / name).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_hash:
                    raise ValueError
    except (OSError, ValueError):
        raise ContextVeilError("The local GLiNER bundle is missing, altered, or not the pinned revision.") from None
    return {part: {"repo": value["repo"], "revision": value["revision"],
                   "files": dict(value["files"])} for part, value in GLINER_REFERENCE.items()}
