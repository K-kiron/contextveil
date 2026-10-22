"""Download pinned public benchmark assets into an explicitly external directory.

Usage: python -m benchmarks.download --dest PATH {all,data,model}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

_REGISTRY = Path(__file__).with_name("assets.json")
_REPOSITORY = Path(__file__).resolve().parents[1]


def _asset_path(destination: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (not re.fullmatch(r"[A-Za-z0-9._/-]+", name) or relative.is_absolute()
            or not relative.parts or any(part in {".", ".."} for part in relative.parts)):
        raise ValueError("Unsafe asset path in registry")
    target = destination.joinpath(*relative.parts)
    if destination not in target.resolve().parents:
        raise ValueError("Asset path escapes destination")
    return target


def _verify(path: Path, expected_hash: str, expected_bytes: int | None) -> bool:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest() == expected_hash and (expected_bytes is None or size == expected_bytes)


def _download_asset(destination: Path, name: str, spec: dict) -> str:
    target = _asset_path(destination, name)
    url, expected_hash = spec["url"], spec["sha256"]
    expected_bytes = spec.get("bytes")
    if (not isinstance(url, str) or not url.startswith("https://")
            or not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            or (expected_bytes is not None and (type(expected_bytes) is not int or expected_bytes < 0))):
        raise ValueError("Invalid asset specification")
    if target.exists():
        if not target.is_file() or not _verify(target, expected_hash, expected_bytes):
            raise ValueError(f"Existing asset hash or size mismatch: {name}")
        return "verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".part-", delete=False) as stream:
            partial = Path(stream.name)
            with urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
        if not _verify(partial, expected_hash, expected_bytes):
            raise ValueError(f"Downloaded asset hash or size mismatch: {name}")
        os.replace(partial, target)
        partial = None
        return "downloaded"
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def download(destination: str | Path, selection: str = "all") -> dict[str, str]:
    """Verify existing files and fetch missing assets; never write inside this repository."""
    if selection not in {"all", "data", "model"}:
        raise ValueError("Selection must be all, data, or model")
    destination = Path(destination).resolve()
    if destination == _REPOSITORY or _REPOSITORY in destination.parents:
        raise ValueError("Destination must be outside the repository")
    registry = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    groups = ("data", "model") if selection == "all" else (selection,)
    result = {}
    for group in groups:
        for name, spec in registry[group].items():
            result[name] = _download_asset(destination, name, spec)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True, help="Asset directory outside this repository")
    parser.add_argument("selection", choices=("all", "data", "model"))
    args = parser.parse_args()
    for name, status in download(args.dest, args.selection).items():
        print(f"{status}: {name}")


if __name__ == "__main__":
    main()
