"""Offline checks for the pinned public asset downloader."""

import hashlib
import io
import json

import pytest

from benchmarks import download as assets


def _registry(tmp_path, monkeypatch, *, expected=b"public bytes"):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"data": {"data/sample.bin": {
        "url": "https://example.org/pinned/sample.bin",
        "bytes": len(expected), "sha256": hashlib.sha256(expected).hexdigest(),
    }}, "model": {}}), encoding="utf-8")
    monkeypatch.setattr(assets, "_REGISTRY", registry)
    return registry


def test_download_is_atomic_and_rechecks_existing_hash(tmp_path, monkeypatch):
    payload = b"public bytes"
    _registry(tmp_path, monkeypatch, expected=payload)
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(assets, "urlopen", fake_urlopen)
    destination = tmp_path / "assets"
    assert assets.download(destination, "data") == {"data/sample.bin": "downloaded"}
    assert (destination / "data/sample.bin").read_bytes() == payload
    assert assets.download(destination, "data") == {"data/sample.bin": "verified"}
    assert len(calls) == 1
    (destination / "data/sample.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Existing asset hash or size mismatch"):
        assets.download(destination, "data")
    assert (destination / "data/sample.bin").read_bytes() == b"tampered"


def test_bad_download_never_replaces_target_or_leaves_partial(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch)
    monkeypatch.setattr(assets, "urlopen", lambda url, timeout: io.BytesIO(b"wrong bytes"))
    destination = tmp_path / "assets"
    with pytest.raises(ValueError, match="Downloaded asset hash or size mismatch"):
        assets.download(destination, "data")
    assert not (destination / "data/sample.bin").exists()
    assert list(destination.rglob("*.part-*")) == []


def test_destination_and_registry_paths_are_restricted(tmp_path, monkeypatch):
    registry = _registry(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside the repository"):
        assets.download(assets._REPOSITORY / "output", "data")
    manifest = json.loads(registry.read_text(encoding="utf-8"))
    manifest["data"] = {"../escape.bin": next(iter(manifest["data"].values()))}
    registry.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe asset path"):
        assets.download(tmp_path / "assets", "data")


def test_registry_has_exact_pinned_public_assets():
    registry = json.loads(assets._REGISTRY.read_text(encoding="utf-8"))
    assert set(registry["data"]) == {
        "data/gretel/train.parquet", "data/gretel/validation.parquet", "data/gretel/test.parquet",
        "data/gretel/README.md", "data/tab/train.json", "data/tab/dev.json", "data/tab/test.json",
        "data/tab/LICENSE.txt",
        "data/ag_news/train.parquet", "data/ag_news/test.parquet",
    }
    assert set(registry["model"]) == {
        "models/gliner-pii/README.md", "models/gliner-pii/gliner_config.json",
        "models/gliner-pii/pytorch_model.bin", "models/gliner-pii/encoder/config.json",
        "models/gliner-pii/encoder/tokenizer_config.json", "models/gliner-pii/encoder/spm.model",
        "models/gliner-pii/encoder/README.md",
    }
    for name, spec in registry["data"].items():
        revision = {"gretel": "e06eb1499ca8d54470f085021cd8e54f9efac7fd",
                    "tab": "558e09e26d6b36f5f78440074e6a233946d98bd9",
                    "ag_news": "eb185aade064a813bc0b7f42de02595523103ca4"}[name.split("/")[1]]
        assert revision in spec["url"]
    for name, spec in registry["model"].items():
        revision = ("a0484667b22365f84929a935b5e50a51f71f159d" if "/encoder/" in name
                    else "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d")
        assert revision in spec["url"]
