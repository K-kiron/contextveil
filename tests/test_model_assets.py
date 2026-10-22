"""Tiny synthetic files exercise the pinned bundle integrity check."""

import hashlib

import pytest

from contextveil import model_assets
from contextveil.models import ContextVeilError


def _bundle(tmp_path, monkeypatch):
    blobs = {"model": {"gliner_config.json": b"config", "pytorch_model.bin": b"weights"},
             "encoder": {"config.json": b"encoder", "tokenizer_config.json": b"tokenizer",
                         "spm.model": b"sentencepiece"}}
    reference = {
        part: {"repo": part + "/example", "revision": "a" * 40,
               "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
        for part, files in blobs.items()
    }
    monkeypatch.setattr(model_assets, "GLINER_REFERENCE", reference)
    for part, files in blobs.items():
        directory = tmp_path if part == "model" else tmp_path / "encoder"
        directory.mkdir(exist_ok=True)
        for name, data in files.items():
            (directory / name).write_bytes(data)
    return reference


def test_bundle_without_manifests_verifies_bytes_and_exposes_copy_of_reference(tmp_path, monkeypatch):
    reference = _bundle(tmp_path, monkeypatch)
    metadata = model_assets.verify_gliner_bundle(tmp_path)
    assert metadata == reference
    metadata["model"]["files"]["pytorch_model.bin"] = "changed"
    assert model_assets.GLINER_REFERENCE == reference


def test_corrupted_weights_fail_even_when_manifest_matches(tmp_path, monkeypatch):
    _bundle(tmp_path, monkeypatch)
    (tmp_path / "pytorch_model.bin").write_bytes(b"corrupt")
    with pytest.raises(ContextVeilError, match="missing, altered, or not the pinned revision") as error:
        model_assets.verify_gliner_bundle(tmp_path)
    assert str(tmp_path) not in str(error.value)


def test_missing_encoder_file_fails(tmp_path, monkeypatch):
    _bundle(tmp_path, monkeypatch)
    (tmp_path / "encoder/spm.model").unlink()
    with pytest.raises(ContextVeilError):
        model_assets.verify_gliner_bundle(tmp_path)
