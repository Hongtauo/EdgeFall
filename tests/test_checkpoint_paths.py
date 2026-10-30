from pathlib import Path

import pytest

from edgefall_st.evaluation.checkpoint import resolve_checkpoint_dependency


def test_checkpoint_dependency_keeps_repository_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dependency = tmp_path / "artifacts/geometry/best.pt"
    dependency.parent.mkdir(parents=True)
    dependency.touch()
    checkpoint = tmp_path / "artifacts/confirmer/epoch_008.pt"

    assert resolve_checkpoint_dependency(
        "artifacts/geometry/best.pt", checkpoint
    ) == dependency.resolve()


def test_checkpoint_dependency_loads_portable_sibling(tmp_path, monkeypatch):
    working_directory = tmp_path / "elsewhere"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    models = tmp_path / "models"
    models.mkdir()
    dependency = models / "geometry_v33g_inference.pt"
    dependency.touch()

    assert resolve_checkpoint_dependency(
        "geometry_v33g_inference.pt", models / "hn3br_epoch008_inference.pt"
    ) == dependency.resolve()


def test_checkpoint_dependency_prefers_existing_working_directory_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    direct = tmp_path / "geometry.pt"
    direct.touch()
    models = tmp_path / "models"
    models.mkdir()
    (models / "geometry.pt").touch()

    assert resolve_checkpoint_dependency("geometry.pt", models / "confirmer.pt") == direct.resolve()


def test_checkpoint_dependency_accepts_absolute_path(tmp_path):
    dependency = tmp_path / "geometry.pt"
    dependency.touch()

    assert resolve_checkpoint_dependency(dependency, tmp_path / "confirmer.pt") == dependency.resolve()


def test_checkpoint_dependency_reports_all_missing_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "models/confirmer.pt"

    with pytest.raises(FileNotFoundError) as error:
        resolve_checkpoint_dependency("missing.pt", checkpoint)

    assert str(tmp_path / "missing.pt") in str(error.value)
    assert str(tmp_path / "models/missing.pt") in str(error.value)
    assert str(checkpoint) in str(error.value)
