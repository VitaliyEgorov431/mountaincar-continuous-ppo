import importlib

import pytest


def test_interrupted_rerun_invalidates_completion_marker(tmp_path, monkeypatch):
    training = importlib.import_module("mountaincar.train")
    marker = tmp_path / "run_summary.json"
    marker.write_text('{"steps": 1000000}', encoding="utf-8")

    def fail_make(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(training.gym, "make", fail_make)
    with pytest.raises(RuntimeError, match="interrupted"):
        training.train({"num_envs": 1, "total_timesteps": 1}, 0, tmp_path)
    assert not marker.exists()
