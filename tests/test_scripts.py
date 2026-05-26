from __future__ import annotations

from pathlib import Path

from simpleai.scripts import reasoning_smoke, run_provider_smoke


class _SmokeResult:
    def __init__(self, status: str) -> None:
        self.status = status


def test_run_provider_smoke_main_uses_packaged_entrypoint(monkeypatch) -> None:
    calls: dict[str, object] = {}
    sample_file = Path("/tmp/resume.pdf")

    monkeypatch.setattr(
        run_provider_smoke,
        "resolve_sample_file_path",
        lambda file_arg: sample_file,
    )

    def fake_matrix(**kwargs):
        calls.update(kwargs)
        return [_SmokeResult("success")]

    monkeypatch.setattr(run_provider_smoke, "run_provider_matrix", fake_matrix)

    code = run_provider_smoke.main(
        [
            "--file",
            "resume.pdf",
            "--settings-file",
            "ai_settings.json",
            "--providers",
            "openai",
            "gemini",
            "--no-color",
        ]
    )

    assert code == 0
    assert calls == {
        "file_path": sample_file,
        "settings_file": "ai_settings.json",
        "providers": ["openai", "gemini"],
        "use_color": False,
    }


def test_reasoning_smoke_show_config_uses_packaged_module(monkeypatch, capsys) -> None:
    monkeypatch.setattr(reasoning_smoke, "load_settings", lambda settings_file=None: {})
    monkeypatch.setattr(reasoning_smoke, "_settings_source", lambda settings_file: "test settings")
    monkeypatch.setattr(
        reasoning_smoke,
        "_resolve_smoke_model",
        lambda settings, provider: f"{provider}-model",
    )

    code = reasoning_smoke.main(["--provider", "openai", "--show-config"])

    captured = capsys.readouterr()
    assert code == 0
    assert "simpleai package:" in captured.out
    assert "reasoning_smoke:" in captured.out
    assert "simpleai/scripts/reasoning_smoke.py" in captured.out
    assert "settings source: test settings" in captured.out
    assert "openai: openai-model" in captured.out

