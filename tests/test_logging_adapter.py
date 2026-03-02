from __future__ import annotations

import json
from pathlib import Path

from simpleai.adapters.logging_adapter import PromptLogger


def test_prompt_logger_writes_json_lines(tmp_path: Path) -> None:
    logfile = tmp_path / "simpleai.log"
    logger = PromptLogger(
        {
            "enabled": True,
            "logfile_location": str(logfile),
            "django_logfile": "django",
        }
    )

    event_id = logger.log_start(args={"prompt": "hi"}, adapter_payload={"provider": "openai"})
    logger.log_end(event_id=event_id, started_at=0.0, result_preview="ok", citations_count=1)

    lines = logfile.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    start_payload = json.loads(lines[0])
    end_payload = json.loads(lines[1])

    assert start_payload["event"] == "run_prompt.start"
    assert end_payload["event"] == "run_prompt.end"
    assert end_payload["citations_count"] == 1
