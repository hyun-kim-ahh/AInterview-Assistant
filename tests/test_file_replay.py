"""FileReplaySource 계약 확인 테스트 (dev-plan 4단계)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from interview_assistant.contracts import Speaker
from interview_assistant.sources.file_replay import FileReplaySource

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "fake_transcript.jsonl"


def test_replays_all_events_from_fixture(capsys):
    events = list(FileReplaySource(FIXTURE_PATH).stream())

    assert [e.turn_id for e in events] == [f"t{n:03d}" for n in range(1, 11)]
    assert [e.timestamp for e in events] == [
        5.0, 12.0, 25.0, 30.0, 48.0, 55.0, 78.0, 85.0, 110.0, 118.0,
    ]
    assert [e.speaker for e in events] == [
        Speaker.INTERVIEWER, Speaker.EXPERT,
        Speaker.INTERVIEWER, Speaker.EXPERT,
        Speaker.INTERVIEWER, Speaker.EXPERT,
        Speaker.INTERVIEWER, Speaker.EXPERT,
        Speaker.INTERVIEWER, Speaker.EXPERT,
    ]
    assert events[0].text == "로스팅을 언제 멈춰야 할지는 어떻게 판단하세요?"
    assert events[-1].text == (
        "아 신입한테는 오히려 시간표부터 외우게 해요. 감이 없으니까 일단 숫자로 "
        "시작해야죠. 그건 좀 모순 같긴 한데."
    )
    assert capsys.readouterr().out == ""


def test_default_is_not_realtime_and_is_fast():
    start = time.monotonic()
    list(FileReplaySource(FIXTURE_PATH).stream())
    assert time.monotonic() - start < 0.5


def test_realtime_sleeps_and_prints(tmp_path, capsys):
    path = tmp_path / "mini.jsonl"
    path.write_text(
        '{"speaker": "interviewer", "text": "소리가 이상해요", "timestamp": 0.0, "turn_id": "t001"}\n'
        '{"speaker": "expert", "text": "그럴 땐 화력을 낮춰요", "timestamp": 0.05, "turn_id": "t002"}\n',
        encoding="utf-8",
    )

    start = time.monotonic()
    events = list(FileReplaySource(path, realtime=True).stream())
    elapsed = time.monotonic() - start

    assert len(events) == 2
    assert elapsed >= 0.04
    out = capsys.readouterr().out
    assert "t001" in out and "소리가 이상해요" in out
    assert "t002" in out and "그럴 땐 화력을 낮춰요" in out


def test_missing_speaker_key_defaults_to_none(tmp_path):
    path = tmp_path / "mini.jsonl"
    path.write_text(
        '{"text": "화자 정보가 없어요", "timestamp": 1.0, "turn_id": "t001"}\n',
        encoding="utf-8",
    )

    events = list(FileReplaySource(path).stream())

    assert len(events) == 1
    assert events[0].speaker is None


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "mini.jsonl"
    path.write_text(
        '{"speaker": "expert", "text": "첫 문장이에요", "timestamp": 0.0, "turn_id": "t001"}\n'
        "\n"
        '{"speaker": "expert", "text": "둘째 문장이에요", "timestamp": 1.0, "turn_id": "t002"}\n',
        encoding="utf-8",
    )

    events = list(FileReplaySource(path).stream())

    assert [e.text for e in events] == ["첫 문장이에요", "둘째 문장이에요"]


def test_malformed_speaker_value_raises(tmp_path):
    path = tmp_path / "mini.jsonl"
    path.write_text(
        '{"speaker": "narrator", "text": "잘못된 화자 값", "timestamp": 0.0, "turn_id": "t001"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        list(FileReplaySource(path).stream())
