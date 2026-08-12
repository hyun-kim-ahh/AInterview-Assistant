"""replay_with_pacing 계약 확인 테스트 (dev-plan 15단계 — FileReplaySource에서 추출)."""

from __future__ import annotations

import time

from interview_assistant.contracts import TranscriptEvent
from interview_assistant.sources.pacing import replay_with_pacing


def _make_events() -> list[TranscriptEvent]:
    return [
        TranscriptEvent(text="첫 번째 발화", timestamp=0.0, turn_id="t001"),
        TranscriptEvent(text="두 번째 발화", timestamp=0.05, turn_id="t002"),
        TranscriptEvent(text="세 번째 발화", timestamp=0.1, turn_id="t003"),
    ]


def test_realtime_false_yields_immediately_without_printing(capsys):
    start = time.monotonic()
    events = list(replay_with_pacing(_make_events(), realtime=False))
    elapsed = time.monotonic() - start

    assert [e.turn_id for e in events] == ["t001", "t002", "t003"]
    assert elapsed < 0.05
    assert capsys.readouterr().out == ""


def test_realtime_true_sleeps_between_events_and_prints(capsys):
    start = time.monotonic()
    events = list(replay_with_pacing(_make_events(), realtime=True))
    elapsed = time.monotonic() - start

    assert [e.turn_id for e in events] == ["t001", "t002", "t003"]
    assert elapsed >= 0.09  # 두 델타(0.05+0.05)만큼은 sleep했어야 함
    out = capsys.readouterr().out
    assert "t001" in out and "t002" in out and "t003" in out


def test_preserves_event_order_and_content():
    events = list(replay_with_pacing(_make_events(), realtime=False))

    assert [e.text for e in events] == ["첫 번째 발화", "두 번째 발화", "세 번째 발화"]


def test_realtime_true_delays_first_event_by_its_own_timestamp():
    """녹음 시작부터 첫 발화 전까지 무음 구간이 있으면(첫 이벤트 timestamp > 0),
    그 구간만큼도 지연돼야 한다 — 그래야 오디오 재생 위치와 어긋나지 않는다."""
    events_with_lead_in_silence = [
        TranscriptEvent(text="첫 번째 발화", timestamp=0.1, turn_id="t001"),
    ]

    start = time.monotonic()
    list(replay_with_pacing(events_with_lead_in_silence, realtime=True))
    elapsed = time.monotonic() - start

    assert elapsed >= 0.09


def test_realtime_true_applies_extra_delay_seconds_before_first_event():
    """delay_seconds는 첫 이벤트의 timestamp가 0이어도(무음 리드인이 없어도) 별도로
    한 번 더 지연을 얹는다 — 오디오 파일 테스트 모드의 "일부러 STT 처리 지연처럼
    보이게" 하는 용도라 첫 이벤트에도 반드시 적용돼야 한다."""
    events = [TranscriptEvent(text="첫 번째 발화", timestamp=0.0, turn_id="t001")]

    start = time.monotonic()
    list(replay_with_pacing(events, realtime=True, delay_seconds=0.1))
    elapsed = time.monotonic() - start

    assert elapsed >= 0.09


def test_delay_seconds_ignored_when_not_realtime():
    """realtime=False(배치 모드)에서는 delay_seconds도 무시돼야 한다 — 화면 재생
    타이밍이 아니라 그냥 즉시 전체를 반환하는 경로이기 때문."""
    events = [TranscriptEvent(text="첫 번째 발화", timestamp=0.0, turn_id="t001")]

    start = time.monotonic()
    list(replay_with_pacing(events, realtime=False, delay_seconds=0.1))
    elapsed = time.monotonic() - start

    assert elapsed < 0.09
