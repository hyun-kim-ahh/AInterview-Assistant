"""SessionLogWriter 계약 확인 테스트 (dev-plan 5단계)."""

from __future__ import annotations

import json

from interview_assistant.contracts import Speaker, TranscriptEvent
from interview_assistant.sources.file_replay import FileReplaySource
from interview_assistant.storage.session_log import SessionLogWriter


def test_appends_event_as_jsonl_line(tmp_path):
    path = tmp_path / "session.jsonl"
    event = TranscriptEvent(
        text="로스팅은 언제 멈추세요", timestamp=5.0, turn_id="t001", speaker=Speaker.INTERVIEWER
    )

    SessionLogWriter(path).append(event)

    raw = path.read_text(encoding="utf-8")
    assert "로스팅은 언제 멈추세요" in raw  # ensure_ascii=False 회귀 방지
    lines = raw.splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {
        "speaker": "interviewer",
        "text": "로스팅은 언제 멈추세요",
        "timestamp": 5.0,
        "turn_id": "t001",
        "corrected": False,
    }


def test_appends_multiple_events_in_order(tmp_path):
    path = tmp_path / "session.jsonl"
    events = [
        TranscriptEvent(text="소리보다 향이 먼저예요", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT),
        TranscriptEvent(text="타이머는 안 보세요?", timestamp=2.0, turn_id="t002", speaker=Speaker.INTERVIEWER),
        TranscriptEvent(text="참고만 해요", timestamp=3.0, turn_id="t003", speaker=Speaker.EXPERT),
    ]
    writer = SessionLogWriter(path)
    for e in events:
        writer.append(e)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["text"] for line in lines] == [e.text for e in events]
    assert [json.loads(line)["turn_id"] for line in lines] == [e.turn_id for e in events]


def test_speaker_none_serializes_as_null(tmp_path):
    path = tmp_path / "session.jsonl"
    event = TranscriptEvent(text="화자 정보가 없어요", timestamp=1.0, turn_id="t001")

    SessionLogWriter(path).append(event)

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["speaker"] is None


def test_creates_parent_directory_if_missing(tmp_path):
    path = tmp_path / "nested" / "deeper" / "session.jsonl"
    event = TranscriptEvent(text="폴더가 없어도 저장돼요", timestamp=1.0, turn_id="t001")

    SessionLogWriter(path).append(event)

    assert path.exists()
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["text"] == "폴더가 없어도 저장돼요"


def test_new_writer_instance_continues_appending(tmp_path):
    path = tmp_path / "session.jsonl"
    event_a = TranscriptEvent(text="첫 번째 발화입니다", timestamp=1.0, turn_id="t001")
    event_b = TranscriptEvent(text="재시작 후 발화입니다", timestamp=2.0, turn_id="t002")

    writer1 = SessionLogWriter(path)
    writer1.append(event_a)
    del writer1  # 프로세스 재시작 시뮬레이션 — close() 호출 불필요

    writer2 = SessionLogWriter(path)
    writer2.append(event_b)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["text"] for line in lines] == [event_a.text, event_b.text]


def test_written_log_is_replayable_by_file_replay_source(tmp_path):
    path = tmp_path / "session.jsonl"
    written_events = [
        TranscriptEvent(text="색 균일도랑 크기요", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT),
        TranscriptEvent(text="생두를 고를 때는 뭘 보세요?", timestamp=2.0, turn_id="t002", speaker=Speaker.INTERVIEWER),
    ]
    writer = SessionLogWriter(path)
    for e in written_events:
        writer.append(e)

    replayed_events = list(FileReplaySource(path).stream())

    assert replayed_events == written_events


def test_export_corrected_snapshot_overwrites_and_preserves_corrected_flag(tmp_path):
    path = tmp_path / "session.jsonl"
    writer = SessionLogWriter(path)
    writer.append(TranscriptEvent(text="원문", timestamp=1.0, turn_id="t001"))

    writer.export_corrected_snapshot(
        [TranscriptEvent(text="교정된 문장", timestamp=1.0, turn_id="t001", corrected=True)]
    )

    snapshot_path = tmp_path / "transcript_corrected.json"
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert data == [
        {"speaker": None, "text": "교정된 문장", "timestamp": 1.0, "turn_id": "t001", "corrected": True}
    ]
    # 원본 append-only 로그는 손대지 않는다
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["text"] == "원문"
