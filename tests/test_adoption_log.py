"""AdoptionLogWriter 계약 확인 테스트 (dev-plan 14단계, F11)."""

from __future__ import annotations

import json

from interview_assistant.contracts import AdoptionAction, AdoptionEvent, QuestionType
from interview_assistant.storage.adoption_log import AdoptionLogWriter, read_adoption_log


def test_appends_adopted_event_as_jsonl_line(tmp_path):
    path = tmp_path / "adoptions.jsonl"
    event = AdoptionEvent(
        generated_at="t002",
        question_type=QuestionType.PROBE,
        question_text="그 단내는 언제 확 올라오나요?",
        action=AdoptionAction.ADOPTED,
        target_item="item_01",
    )

    AdoptionLogWriter(path).append(event)

    raw = path.read_text(encoding="utf-8")
    assert "그 단내는 언제 확 올라오나요?" in raw  # ensure_ascii=False 회귀 방지
    lines = raw.splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {
        "generated_at": "t002",
        "question_type": "probe",
        "question_text": "그 단내는 언제 확 올라오나요?",
        "action": "adopted",
        "edited_text": "",
        "target_item": "item_01",
    }


def test_appends_edited_event_with_edited_text(tmp_path):
    path = tmp_path / "adoptions.jsonl"
    event = AdoptionEvent(
        generated_at="t004",
        question_type=QuestionType.GAP,
        question_text="원래 추천 텍스트",
        action=AdoptionAction.EDITED,
        edited_text="실제로 물어본 수정된 텍스트",
        target_item="item_02",
    )

    AdoptionLogWriter(path).append(event)

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["action"] == "edited"
    assert record["edited_text"] == "실제로 물어본 수정된 텍스트"
    assert record["question_text"] == "원래 추천 텍스트"


def test_target_item_none_serializes_as_null(tmp_path):
    path = tmp_path / "adoptions.jsonl"
    event = AdoptionEvent(
        generated_at="t006",
        question_type=QuestionType.EXPAND,
        question_text="관련 없는 질문",
        action=AdoptionAction.DISMISSED,
    )

    AdoptionLogWriter(path).append(event)

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["target_item"] is None


def test_appends_multiple_events_in_order(tmp_path):
    path = tmp_path / "adoptions.jsonl"
    events = [
        AdoptionEvent(
            generated_at="t002",
            question_type=QuestionType.PROBE,
            question_text="첫 번째 질문",
            action=AdoptionAction.ADOPTED,
        ),
        AdoptionEvent(
            generated_at="t004",
            question_type=QuestionType.CONTRADICTION,
            question_text="두 번째 질문",
            action=AdoptionAction.DISMISSED,
        ),
    ]
    writer = AdoptionLogWriter(path)
    for e in events:
        writer.append(e)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["question_text"] for line in lines] == [
        e.question_text for e in events
    ]


def test_creates_parent_directory_if_missing(tmp_path):
    path = tmp_path / "nested" / "deeper" / "adoptions.jsonl"
    event = AdoptionEvent(
        generated_at="t001",
        question_type=QuestionType.PROBE,
        question_text="폴더가 없어도 저장돼요",
        action=AdoptionAction.ADOPTED,
    )

    AdoptionLogWriter(path).append(event)

    assert path.exists()
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["question_text"] == "폴더가 없어도 저장돼요"


def test_read_adoption_log_round_trips_written_events(tmp_path):
    path = tmp_path / "adoptions.jsonl"
    events = [
        AdoptionEvent(
            generated_at="t002",
            question_type=QuestionType.PROBE,
            question_text="원래 질문",
            action=AdoptionAction.ADOPTED,
            target_item="item_01",
        ),
        AdoptionEvent(
            generated_at="t004",
            question_type=QuestionType.GAP,
            question_text="원래 질문 2",
            action=AdoptionAction.EDITED,
            edited_text="수정된 질문 2",
        ),
    ]
    writer = AdoptionLogWriter(path)
    for e in events:
        writer.append(e)

    loaded = read_adoption_log(path)

    assert loaded == events


def test_read_adoption_log_missing_file_returns_empty_list(tmp_path):
    assert read_adoption_log(tmp_path / "없음.jsonl") == []
