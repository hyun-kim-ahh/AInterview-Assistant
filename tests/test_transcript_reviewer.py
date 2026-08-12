"""transcript_reviewer 계약 확인 테스트.

다른 LLM 호출 모듈들과 마찬가지로 실제 OpenRouter 호출 없이 openai 클라이언트
모양을 흉내낸 가짜 client를 주입해 파싱/필터링 로직만 검증한다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from interview_assistant.contracts import InterviewSchema, SchemaItemDef, SessionContext, TranscriptEvent
from interview_assistant.structurer.transcript_reviewer import review_transcript


def _make_schema() -> InterviewSchema:
    return InterviewSchema(
        domain="숙련 커피 로스터",
        items=[
            SchemaItemDef(
                item_id="item_01",
                label="로스팅 종료 시점 판단",
                criteria="소리·향·시간 중 무엇을 보는가",
            ),
        ],
    )


def _make_session_context() -> SessionContext:
    return SessionContext(
        session_id="s1",
        interview_goal="로스팅 판단 기준을 구조화한다",
        expert_profile="15년차 로스터리 운영자",
        focus_notes="감각 vs 계기 판단",
    )


def _fake_client(json_text: str):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json_text))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.calls = calls
    return client


def test_review_transcript_skips_llm_call_when_fewer_than_two_turns_in_window():
    events = [TranscriptEvent(text="딱 한 턴", timestamp=0.0, turn_id="t001")]
    client = _fake_client(json.dumps({"corrections": []}))

    result = review_transcript(events, _make_schema(), _make_session_context(), client=client)

    assert result == []
    assert client.calls == []


def test_review_transcript_parses_mocked_correction():
    events = [
        TranscriptEvent(text="저 자에 로스팅", timestamp=0.0, turn_id="t001"),
        TranscriptEvent(text="아까 말한 저는 로스팅 얘기인데요", timestamp=1.0, turn_id="t002"),
    ]
    json_text = json.dumps(
        {
            "corrections": [
                {
                    "turn_id": "t001",
                    "corrected_text": "저는 로스팅",
                    "confidence": 0.95,
                    "reason": "다음 턴에서 '저는 로스팅'이라고 재확인함",
                }
            ]
        }
    )

    result = review_transcript(
        events, _make_schema(), _make_session_context(), client=_fake_client(json_text)
    )

    assert len(result) == 1
    assert result[0].turn_id == "t001"
    assert result[0].corrected_text == "저는 로스팅"
    assert result[0].confidence == 0.95
    assert result[0].reason == "다음 턴에서 '저는 로스팅'이라고 재확인함"


def test_review_transcript_filters_out_unknown_turn_id():
    events = [
        TranscriptEvent(text="첫 턴", timestamp=0.0, turn_id="t001"),
        TranscriptEvent(text="둘째 턴", timestamp=1.0, turn_id="t002"),
    ]
    json_text = json.dumps(
        {
            "corrections": [
                {
                    "turn_id": "t999_no_such_turn",
                    "corrected_text": "아무거나",
                    "confidence": 0.95,
                    "reason": "존재하지 않는 턴",
                }
            ]
        }
    )

    result = review_transcript(
        events, _make_schema(), _make_session_context(), client=_fake_client(json_text)
    )

    assert result == []


def test_review_transcript_clamps_confidence_to_unit_range():
    events = [
        TranscriptEvent(text="첫 턴", timestamp=0.0, turn_id="t001"),
        TranscriptEvent(text="둘째 턴", timestamp=1.0, turn_id="t002"),
    ]
    json_text = json.dumps(
        {
            "corrections": [
                {
                    "turn_id": "t001",
                    "corrected_text": "고친 문장",
                    "confidence": 1.5,
                    "reason": "확신도가 이상한 응답",
                }
            ]
        }
    )

    result = review_transcript(
        events, _make_schema(), _make_session_context(), client=_fake_client(json_text)
    )

    assert result[0].confidence == 1.0
