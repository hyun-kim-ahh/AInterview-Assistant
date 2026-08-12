"""trigger_judge 계약 확인 테스트 — "지금 추천 질문을 생성할 때인가" 판단.

다른 LLM 호출 모듈들과 동일한 스타일: API 호출 없이 openai 클라이언트 모양을
흉내낸 가짜 client를 주입해 파싱/스킵 로직을 검증한다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from interview_assistant.contracts import (
    CoverageStatus,
    KnowledgeState,
    SchemaItem,
    SessionContext,
    TranscriptEvent,
)
from interview_assistant.question_engine.trigger_judge import (
    _build_system_prompt,
    _build_user_message,
    judge_trigger,
)


def _make_session_context() -> SessionContext:
    return SessionContext(
        session_id="s1",
        interview_goal="로스팅 판단 기준을 구조화한다",
        focus_notes="감각 vs 계기 판단",
    )


def _make_state() -> KnowledgeState:
    return KnowledgeState(
        session_id="s1",
        schema_items=[
            SchemaItem(
                item_id="item_01",
                label="로스팅 종료 시점 판단",
                status=CoverageStatus.COVERED,
                summary="향으로 판단",
            ),
            SchemaItem(item_id="item_02", label="생두 품질 선별", status=CoverageStatus.UNCOVERED),
        ],
    )


def _make_events() -> list[TranscriptEvent]:
    return [
        TranscriptEvent(text="첫 발화", timestamp=0.0, turn_id="t001"),
        TranscriptEvent(text="둘째 발화", timestamp=10.0, turn_id="t002"),
    ]


def _fake_client(json_text: str):
    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json_text))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _spy_client(json_text: str):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json_text))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, captured


def test_judge_trigger_skips_call_when_too_little_recent_conversation():
    result = judge_trigger(
        _make_state(),
        [TranscriptEvent(text="딱 한 턴", timestamp=0.0, turn_id="t001")],
        _make_session_context(),
        elapsed_seconds=30.0,
        client=_fake_client("이 값은 안 쓰임"),
    )

    assert result.should_trigger is False
    assert result.reason == ""


def test_judge_trigger_parses_mocked_response_when_triggered():
    json_text = json.dumps(
        {"should_trigger": True, "reason": "전문가가 원두 가격 얘기하다 말을 흐림"}
    )

    result = judge_trigger(
        _make_state(), _make_events(), _make_session_context(), elapsed_seconds=600.0,
        client=_fake_client(json_text),
    )

    assert result.should_trigger is True
    assert result.reason == "전문가가 원두 가격 얘기하다 말을 흐림"


def test_judge_trigger_parses_mocked_response_when_not_triggered():
    json_text = json.dumps({"should_trigger": False, "reason": ""})

    result = judge_trigger(
        _make_state(), _make_events(), _make_session_context(), elapsed_seconds=30.0,
        client=_fake_client(json_text),
    )

    assert result.should_trigger is False
    assert result.reason == ""


def test_build_system_prompt_includes_goal_focus_and_progress():
    prompt = _build_system_prompt(_make_session_context(), turn_count=12, elapsed_seconds=185.0)

    assert "로스팅 판단 기준을 구조화한다" in prompt
    assert "감각 vs 계기 판단" in prompt
    assert "12턴" in prompt
    assert "3.1분" in prompt


def test_build_system_prompt_includes_interesting_point_criterion():
    prompt = _build_system_prompt(_make_session_context(), turn_count=12, elapsed_seconds=185.0)

    assert "흥미로운 지점" in prompt


def test_build_user_message_includes_windowed_transcript():
    message = _build_user_message(_make_state(), _make_events())

    assert "t001" in message
    assert "첫 발화" in message
    assert "t002" in message
    assert "둘째 발화" in message


def test_build_user_message_includes_schema_state():
    message = _build_user_message(_make_state(), [])

    assert "로스팅 종료 시점 판단" in message
    assert "향으로 판단" in message
    assert "생두 품질 선별" in message



def test_judge_trigger_sends_windowed_events_only():
    events = [
        TranscriptEvent(text=f"발화 {i}", timestamp=float(i * 30), turn_id=f"t{i:03d}")
        for i in range(1, 8)  # 30초 간격, t001@30s ~ t007@210s
    ]
    json_text = json.dumps({"should_trigger": False, "reason": ""})
    client, captured = _spy_client(json_text)

    judge_trigger(_make_state(), events, _make_session_context(), elapsed_seconds=210.0, client=client)

    user_content = captured["messages"][1]["content"]
    assert "t001" not in user_content  # 120초 창 밖(TRIGGER_JUDGE_WINDOW_SECONDS=120)
    assert "t007" in user_content
