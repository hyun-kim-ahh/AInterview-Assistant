"""Question Engine LLM 투입(9단계) 계약 확인 테스트.

_build_system_prompt/_build_user_message는 순수 함수로 API 없이 테스트한다.
generate_questions는 이제 네트워크 우회 경로가 없으므로(항상 client.chat.completions.create를
부른다) 모든 파싱/윈도잉 테스트에 가짜(canned 또는 spy) client를 주입한다. 실제 API 품질
검증(모순확인 opportunistic 능력)은 OPENROUTER_API_KEY가 있을 때만 도는 라이브 테스트 하나로.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from interview_assistant.contracts import (
    Contradiction,
    CoverageStatus,
    KnowledgeState,
    QuestionType,
    SchemaItem,
    SessionContext,
    TranscriptEvent,
)
from interview_assistant.question_engine.question_engine import (
    _build_system_prompt,
    _build_user_message,
    _window_by_time,
    generate_questions,
)


def _make_session_context() -> SessionContext:
    return SessionContext(
        session_id="s1",
        interview_goal="로스팅 판단 기준을 구조화한다",
        focus_notes="감각 vs 계기 판단",
    )


def _canned_client(json_text: str):
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


def test_build_system_prompt_includes_session_context():
    prompt = _build_system_prompt(_make_session_context())

    assert "로스팅 판단 기준을 구조화한다" in prompt
    assert "감각 vs 계기 판단" in prompt


def test_build_system_prompt_includes_gap_type_by_default():
    prompt = _build_system_prompt(_make_session_context())

    assert "gap(누락)" in prompt


def test_build_system_prompt_omits_gap_type_when_no_schema():
    prompt = _build_system_prompt(_make_session_context(), has_schema=False)

    assert "gap(누락)" not in prompt
    assert "스키마 없이" in prompt


def test_build_system_prompt_omits_hint_line_when_no_trigger_hint():
    prompt = _build_system_prompt(_make_session_context())

    assert "적절한 시점" not in prompt


def test_build_system_prompt_includes_trigger_hint_when_provided():
    prompt = _build_system_prompt(_make_session_context(), trigger_hint="생두 가격 얘기하다 말을 흐림")

    assert "생두 가격 얘기하다 말을 흐림" in prompt
    assert "적절한 시점" in prompt


def test_build_user_message_includes_schema_state():
    state = KnowledgeState(
        session_id="s1",
        schema_items=[
            SchemaItem(
                item_id="item_01",
                label="로스팅 종료 시점 판단",
                status=CoverageStatus.UNCOVERED,
                summary="향으로 판단 (t002)",
            )
        ],
    )
    events = [TranscriptEvent(text="향이 먼저예요", timestamp=1.0, turn_id="t002")]

    message = _build_user_message(state, events)

    assert "item_01" in message
    assert "로스팅 종료 시점 판단" in message
    assert "uncovered" in message
    assert "향으로 판단" in message
    assert "t002" in message
    assert "향이 먼저예요" in message


def test_build_user_message_includes_confirmed_contradictions():
    state = KnowledgeState(
        session_id="s1",
        schema_items=[
            SchemaItem(
                item_id="item_01",
                label="로스팅 종료 시점 판단",
                status=CoverageStatus.COVERED,
                contradictions=[Contradiction(with_ref="t004", note="시간 안 본다더니 신입은 시간표로")],
            )
        ],
    )

    message = _build_user_message(state, [])

    assert "t004" in message
    assert "시간 안 본다더니 신입은 시간표로" in message


def test_generate_questions_windows_recent_events():
    # 윈도잉은 턴 개수가 아니라 시간(초) 기준(_RECENT_WINDOW_SECONDS=60) — 20초 간격으로
    # 11개(t001@20초~t011@220초)를 흩뿌려서, 마지막 이벤트(220초) 기준 컷오프는 160초.
    # t001~t007(20~140초)은 윈도우 밖, t008~t011(160~220초)만 안에 들어와야 한다.
    events = [
        TranscriptEvent(text=f"발화 번호 {i}", timestamp=float(i * 20), turn_id=f"t{i:03d}")
        for i in range(1, 12)
    ]
    response_json = json.dumps(
        {"candidates": [{"type": "expand", "text": "질문", "target_item": "", "refs": []}]}
    )
    client, captured = _spy_client(response_json)

    generate_questions(
        KnowledgeState(session_id="s1"), events, _make_session_context(), client=client
    )

    user_content = captured["messages"][1]["content"]
    for i in range(1, 8):  # t001~t007은 60초 창 밖이어야 함
        assert f"t{i:03d}" not in user_content
    for i in range(8, 12):  # t008~t011만 창 안에 있어야 함
        assert f"t{i:03d}" in user_content


def test_window_by_time_keeps_only_events_within_cutoff_of_last_event():
    events = [
        TranscriptEvent(text="a", timestamp=0.0, turn_id="t001"),
        TranscriptEvent(text="b", timestamp=50.0, turn_id="t002"),
        TranscriptEvent(text="c", timestamp=100.0, turn_id="t003"),
    ]

    windowed = _window_by_time(events, window_seconds=60)

    assert [e.turn_id for e in windowed] == ["t002", "t003"]


def test_window_by_time_empty_input_returns_empty():
    assert _window_by_time([], window_seconds=60) == []


def test_generate_questions_parses_mocked_response():
    response_json = json.dumps(
        {
            "candidates": [
                {"type": "probe", "text": "질문 A", "target_item": "", "refs": ["t002"]},
                {"type": "gap", "text": "질문 B", "target_item": "item_02", "refs": []},
                {
                    "type": "contradiction",
                    "text": "질문 C",
                    "target_item": "",
                    "refs": ["t004", "t010"],
                },
            ]
        }
    )

    result = generate_questions(
        KnowledgeState(session_id="s1"),
        [TranscriptEvent(text="발화", timestamp=1.0, turn_id="t010")],
        _make_session_context(),
        client=_canned_client(response_json),
    )

    types = [c.type for c in result.candidates]
    assert QuestionType.PROBE in types
    assert QuestionType.GAP in types
    assert QuestionType.CONTRADICTION in types
    probe = next(c for c in result.candidates if c.type == QuestionType.PROBE)
    assert probe.target_item is None  # 빈 문자열 -> None 변환
    gap = next(c for c in result.candidates if c.type == QuestionType.GAP)
    assert gap.target_item == "item_02"
    contradiction = next(c for c in result.candidates if c.type == QuestionType.CONTRADICTION)
    assert contradiction.refs == ["t004", "t010"]


def test_generate_questions_caps_at_four_and_prioritizes_contradiction():
    candidates_data = [
        {"type": "probe", "text": f"질문 {i}", "target_item": "", "refs": []} for i in range(4)
    ]
    candidates_data.append(
        {"type": "contradiction", "text": "모순 질문", "target_item": "", "refs": ["t004", "t010"]}
    )
    response_json = json.dumps({"candidates": candidates_data})

    result = generate_questions(
        KnowledgeState(session_id="s1"),
        [TranscriptEvent(text="발화", timestamp=1.0, turn_id="t010")],
        _make_session_context(),
        client=_canned_client(response_json),
    )

    assert len(result.candidates) == 4
    assert any(c.type == QuestionType.CONTRADICTION for c in result.candidates)


def test_generated_at_uses_last_windowed_event_turn_id():
    events = [
        TranscriptEvent(text="첫 발화", timestamp=1.0, turn_id="t001"),
        TranscriptEvent(text="둘째 발화", timestamp=2.0, turn_id="t005"),
    ]
    response_json = json.dumps(
        {"candidates": [{"type": "expand", "text": "질문", "target_item": "", "refs": []}]}
    )

    result = generate_questions(
        KnowledgeState(session_id="s1"),
        events,
        _make_session_context(),
        client=_canned_client(response_json),
    )

    assert result.generated_at == "t005"


def test_generate_questions_omits_gap_instruction_when_state_has_no_schema_items():
    response_json = json.dumps(
        {"candidates": [{"type": "expand", "text": "질문", "target_item": "", "refs": []}]}
    )
    client, captured = _spy_client(response_json)

    generate_questions(
        KnowledgeState(session_id="s1"),  # schema_items 기본값 []
        [TranscriptEvent(text="발화", timestamp=1.0, turn_id="t001")],
        _make_session_context(),
        client=client,
    )

    system_content = captured["messages"][0]["content"]
    assert "gap(누락)" not in system_content


def test_generate_questions_passes_trigger_hint_into_system_prompt():
    response_json = json.dumps(
        {"candidates": [{"type": "expand", "text": "질문", "target_item": "", "refs": []}]}
    )
    client, captured = _spy_client(response_json)

    generate_questions(
        KnowledgeState(session_id="s1"),
        [TranscriptEvent(text="발화", timestamp=1.0, turn_id="t001")],
        _make_session_context(),
        trigger_hint="전문가가 원두 가격 얘기하다 말을 흐림",
        client=client,
    )

    system_content = captured["messages"][0]["content"]
    assert "전문가가 원두 가격 얘기하다 말을 흐림" in system_content


def test_generated_at_empty_when_no_recent_events():
    response_json = json.dumps(
        {"candidates": [{"type": "expand", "text": "질문", "target_item": "", "refs": []}]}
    )

    result = generate_questions(
        KnowledgeState(session_id="s1"),
        [],
        _make_session_context(),
        client=_canned_client(response_json),
    )

    assert result.generated_at == ""


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY가 있어야 실제 LLM 모순확인 능력을 검증한다",
)
def test_generate_questions_live_llm_detects_contradiction():
    state = KnowledgeState(
        session_id="s1",
        schema_items=[
            SchemaItem(
                item_id="item_01",
                label="로스팅 종료 시점 판단",
                summary="타이머는 참고만 하고, 시간만 믿으면 망한다 — 시간은 절대 기준이 아니다 (t004)",
            ),
            SchemaItem(
                item_id="item_02",
                label="신입 교육 방식",
                summary="신입한테는 감이 없으니 오히려 시간표부터 외우게 한다 (t010)",
            ),
        ],
    )
    recent_events = [
        TranscriptEvent(text="참고는 하는데 절대 기준은 아니에요. 시간만 믿으면 망해요.", timestamp=30.0, turn_id="t004"),
        TranscriptEvent(text="신입한테는 오히려 시간표부터 외우게 해요.", timestamp=118.0, turn_id="t010"),
    ]

    result = generate_questions(state, recent_events, _make_session_context())

    assert 1 <= len(result.candidates) <= 4
    assert all(isinstance(c.type, QuestionType) for c in result.candidates)
    assert all(c.text.strip() for c in result.candidates)
    assert any(c.type == QuestionType.CONTRADICTION for c in result.candidates)
