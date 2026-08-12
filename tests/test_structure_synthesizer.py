"""structure_synthesizer 계약 확인 테스트 (2026-08-05, llm_classifier.py 대체).

_build_system_prompt/_build_user_message는 API 호출 없이 순수하게 테스트한다.
synthesize_structure의 파싱 로직은 실제 OpenRouter 호출 없이 openai 클라이언트
모양을 흉내낸 가짜 client를 주입해 검증한다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from interview_assistant.contracts import (
    Contradiction,
    CoverageStatus,
    InterviewSchema,
    KnowledgeState,
    SchemaItem,
    SchemaItemDef,
    SessionContext,
    TranscriptEvent,
)
from interview_assistant.structurer.structure_synthesizer import (
    RenameProposal,
    _build_system_prompt,
    _build_user_message,
    synthesize_structure,
)


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


def _make_state(**overrides) -> KnowledgeState:
    state = KnowledgeState(
        session_id="s1",
        schema_items=[SchemaItem(item_id="item_01", label="로스팅 종료 시점 판단")],
    )
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _fake_client(json_text: str):
    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json_text))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_build_system_prompt_includes_schema_and_session_context():
    prompt = _build_system_prompt(_make_schema(), _make_session_context())

    assert "숙련 커피 로스터" in prompt
    assert "로스팅 종료 시점 판단" in prompt
    assert "소리·향·시간 중 무엇을 보는가" in prompt
    assert "로스팅 판단 기준을 구조화한다" in prompt
    assert "15년차 로스터리 운영자" in prompt
    assert "감각 vs 계기 판단" in prompt


def test_build_system_prompt_includes_batch_context_and_refs_instruction():
    prompt = _build_system_prompt(_make_schema(), _make_session_context())

    assert "하나의 맥락으로" in prompt
    assert "new_refs" in prompt
    assert "uncovered" in prompt and "partial" in prompt and "covered" in prompt


def test_build_system_prompt_forbids_vague_number_placeholders():
    prompt = _build_system_prompt(_make_schema(), _make_session_context())

    assert "(특정)" in prompt
    assert "22퍼센트" in prompt


def test_build_system_prompt_forbids_narrating_interview_progress():
    prompt = _build_system_prompt(_make_schema(), _make_session_context())

    assert "설명이 이어졌습니다" in prompt
    assert "아직 이야기되지 않았습니다" in prompt


def test_build_system_prompt_forbids_inline_turn_id_citation():
    prompt = _build_system_prompt(_make_schema(), _make_session_context())

    assert "[t009]" in prompt
    assert "인용" in prompt


def test_build_user_message_includes_current_summary_and_new_events():
    state = _make_state()
    state.schema_items[0].summary = "지금까지는 소리로만 판단한다는 내용"
    events = [TranscriptEvent(text="근데 사실 향이 더 먼저예요", timestamp=1.0, turn_id="t005")]

    message = _build_user_message(events, state)

    assert "지금까지는 소리로만 판단한다는 내용" in message
    assert "[t005] 근데 사실 향이 더 먼저예요" in message


def test_build_user_message_handles_empty_state():
    message = _build_user_message([], _make_state(schema_items=[]))

    assert "(아직 섹션 없음)" in message


def test_synthesize_structure_parses_section_update():
    json_text = json.dumps(
        {
            "sections": [
                {
                    "item_id": "item_01",
                    "summary": "향이 소리보다 먼저 나타나는 신호",
                    "status": "partial",
                    "new_refs": ["t005"],
                    "new_contradictions": [],
                    "resolved_contradiction_refs": [],
                }
            ],
            "new_sections": [],
            "renames": [],
            "question_turn_ids": [],
        }
    )
    events = [TranscriptEvent(text="향이 더 먼저예요", timestamp=1.0, turn_id="t005")]

    result = synthesize_structure(
        events, _make_schema(), _make_session_context(), _make_state(),
        client=_fake_client(json_text),
    )

    assert len(result.sections) == 1
    sec = result.sections[0]
    assert sec.item_id == "item_01"
    assert sec.summary == "향이 소리보다 먼저 나타나는 신호"
    assert sec.status == CoverageStatus.PARTIAL
    assert sec.new_refs == ["t005"]


def test_synthesize_structure_parses_new_section():
    json_text = json.dumps(
        {
            "sections": [],
            "new_sections": [
                {
                    "label": "후처리 방식",
                    "summary": "건조 후 숙성 과정에 대한 내용",
                    "status": "uncovered",
                    "refs": ["t010"],
                }
            ],
            "renames": [],
            "question_turn_ids": [],
        }
    )

    result = synthesize_structure(
        [TranscriptEvent(text="후처리도 중요해요", timestamp=1.0, turn_id="t010")],
        _make_schema(), _make_session_context(), _make_state(),
        client=_fake_client(json_text),
    )

    assert len(result.new_sections) == 1
    assert result.new_sections[0].label == "후처리 방식"
    assert result.new_sections[0].status == CoverageStatus.UNCOVERED
    assert result.new_sections[0].refs == ["t010"]


def test_synthesize_structure_parses_renames():
    json_text = json.dumps(
        {
            "sections": [],
            "new_sections": [],
            "renames": [{"item_id": "item_01", "new_label": "로스팅 재현성 확보"}],
            "question_turn_ids": [],
        }
    )

    result = synthesize_structure(
        [TranscriptEvent(text="발화", timestamp=1.0, turn_id="t001")],
        _make_schema(), _make_session_context(), _make_state(),
        client=_fake_client(json_text),
    )

    assert result.renames == [RenameProposal(item_id="item_01", new_label="로스팅 재현성 확보")]


def test_synthesize_structure_parses_question_turn_ids():
    json_text = json.dumps(
        {
            "sections": [],
            "new_sections": [],
            "renames": [],
            "question_turn_ids": ["t005"],
        }
    )

    result = synthesize_structure(
        [TranscriptEvent(text="생두는 어떻게 고르세요?", timestamp=1.0, turn_id="t005")],
        _make_schema(), _make_session_context(), _make_state(),
        client=_fake_client(json_text),
    )

    assert result.question_turn_ids == ["t005"]


def test_synthesize_structure_parses_new_and_resolved_contradictions():
    json_text = json.dumps(
        {
            "sections": [
                {
                    "item_id": "item_01",
                    "summary": "갱신된 요약",
                    "status": "covered",
                    "new_refs": ["t010"],
                    "new_contradictions": [{"with_ref": "t004", "note": "시간 기준과 상충"}],
                    "resolved_contradiction_refs": ["t002"],
                }
            ],
            "new_sections": [],
            "renames": [],
            "question_turn_ids": [],
        }
    )

    result = synthesize_structure(
        [TranscriptEvent(text="발화", timestamp=1.0, turn_id="t010")],
        _make_schema(), _make_session_context(), _make_state(),
        client=_fake_client(json_text),
    )

    sec = result.sections[0]
    assert sec.new_contradictions == [Contradiction(with_ref="t004", note="시간 기준과 상충")]
    assert sec.resolved_contradiction_refs == ["t002"]
