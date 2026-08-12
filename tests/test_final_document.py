"""final_document 계약 확인 테스트 — 인터뷰 종료 시 1회 마지막 다듬기 문서 작성.

structure_synthesizer.py 테스트와 동일한 스타일: API 호출 없이 openai 클라이언트
모양을 흉내낸 가짜 client를 주입해 파싱/필터링 로직을 검증한다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from interview_assistant.contracts import (
    Contradiction,
    InterviewSchema,
    KnowledgeState,
    SchemaItem,
    SessionContext,
)
from interview_assistant.structurer.final_document import (
    _build_system_prompt,
    _build_user_message,
    write_final_document,
)


def _make_schema() -> InterviewSchema:
    return InterviewSchema(domain="숙련 커피 로스터", items=[])


def _make_session_context() -> SessionContext:
    return SessionContext(
        session_id="s1",
        interview_goal="로스팅 판단 기준을 구조화한다",
        expert_profile="15년차 로스터리 운영자",
        focus_notes="감각 vs 계기 판단",
    )


def _make_state_with_content() -> KnowledgeState:
    return KnowledgeState(
        session_id="s1",
        schema_items=[
            SchemaItem(
                item_id="item_01",
                label="로스팅 종료 시점 판단",
                summary="향으로 판단",
                contradictions=[Contradiction(with_ref="t004", note="시간 기준과 상충")],
            ),
            SchemaItem(item_id="item_02", label="생두 품질 선별"),  # 요약 없음
        ],
    )


def _fake_client(json_text: str):
    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json_text))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_write_final_document_skips_call_when_nothing_to_summarize():
    empty_state = KnowledgeState(session_id="s1", schema_items=[SchemaItem(item_id="item_01", label="항목")])

    result = write_final_document(
        empty_state, _make_schema(), _make_session_context(), client=_fake_client("이 값은 안 쓰임")
    )

    assert result.overview == ""
    assert result.section_summaries == {}


def test_write_final_document_parses_mocked_response():
    json_text = json.dumps(
        {
            "overview": "## 종합 정리\n\n전체적으로 감각적 판단을 계기보다 우선하는 경향이 뚜렷했다.",
            "sections": [{"item_id": "item_01", "summary": "향으로 종료 시점을 판단한다."}],
        }
    )

    result = write_final_document(
        _make_state_with_content(),
        _make_schema(),
        _make_session_context(),
        client=_fake_client(json_text),
    )

    assert result.overview == "## 종합 정리\n\n전체적으로 감각적 판단을 계기보다 우선하는 경향이 뚜렷했다."
    assert result.section_summaries == {"item_01": "향으로 종료 시점을 판단한다."}


def test_write_final_document_filters_out_unknown_item_id():
    json_text = json.dumps(
        {
            "overview": "개요",
            "sections": [
                {"item_id": "item_01", "summary": "정상 섹션"},
                {"item_id": "item_99_없는_섹션", "summary": "할루시네이션"},
            ],
        }
    )

    result = write_final_document(
        _make_state_with_content(),
        _make_schema(),
        _make_session_context(),
        client=_fake_client(json_text),
    )

    assert result.section_summaries == {"item_01": "정상 섹션"}


def test_build_system_prompt_includes_domain_and_goal():
    prompt = _build_system_prompt(_make_schema(), _make_session_context())

    assert "숙련 커피 로스터" in prompt
    assert "로스팅 판단 기준을 구조화한다" in prompt
    assert "지어내" in prompt  # 보수적 지시 포함 확인
    assert "Markdown" in prompt
    assert "표" in prompt  # 표 활용 지시 포함 확인
    assert "부제목" in prompt  # 계층 구조 지시 포함 확인


def test_build_system_prompt_forbids_vague_number_placeholders():
    prompt = _build_system_prompt(_make_schema(), _make_session_context())

    assert "(특정)" in prompt
    assert "22퍼센트" in prompt


def test_build_system_prompt_forbids_inline_turn_id_citation():
    prompt = _build_system_prompt(_make_schema(), _make_session_context())

    assert "[t009]" in prompt
    assert "인용" in prompt


def test_build_user_message_includes_sections():
    message = _build_user_message(_make_state_with_content())

    assert "item_01" in message
    assert "향으로 판단" in message
    assert "시간 기준과 상충" in message
    assert "(아직 내용 없음)" in message  # item_02
