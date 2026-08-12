"""Structurer 계약 확인 테스트 — 배치 재종합 방식(2026-08-05).

synthesize 콜백을 가짜 함수로 주입해 실제 API 호출 없이 Structurer의 라우팅 로직(섹션
요약 갱신 / 새 섹션 동적 생성 / 할루시네이션 방어 / 폭주 방지 / 모순 new·resolved
처리)을 검증한다. 실제 OpenRouter 호출을 쓰는 "품질 검증" 테스트 하나만
OPENROUTER_API_KEY가 있을 때만 돌도록 skip 처리한다(dev-plan 8단계 원문: "fixture로
KnowledgeState 품질 검증").
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from interview_assistant.contracts import (
    Contradiction,
    CoverageStatus,
    InterviewSchema,
    SchemaItemDef,
    SessionContext,
    Speaker,
    TranscriptEvent,
)
from interview_assistant.sources.file_replay import FileReplaySource
from interview_assistant.structurer.structure_synthesizer import (
    NewSectionProposal,
    RenameProposal,
    SectionUpdate,
    SynthesisResult,
)
from interview_assistant.structurer.structurer import MAX_DYNAMIC_ITEMS, MAX_RENAMES_PER_ITEM, Structurer

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"

_EMPTY_RESULT = SynthesisResult(sections=[], new_sections=[])


def _make_schema() -> InterviewSchema:
    return InterviewSchema(
        domain="숙련 커피 로스터 (테스트용)",
        items=[
            SchemaItemDef(item_id="item_01", label="로스팅 종료 시점 판단"),
            SchemaItemDef(item_id="item_02", label="생두 품질 선별"),
        ],
    )


def _make_session_context() -> SessionContext:
    return SessionContext(session_id="test_session_01", interview_goal="테스트용 목적")


def _no_updates(events, schema, session_context, state):
    return _EMPTY_RESULT


def test_initializes_schema_items_from_schema():
    state = Structurer(_make_schema(), _make_session_context(), synthesize=_no_updates).state

    assert state.session_id == "test_session_01"
    assert [i.item_id for i in state.schema_items] == ["item_01", "item_02"]
    assert [i.label for i in state.schema_items] == ["로스팅 종료 시점 판단", "생두 품질 선별"]
    assert all(i.status == CoverageStatus.UNCOVERED for i in state.schema_items)
    assert all(i.summary == "" for i in state.schema_items)
    assert all(i.source_refs == [] for i in state.schema_items)
    assert all(i.contradictions == [] for i in state.schema_items)


def test_ingest_batch_of_only_interviewer_turns_is_skipped_without_calling_synthesize():
    calls = []

    def spy(events, schema, session_context, state):
        calls.append(events)
        return _EMPTY_RESULT

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=spy)
    event = TranscriptEvent(
        text="타이머는 안 보세요?", timestamp=1.0, turn_id="t003", speaker=Speaker.INTERVIEWER
    )

    structurer.ingest_batch([event])

    assert calls == []  # synthesize가 아예 호출되지 않아야 함


def test_ingest_batch_filters_out_interviewer_turns_but_synthesizes_the_rest():
    calls = []

    def spy(events, schema, session_context, state):
        calls.append(events)
        return _EMPTY_RESULT

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=spy)
    interviewer_turn = TranscriptEvent(
        text="타이머는 안 보세요?", timestamp=1.0, turn_id="t003", speaker=Speaker.INTERVIEWER
    )
    expert_turn = TranscriptEvent(
        text="참고는 하는데 절대 기준은 아니에요", timestamp=2.0, turn_id="t004", speaker=Speaker.EXPERT
    )

    structurer.ingest_batch([interviewer_turn, expert_turn])

    assert len(calls) == 1
    assert [e.turn_id for e in calls[0]] == ["t004"]  # 인터뷰어 턴만 배치에서 제외됨


def test_ingest_batch_updates_section_summary_status_and_source_refs():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[
                SectionUpdate(
                    item_id="item_01",
                    summary="향으로 판단하는 것이 기준",
                    status=CoverageStatus.PARTIAL,
                    new_refs=["t002"],
                    new_contradictions=[],
                    resolved_contradiction_refs=[],
                )
            ],
            new_sections=[],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    event = TranscriptEvent(
        text="소리보다 향이 먼저예요", timestamp=1.0, turn_id="t002", speaker=Speaker.EXPERT
    )

    structurer.ingest_batch([event])

    item_01 = structurer.state.schema_items[0]
    assert item_01.summary == "향으로 판단하는 것이 기준"
    assert item_01.status == CoverageStatus.PARTIAL
    assert item_01.source_refs == ["t002"]


def test_ingest_batch_accumulates_source_refs_across_multiple_batches():
    calls = {"n": 0}

    def synthesize(events, schema, session_context, state):
        calls["n"] += 1
        ref = events[0].turn_id
        return SynthesisResult(
            sections=[
                SectionUpdate(
                    item_id="item_01", summary=f"요약 {calls['n']}", status=CoverageStatus.PARTIAL,
                    new_refs=[ref], new_contradictions=[], resolved_contradiction_refs=[],
                )
            ], new_sections=[],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.ingest_batch([TranscriptEvent(text="발화1", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT)])
    structurer.ingest_batch([TranscriptEvent(text="발화2", timestamp=2.0, turn_id="t002", speaker=Speaker.EXPERT)])

    item_01 = structurer.state.schema_items[0]
    assert item_01.summary == "요약 2"  # 최신 요약으로 교체됨(누적 아님)
    assert item_01.source_refs == ["t001", "t002"]  # 근거는 누적됨


def test_ingest_batch_merges_content_spanning_multiple_turns_into_one_section():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[
                SectionUpdate(
                    item_id="item_02",
                    summary="생두 선별 시 색 균일도와 크기를 기준으로 봄",
                    status=CoverageStatus.PARTIAL,
                    new_refs=["t010", "t011"],
                    new_contradictions=[],
                    resolved_contradiction_refs=[],
                )
            ],
            new_sections=[],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    events = [
        TranscriptEvent(text="생두를 고를 때는", timestamp=1.0, turn_id="t010", speaker=Speaker.EXPERT),
        TranscriptEvent(text="일단 색 균일도랑 크기를 봅니다", timestamp=2.0, turn_id="t011", speaker=Speaker.EXPERT),
    ]

    structurer.ingest_batch(events)

    state = structurer.state
    item_02 = state.schema_items[1]
    assert item_02.source_refs == ["t010", "t011"]


def test_ingest_batch_mixed_question_and_informative_turns():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[
                SectionUpdate(
                    item_id="item_01", summary="온도보다 냄새가 먼저 변함", status=CoverageStatus.PARTIAL,
                    new_refs=["t012"], new_contradictions=[], resolved_contradiction_refs=[],
                )
            ],
            new_sections=[],
            question_turn_ids=["t011"],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    events = [
        TranscriptEvent(text="생두는 어떻게 고르세요?", timestamp=1.0, turn_id="t011", speaker=Speaker.EXPERT),
        TranscriptEvent(
            text="온도보다 냄새가 항상 먼저 변하거든요", timestamp=2.0, turn_id="t012", speaker=Speaker.EXPERT
        ),
    ]

    structurer.ingest_batch(events)

    state = structurer.state
    assert state.schema_items[0].source_refs == ["t012"]


def test_ingest_batch_drops_section_update_when_all_refs_are_hallucinated():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[
                SectionUpdate(
                    item_id="item_01", summary="할루시네이션된 근거", status=CoverageStatus.PARTIAL,
                    new_refs=["t999_없는_턴"], new_contradictions=[], resolved_contradiction_refs=[],
                )
            ],
            new_sections=[],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    event = TranscriptEvent(text="발화", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT)

    structurer.ingest_batch([event])

    state = structurer.state
    assert state.schema_items[0].summary == ""  # 갱신 자체가 버려짐


def test_ingest_batch_drops_section_update_with_empty_summary():
    """요약이 빈 문자열인 갱신은 통째로 버려야 한다 — 안 그러면 이미 있던 요약을
    빈 문자열로 덮어써 내용을 잃는다."""

    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[
                SectionUpdate(
                    item_id="item_01", summary="", status=CoverageStatus.PARTIAL,
                    new_refs=["t002"], new_contradictions=[], resolved_contradiction_refs=[],
                )
            ],
            new_sections=[],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.state.schema_items[0].summary = "기존 요약"
    structurer.state.schema_items[0].source_refs = ["t002"]

    structurer.ingest_batch(
        [TranscriptEvent(text="발화", timestamp=1.0, turn_id="t002", speaker=Speaker.EXPERT)]
    )

    assert structurer.state.schema_items[0].summary == "기존 요약"  # 덮어써지지 않음


def test_ingest_batch_drops_new_section_proposal_with_empty_summary():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[],
            new_sections=[
                NewSectionProposal(label="빈 섹션", summary="", status=CoverageStatus.UNCOVERED, refs=[events[0].turn_id])
            ],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)

    structurer.ingest_batch([TranscriptEvent(text="발화", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT)])

    assert len(structurer.state.schema_items) == 2  # 새 섹션이 만들어지지 않음


def test_ingest_batch_unknown_item_id_is_ignored():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[
                SectionUpdate(
                    item_id="item_99_does_not_exist", summary="x", status=CoverageStatus.PARTIAL,
                    new_refs=["t001"], new_contradictions=[], resolved_contradiction_refs=[],
                )
            ],
            new_sections=[],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    event = TranscriptEvent(text="발화", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT)

    structurer.ingest_batch([event])

    state = structurer.state
    assert all(i.summary == "" for i in state.schema_items)


def test_ingest_batch_new_section_creates_dynamic_item():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[],
            new_sections=[
                NewSectionProposal(
                    label="후처리 방식", summary="건조 후 숙성 얘기", status=CoverageStatus.UNCOVERED,
                    refs=[events[0].turn_id],
                )
            ],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    event = TranscriptEvent(text="후처리도 중요해요", timestamp=1.0, turn_id="t010", speaker=Speaker.EXPERT)

    structurer.ingest_batch([event])

    state = structurer.state
    assert len(state.schema_items) == 3
    new_item = state.schema_items[2]
    assert new_item.label == "후처리 방식"
    assert new_item.item_id.startswith("dyn_")
    assert new_item.summary == "건조 후 숙성 얘기"
    assert new_item.source_refs == ["t010"]


def test_ingest_batch_multiple_new_sections_get_distinct_dynamic_ids():
    def synthesize(events, schema, session_context, state):
        ref = events[0].turn_id
        return SynthesisResult(
            sections=[],
            new_sections=[
                NewSectionProposal(label="주제 A", summary="a", status=CoverageStatus.UNCOVERED, refs=[ref]),
                NewSectionProposal(label="주제 B", summary="b", status=CoverageStatus.UNCOVERED, refs=[ref]),
            ],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.ingest_batch([TranscriptEvent(text="발화", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT)])

    ids = [i.item_id for i in structurer.state.schema_items[2:]]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_ingest_batch_new_section_with_label_matching_existing_section_folds_in():
    call_count = {"n": 0}

    def synthesize(events, schema, session_context, state):
        call_count["n"] += 1
        ref = events[0].turn_id
        if call_count["n"] == 1:
            return SynthesisResult(
                sections=[],
                new_sections=[NewSectionProposal(label="후처리 방식", summary="첫 언급", status=CoverageStatus.UNCOVERED, refs=[ref])],
            )
        return SynthesisResult(
            sections=[],
            new_sections=[NewSectionProposal(label="후처리 방식", summary="추가 언급까지 반영한 요약", status=CoverageStatus.PARTIAL, refs=[ref])],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.ingest_batch([TranscriptEvent(text="발화1", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT)])
    structurer.ingest_batch([TranscriptEvent(text="발화2", timestamp=2.0, turn_id="t002", speaker=Speaker.EXPERT)])

    state = structurer.state
    dynamic_items = [i for i in state.schema_items if i.label == "후처리 방식"]
    assert len(dynamic_items) == 1  # 중복 생성 안 됨, 하나로 합쳐짐
    assert dynamic_items[0].summary == "추가 언급까지 반영한 요약"
    assert dynamic_items[0].source_refs == ["t001", "t002"]


def test_ingest_batch_stops_creating_new_sections_after_max_dynamic_items_reached():
    def synthesize(events, schema, session_context, state):
        ref = events[0].turn_id
        return SynthesisResult(
            sections=[],
            new_sections=[NewSectionProposal(label=f"넘치는 주제 {ref}", summary="c", status=CoverageStatus.UNCOVERED, refs=[ref])],
        )

    structurer = Structurer(InterviewSchema(domain="d", items=[]), _make_session_context(), synthesize=synthesize)
    for i in range(MAX_DYNAMIC_ITEMS):
        structurer.ingest_batch(
            [TranscriptEvent(text=f"발화{i}", timestamp=float(i), turn_id=f"t{i:03d}", speaker=Speaker.EXPERT)]
        )
    assert len(structurer.state.schema_items) == MAX_DYNAMIC_ITEMS

    structurer.ingest_batch(
        [TranscriptEvent(text="넘치는 발화", timestamp=999.0, turn_id="t999", speaker=Speaker.EXPERT)]
    )

    assert len(structurer.state.schema_items) == MAX_DYNAMIC_ITEMS


def test_ingest_batch_applies_rename_to_matching_item():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[], new_sections=[],
            renames=[RenameProposal(item_id="item_01", new_label="새 제목")],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.ingest_batch(
        [TranscriptEvent(text="사실 이건 다른 얘기였네요", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT)]
    )

    assert structurer.state.schema_items[0].label == "새 제목"


def test_ingest_batch_question_turn_still_applies_rename():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[], new_sections=[],
            renames=[RenameProposal(item_id="item_01", new_label="새 제목")],
            question_turn_ids=[events[0].turn_id],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.ingest_batch(
        [TranscriptEvent(text="이건 왜 그런가요?", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT)]
    )

    assert structurer.state.schema_items[0].label == "새 제목"


def test_ingest_batch_rename_skipped_when_new_label_collides_with_another_item():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[], new_sections=[],
            renames=[RenameProposal(item_id="item_01", new_label="생두 품질 선별")],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.ingest_batch([TranscriptEvent(text="발화", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT)])

    labels = [i.label for i in structurer.state.schema_items]
    assert labels == ["로스팅 종료 시점 판단", "생두 품질 선별"]


def test_ingest_batch_rename_of_unknown_item_id_is_ignored():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[], new_sections=[],
            renames=[RenameProposal(item_id="item_99_does_not_exist", new_label="새 제목")],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.ingest_batch([TranscriptEvent(text="발화", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT)])

    labels = [i.label for i in structurer.state.schema_items]
    assert labels == ["로스팅 종료 시점 판단", "생두 품질 선별"]


def test_ingest_batch_rename_with_blank_label_is_ignored():
    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[], new_sections=[],
            renames=[RenameProposal(item_id="item_01", new_label="   ")],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.ingest_batch([TranscriptEvent(text="발화", timestamp=1.0, turn_id="t001", speaker=Speaker.EXPERT)])

    assert structurer.state.schema_items[0].label == "로스팅 종료 시점 판단"


def test_ingest_batch_rename_stops_after_max_renames_per_item_reached():
    labels = iter([f"제목 {i}" for i in range(MAX_RENAMES_PER_ITEM + 1)])

    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[], new_sections=[],
            renames=[RenameProposal(item_id="item_01", new_label=next(labels))],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    for i in range(MAX_RENAMES_PER_ITEM + 1):
        structurer.ingest_batch(
            [TranscriptEvent(text=f"발화{i}", timestamp=float(i), turn_id=f"t{i:03d}", speaker=Speaker.EXPERT)]
        )

    assert structurer.state.schema_items[0].label == f"제목 {MAX_RENAMES_PER_ITEM - 1}"


def test_ingest_batch_new_contradiction_requires_known_turn_id():
    """with_ref는 이번 배치뿐 아니라 이 섹션이 지금까지 누적한 turn_id(source_refs)
    전체에서 골라도 허용돼야 하지만, 완전히 낯선 turn_id면 할루시네이션으로 버려야
    한다."""

    def synthesize(events, schema, session_context, state):
        return SynthesisResult(
            sections=[
                SectionUpdate(
                    item_id="item_01", summary="a", status=CoverageStatus.PARTIAL,
                    new_refs=["t002"],
                    new_contradictions=[
                        Contradiction(with_ref="t999_없는_턴", note="할루시네이션"),
                    ],
                    resolved_contradiction_refs=[],
                )
            ],
            new_sections=[],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.ingest_batch([TranscriptEvent(text="발화", timestamp=1.0, turn_id="t002", speaker=Speaker.EXPERT)])

    assert structurer.state.schema_items[0].contradictions == []


def test_ingest_batch_new_contradiction_can_reference_earlier_batch_turn_id():
    calls = {"n": 0}

    def synthesize(events, schema, session_context, state):
        calls["n"] += 1
        if calls["n"] == 1:
            return SynthesisResult(
                sections=[
                    SectionUpdate(
                        item_id="item_01", summary="시간만 참고", status=CoverageStatus.PARTIAL,
                        new_refs=["t004"], new_contradictions=[], resolved_contradiction_refs=[],
                    )
                ], new_sections=[],
            )
        return SynthesisResult(
            sections=[
                SectionUpdate(
                    item_id="item_01", summary="신입은 시간표부터", status=CoverageStatus.PARTIAL,
                    new_refs=["t010"],
                    new_contradictions=[Contradiction(with_ref="t004", note="시간 기준과 상충")],
                    resolved_contradiction_refs=[],
                )
            ], new_sections=[],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.ingest_batch([TranscriptEvent(text="시간만 참고", timestamp=1.0, turn_id="t004", speaker=Speaker.EXPERT)])
    structurer.ingest_batch([TranscriptEvent(text="신입은 시간표부터", timestamp=2.0, turn_id="t010", speaker=Speaker.EXPERT)])

    contradictions = structurer.state.schema_items[0].contradictions
    assert len(contradictions) == 1
    assert contradictions[0].with_ref == "t004"


def test_ingest_batch_resolved_contradiction_removes_it():
    calls = {"n": 0}

    def synthesize(events, schema, session_context, state):
        calls["n"] += 1
        if calls["n"] == 1:
            return SynthesisResult(
                sections=[
                    SectionUpdate(
                        item_id="item_01", summary="a", status=CoverageStatus.PARTIAL,
                        new_refs=["t004"],
                        new_contradictions=[Contradiction(with_ref="t004", note="모순")],
                        resolved_contradiction_refs=[],
                    )
                ], new_sections=[],
            )
        return SynthesisResult(
            sections=[
                SectionUpdate(
                    item_id="item_01", summary="정정된 요약", status=CoverageStatus.PARTIAL,
                    new_refs=["t011"], new_contradictions=[],
                    resolved_contradiction_refs=["t004"],
                )
            ], new_sections=[],
        )

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=synthesize)
    structurer.ingest_batch([TranscriptEvent(text="발화1", timestamp=1.0, turn_id="t004", speaker=Speaker.EXPERT)])
    assert len(structurer.state.schema_items[0].contradictions) == 1

    structurer.ingest_batch([TranscriptEvent(text="정정합니다", timestamp=2.0, turn_id="t011", speaker=Speaker.EXPERT)])

    assert structurer.state.schema_items[0].contradictions == []


def test_seed_outline_appends_dynamic_items_for_proposed_labels():
    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=_no_updates)

    structurer.seed_outline(propose=lambda schema, ctx: ["새 예상 주제 1", "새 예상 주제 2"])

    labels = [i.label for i in structurer.state.schema_items]
    assert "새 예상 주제 1" in labels
    assert "새 예상 주제 2" in labels
    assert len(structurer.state.schema_items) == 4


def test_seed_outline_skips_labels_matching_existing_items():
    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=_no_updates)

    structurer.seed_outline(propose=lambda schema, ctx: ["로스팅 종료 시점 판단", "정말 새로운 주제"])

    labels = [i.label for i in structurer.state.schema_items]
    assert labels.count("로스팅 종료 시점 판단") == 1
    assert "정말 새로운 주제" in labels


def test_seed_outline_swallows_exception_from_propose_and_leaves_state_unchanged():
    def failing_propose(schema, ctx):
        raise RuntimeError("네트워크 오류")

    structurer = Structurer(_make_schema(), _make_session_context(), synthesize=_no_updates)
    before = list(structurer.state.schema_items)

    structurer.seed_outline(propose=failing_propose)

    assert structurer.state.schema_items == before


def test_full_fixture_end_to_end_with_fake_synthesize():
    """실제 fixture 3종 전부 로드 + FileReplaySource + Structurer 배선 확인(가짜
    재종합기) — 전체 전사를 한 번의 ingest_batch로 처리한다."""
    schema_data = json.loads((SCHEMAS_DIR / "example_schema.json").read_text(encoding="utf-8"))
    schema = InterviewSchema(
        domain=schema_data["domain"],
        items=[SchemaItemDef(**item) for item in schema_data["items"]],
    )
    ctx_data = json.loads(
        (FIXTURES_DIR / "example_session_context.json").read_text(encoding="utf-8")
    )
    session_context = SessionContext(**ctx_data)

    def synthesize(events, schema, session_context, state):
        t002 = next((e for e in events if e.turn_id == "t002"), None)
        if t002 is None:
            return _EMPTY_RESULT
        return SynthesisResult(
            sections=[
                SectionUpdate(
                    item_id="item_01", summary=t002.text, status=CoverageStatus.PARTIAL,
                    new_refs=["t002"], new_contradictions=[], resolved_contradiction_refs=[],
                )
            ],
            new_sections=[],
        )

    structurer = Structurer(schema, session_context, synthesize=synthesize)
    events = list(FileReplaySource(FIXTURES_DIR / "fake_transcript.jsonl").stream())
    structurer.ingest_batch(events)

    state = structurer.state
    assert len(state.schema_items) == 4
    assert state.schema_items[0].summary != ""  # t002만 item_01로 반영됨


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY가 있어야 실제 LLM 품질 검증 테스트를 돈다",
)
def test_full_fixture_end_to_end_live_llm():
    """dev-plan 8단계의 실제 요구사항: 실제 fixture로 KnowledgeState 품질 검증.
    seed_outline()과 실제 재종합을 전부 실 OpenRouter 호출로 확인한다."""
    schema_data = json.loads((SCHEMAS_DIR / "example_schema.json").read_text(encoding="utf-8"))
    schema = InterviewSchema(
        domain=schema_data["domain"],
        items=[SchemaItemDef(**item) for item in schema_data["items"]],
    )
    ctx_data = json.loads(
        (FIXTURES_DIR / "example_session_context.json").read_text(encoding="utf-8")
    )
    session_context = SessionContext(**ctx_data)

    structurer = Structurer(schema, session_context)  # 기본 synthesize = 실제 OpenRouter 호출
    structurer.seed_outline()
    seeded_count = len(structurer.state.schema_items)
    assert seeded_count >= 4

    events = list(FileReplaySource(FIXTURES_DIR / "fake_transcript.jsonl").stream())
    structurer.ingest_batch(events)

    state = structurer.state
    assert any(item.summary for item in state.schema_items)  # 최소 한 섹션은 실제 반영됨
    all_refs = {r for i in state.schema_items for r in i.source_refs}
    assert all_refs <= {"t002", "t004", "t006", "t008", "t010"}  # 인터뷰어 턴은 반영 안 됨
