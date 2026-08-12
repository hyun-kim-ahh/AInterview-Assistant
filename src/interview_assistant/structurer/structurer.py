"""
Structurer — 전사 이벤트를 KnowledgeState로 구조화 (dev-plan 8단계 LLM 투입, 이후
"스키마 커버리지 확인"에서 "성장하는 문서 아웃라인 작성"으로 확장).

(2026-08-05, 재종합 전환) 턴(들)을 기존 섹션에 매칭하거나 미분류로 버리는 델타
분류 방식에서, 매 구조화 틱마다 현재 전체 상태 + 이번 배치를 보고 영향받는 섹션의
summary를 처음부터 다시 쓰는 방식으로 전환했다(synthesize 콜백, 기본값은 OpenRouter
기반 structure_synthesizer.synthesize_structure) — "분류/미분류" 구분 자체를
없애달라는 요청. 새 섹션의 item_id는 LLM이 아니라 여기서 dyn_NNN으로 안전하게
부여한다(할루시네이션/충돌 방지). 화자 라벨 없이도 동작해야 하므로(speaker=None),
확실히 인터뷰어인 경우만 제외하고 나머지는 전부 반영을 시도한다. synthesize/propose는
테스트에서 실제 API 호출 없이 주입할 수 있도록 생성자/메서드 인자로 분리했다.

(2026-08-10 추가) 그때 남겨뒀던 코드 레벨 미분류 안전망(질문도 아니고 어떤 섹션
근거로도 안 쓰인 턴을 원문 그대로 KnowledgeState.unclassified_summary에 붙이던
것)을 완전히 제거했다 — app/web.py의 구조화 배치가 이제 최근 N턴을 룩백으로
다시 보여주게 되면서, 이미 섹션에 반영된 턴이 룩백으로 다시 들어왔을 때 이번엔
안 인용됐다는 이유로 안전망이 같은 내용을 미분류에 중복으로 다시 붙이는 부작용이
생길 걸 발견 — 그 가드를 추가하는 대신 사용자가 "미분류 자체를 빼자"고 결정.
이제 질문도 아니고 어떤 섹션 근거로도 안 쓰인 턴은 그냥 구조화 결과에 안 남는다
(원문은 transcript.jsonl에 그대로 있음).
"""

from __future__ import annotations

from collections.abc import Callable

from interview_assistant.config import MAX_DYNAMIC_ITEMS, MAX_RENAMES_PER_ITEM
from interview_assistant.contracts import (
    InterviewSchema,
    KnowledgeState,
    SchemaItem,
    Speaker,
    SessionContext,
    TranscriptEvent,
)
from interview_assistant.structurer.structure_synthesizer import (
    RenameProposal,
    SynthesisResult,
    synthesize_structure,
)
from interview_assistant.structurer.outline_seeder import propose_initial_sections

SynthesizeFn = Callable[
    [list[TranscriptEvent], InterviewSchema, SessionContext, KnowledgeState],
    SynthesisResult,
]
ProposeFn = Callable[[InterviewSchema, SessionContext], list[str]]


class Structurer:
    """TranscriptEvent를 받아 KnowledgeState를 증분 갱신한다."""

    def __init__(
        self,
        schema: InterviewSchema,
        session_context: SessionContext,
        synthesize: SynthesizeFn = synthesize_structure,
    ) -> None:
        self._schema = schema
        self._session_context = session_context
        self._synthesize = synthesize
        self._next_dynamic_seq = 1
        self._rename_counts: dict[str, int] = {}  # 폭주(플립플롭) 방지 카운터 — item_id별
        self._state = KnowledgeState(
            session_id=session_context.session_id,
            schema_items=[
                SchemaItem(item_id=d.item_id, label=d.label) for d in schema.items
            ],
        )

    @property
    def state(self) -> KnowledgeState:
        return self._state

    def _new_dynamic_id(self) -> str:
        item_id = f"dyn_{self._next_dynamic_seq:03d}"
        self._next_dynamic_seq += 1
        return item_id

    def seed_outline(self, *, propose: ProposeFn = propose_initial_sections) -> None:
        """세션 시작 시 1회 — 나올 법한 주제를 미리 섹션 제목으로 채워둔다.

        propose()가 실패해도(네트워크 오류 등) 세션 시작을 막으면 안 되므로 예외를
        삼킨다 — 실패하면 그냥 미리 채워지는 섹션이 없을 뿐, 세션 자체엔 지장 없다.
        """
        try:
            proposed_labels = propose(self._schema, self._session_context)
        except Exception:
            return
        existing = {item.label for item in self._state.schema_items}
        for label in proposed_labels:
            label = label.strip()
            if not label or label in existing:
                continue
            self._state.schema_items.append(SchemaItem(item_id=self._new_dynamic_id(), label=label))
            existing.add(label)

    def _apply_renames(self, renames: list[RenameProposal]) -> None:
        if not renames:
            return
        items_by_id = {item.item_id: item for item in self._state.schema_items}
        for proposal in renames:
            item = items_by_id.get(proposal.item_id)
            if item is None:
                continue  # 존재하지 않는 item_id — 기존 할루시네이션 방어와 동일 패턴
            new_label = proposal.new_label.strip()
            if not new_label or new_label == item.label:
                continue
            if any(other is not item and other.label == new_label for other in self._state.schema_items):
                continue  # 라벨 충돌 방어 — new_sections의 라벨 유일성 불변식과 일관
            if self._rename_counts.get(item.item_id, 0) >= MAX_RENAMES_PER_ITEM:
                continue  # 폭주 방지 — 내용은 그대로, 제목만 더 이상 안 바뀜
            item.label = new_label
            self._rename_counts[item.item_id] = self._rename_counts.get(item.item_id, 0) + 1

    def ingest_batch(self, events: list[TranscriptEvent]) -> None:
        events = [e for e in events if e.speaker != Speaker.INTERVIEWER]
        if not events:
            return  # 화자 메타데이터로 이미 확실한 턴만 있었던 배치 — 호출 자체를 생략

        result = self._synthesize(events, self._schema, self._session_context, self._state)

        # 섹션 갱신/새 섹션 여부와 무관하게 rename은 항상 먼저 적용.
        self._apply_renames(result.renames)

        valid_turn_ids = {e.turn_id for e in events}
        # 화자를 몰라도(실 STT는 speaker가 항상 None) 발화 내용 자체가 질문이라 담긴
        # 정보가 없다고 판단된 turn_id들 — 할루시네이션 방어로 이번 배치에 실제로
        # 있는 turn_id만 인정한다.
        question_turn_ids = {t for t in result.question_turn_ids if t in valid_turn_ids}

        items_by_id = {item.item_id: item for item in self._state.schema_items}
        for upd in result.sections:
            item = items_by_id.get(upd.item_id)
            if item is None:
                continue  # 존재하지 않는 item_id — 할루시네이션 방어
            # 순수 질문 turn_id는 근거로 인정하지 않는다 — 모델이 question_turn_ids와
            # new_refs를 동시에(불일치) 채워도 코드 레벨에서 무시.
            refs = [r for r in upd.new_refs if r in valid_turn_ids and r not in question_turn_ids]
            if not refs:
                continue  # refs 전부 할루시네이션이거나 질문 turn_id뿐 — 이 갱신 자체를 버림
            if not upd.summary:
                continue  # 요약이 비면 통째로 버림 — 안 그러면 이미 있던 내용을 빈 문자열로 덮어씀
            item.summary = upd.summary
            item.status = upd.status
            item.source_refs.extend(r for r in refs if r not in item.source_refs)

            known_refs = {c.with_ref for c in item.contradictions}
            # with_ref 검증 범위는 이번 배치가 아니라 이 섹션이 지금까지 누적한 전체
            # turn_id — 모순은 몇 틱 전 발언과도 생길 수 있으므로.
            all_known_turn_ids = set(item.source_refs)
            resolved = [r for r in upd.resolved_contradiction_refs if r in known_refs]
            item.contradictions = [c for c in item.contradictions if c.with_ref not in resolved]
            item.contradictions.extend(
                c for c in upd.new_contradictions
                if c.with_ref in all_known_turn_ids and c.with_ref not in known_refs
            )

        items_by_label = {item.label: item for item in self._state.schema_items}
        for proposal in result.new_sections:
            refs = [r for r in proposal.refs if r in valid_turn_ids and r not in question_turn_ids]
            if not refs:
                continue
            if not proposal.summary:
                continue  # 요약 없는 섹션 제안은 의미가 없음(기존 섹션 덮어쓰기 방어와 같은 이유)
            label = proposal.label.strip()
            existing_item = items_by_label.get(label)
            if existing_item is not None:
                # 라벨이 기존 섹션과 완전히 같음 — 새로 안 만들고 그 섹션에 반영
                # (프롬프트에 이미 지시했지만, 코드 레벨 백스톱).
                existing_item.summary = proposal.summary
                existing_item.status = proposal.status
                existing_item.source_refs.extend(r for r in refs if r not in existing_item.source_refs)
            elif len(self._state.schema_items) >= MAX_DYNAMIC_ITEMS:
                continue  # 폭주 방지 — 이 턴들은 이번엔 반영되지 않음
            else:
                new_item = SchemaItem(
                    item_id=self._new_dynamic_id(),
                    label=label,
                    summary=proposal.summary,
                    status=proposal.status,
                    source_refs=list(refs),
                )
                self._state.schema_items.append(new_item)
                items_by_label[label] = new_item
