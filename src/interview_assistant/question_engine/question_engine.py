"""
Question Engine — LLM 투입 (dev-plan 9단계). 지식 상태 + 최근 턴 윈도우 + 세션 컨텍스트
(목적·포커스 노트)로 유형 태그 질문 3~4개를 생성한다.

모순확인(contradiction)은 opportunistic이다 — 12단계(Analyzer)부터 SchemaItem.contradictions가
실제로 채워지며 이 프롬프트에도 노출되지만(_build_user_message), 그와 별개로 이 단계는
LLM에게 KnowledgeState 전체를 보여줘서 스스로 눈에 띄는 모순을 알아채면 태그를 붙이게도
한다 — Analyzer가 아직 못 본(예: 이번 턴 방금 도착) 모순을 놓치지 않기 위한 안전망.

최근 턴 윈도잉은 호출자가 아니라 이 함수 스스로 한다("최근 턴 원문 윈도우만 프롬프트에
넣는다"는 제약은 프롬프트를 만드는 이 모듈 자신의 책임) — 나중에 생길 다른 호출자도
자동으로 안전해진다. 윈도잉 기준은 턴 개수가 아니라 시간(초)이다 — 구조화(Structurer)가
이제 사용자가 고른 주기로 배치 처리되면서(app/web.py의 _structuring_loop) 턴 발생
간격이 고르지 않을 수 있어, "최근 N턴"보다 "최근 N초"가 실제 인터뷰 맥락과 더 잘
맞는다. `generate_questions`는 인터뷰어가 "추천 질문 생성" 버튼을 누른 시점에 그
자리에서 호출되며(2026-07-30부터 — 토큰 효율 때문에 배경 상시 생성을 되돌림),
그때그때 `state`(구조화 결과가 담긴 공유 mutable 객체)를 그대로 읽으므로, "그 시점의
구조화 결과"는 별도 배선 없이 이미 반영된다.

(2026-08-05 추가) 선택적 `trigger_hint` — 백그라운드 주기 트리거는 이제
question_engine/trigger_judge.py가 "지금이 물어볼 때인가"를 먼저 판단하고, 그
판단의 이유를 이 함수에 힌트로 넘겨 실제로 그 지점을 파고드는 질문이 나오게 한다.
버튼(/ask)·인터뷰 종료(/end)는 판단 없이 항상 호출되므로 힌트가 없다(기본값 None).
"""

from __future__ import annotations

import json

from openai import OpenAI

from interview_assistant.contracts import (
    KnowledgeState,
    QuestionCandidate,
    QuestionCandidates,
    QuestionType,
    SessionContext,
    TranscriptEvent,
)
from interview_assistant.config import MAX_QUESTION_CANDIDATES, QUESTION_MAX_TOKENS, RECENT_WINDOW_SECONDS, get_model
from interview_assistant.llm_client import STRUCTURED_EXTRA_BODY, get_client, structured_response_format

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": [t.value for t in QuestionType]},
                    "text": {"type": "string"},
                    "target_item": {"type": "string"},
                    "refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["type", "text", "target_item", "refs"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def _build_system_prompt(
    session_context: SessionContext, has_schema: bool = True, trigger_hint: str | None = None
) -> str:
    gap_line = (
        "- gap(누락): 아직 다루지 않은 스키마 항목을 새로 여는 질문\n"
        if has_schema
        else ""
    )
    no_schema_note = (
        ""
        if has_schema
        else "이번 세션은 스키마 없이 자유 형식으로 진행되므로 gap 유형은 사용하지 마세요.\n\n"
    )
    hint_line = (
        f"\n지금이 물어볼 적절한 시점이라고 판단된 이유: {trigger_hint}\n"
        "이 지점을 파고드는 질문을 후보 중 하나로 포함하면 좋습니다(다른 유형 질문도 "
        "계속 섞어도 됩니다).\n"
        if trigger_hint
        else ""
    )
    return (
        "당신은 전문가 인터뷰를 진행 중인 인터뷰어를 돕는 보조 도구입니다. "
        "아래 지식 상태와 최근 대화를 보고, 인터뷰어가 다음에 물어볼 만한 후속 질문 "
        "후보를 3~4개 생성하세요.\n\n"
        f"이번 인터뷰 목적: {session_context.interview_goal}\n"
        f"포커스: {session_context.focus_notes}\n"
        f"{hint_line}\n"
        f"{no_schema_note}"
        "각 질문에는 다음 유형 중 하나를 태그로 붙이세요:\n"
        "- probe(파고들기): 방금 나온 답변의 근거·디테일을 더 캐묻는 질문\n"
        "- contradiction(모순확인): 지식 상태 안에서 실제로 서로 어긋나 보이는 발언이 "
        "있을 때만 그 지점을 짚는 질문. 억지로 모순을 지어내지 마세요.\n"
        f"{gap_line}"
        "- expand(확장): 지금까지 나온 내용에서 자연스럽게 파생되는 새 주제를 여는 질문\n\n"
        "질문은 인터뷰어가 그대로 입에 올릴 수 있는 자연스러운 한국어 구어체로 쓰세요. "
        "target_item은 관련 스키마 항목의 item_id이고, 해당 항목이 없으면 빈 문자열로 "
        "두세요. refs는 이 질문과 직접 관련된 원문 turn_id 목록이며, 특히 contradiction "
        "유형은 서로 어긋나는 두 turn_id를 반드시 포함하세요. 관련 원문이 없으면 빈 "
        "배열로 두세요."
    )


def _build_user_message(state: KnowledgeState, windowed_events: list[TranscriptEvent]) -> str:
    lines = ["[지식 상태]"]
    for item in state.schema_items:
        lines.append(f"- {item.item_id}: {item.label} (status: {item.status.value})")
        if item.summary:
            lines.append(f"    {item.summary}")
        for contra in item.contradictions:
            lines.append(f"    ⚠ 모순: {contra.with_ref}와 상충 — {contra.note}")

    lines.append("\n[최근 대화]")
    for event in windowed_events:
        lines.append(f"[{event.turn_id}] {event.text}")

    return "\n".join(lines)


def _window_by_time(
    recent_events: list[TranscriptEvent], window_seconds: float
) -> list[TranscriptEvent]:
    """recent_events는 시간순(append-only)이라고 가정 — 마지막 이벤트의 timestamp를
    '지금'으로 삼아 그로부터 window_seconds 이내인 이벤트만 남긴다."""
    if not recent_events:
        return []
    cutoff = recent_events[-1].timestamp - window_seconds
    return [e for e in recent_events if e.timestamp >= cutoff]


def generate_questions(
    state: KnowledgeState,
    recent_events: list[TranscriptEvent],
    session_context: SessionContext,
    *,
    trigger_hint: str | None = None,
    client: OpenAI | None = None,
) -> QuestionCandidates:
    client = client or get_client()
    windowed = _window_by_time(recent_events, RECENT_WINDOW_SECONDS)
    generated_at = windowed[-1].turn_id if windowed else ""

    response = client.chat.completions.create(
        model=get_model(),
        max_tokens=QUESTION_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": _build_system_prompt(
                    session_context, has_schema=bool(state.schema_items), trigger_hint=trigger_hint
                ),
            },
            {"role": "user", "content": _build_user_message(state, windowed)},
        ],
        response_format=structured_response_format("question_candidates", _RESPONSE_SCHEMA),
        extra_body=STRUCTURED_EXTRA_BODY,
    )
    data = json.loads(response.choices[0].message.content)
    candidates = [
        QuestionCandidate(
            type=QuestionType(c["type"]),
            text=c["text"],
            target_item=c["target_item"] or None,
            refs=c["refs"],
        )
        for c in data["candidates"]
    ]
    candidates.sort(key=lambda c: c.type != QuestionType.CONTRADICTION)  # 모순확인 후보를 앞으로
    return QuestionCandidates(generated_at=generated_at, candidates=candidates[:MAX_QUESTION_CANDIDATES])
