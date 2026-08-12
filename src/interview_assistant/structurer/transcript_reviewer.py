"""
전사 재검토(신규) — 전사된 글자는 "퓨어 데이터"가 아니라 STT가 실시간으로 받아적은
것일 뿐이다. 뒤에 나온 턴의 내용으로 미뤄봤을 때 앞선 턴이 오인식됐다고 강하게
확신되면 그 턴의 원문 텍스트 자체를 교정한다.

structure_synthesizer.synthesize_structure(구조화 틱마다 이번 배치 + 현재 상태를 봄)와는
근본적으로 다른 관심사(원문 텍스트 자체의 진위를 여러 턴에 걸쳐 판단)라 파일을
분리했다 — outline_seeder.py가 "세션 시작 시 1회"라는 별개 관심사로 분리된 것과
같은 전례. 구조화 틱(사용자가 고른 주기)마다 딱 1회만 호출된다(턴마다가 아님) —
app/web.py가 Structurer.ingest_batch와 나란히(그것을 거치지 않고) 직접 호출하는
것과 같은 패턴으로, 이 모듈도 Structurer를 거치지 않고 app/web.py가 직접 호출한다.
Structurer는 원래 배치 하나씩만 알고 전체 전사 목록을 모르는 게 설계이므로, 그
경계를 지키기 위해 여기서 완전히 분리했다.

교정은 매우 보수적으로만 제안돼야 한다 — 코드 레벨 방어(confidence 임계값, 존재
않는 turn_id 방어)는 app/web.py의 _apply_transcript_corrections가 담당한다.
"""

from __future__ import annotations

import json

from openai import OpenAI

from interview_assistant.config import TRANSCRIPT_REVIEW_WINDOW_SECONDS, get_model
from interview_assistant.contracts import (
    InterviewSchema,
    SessionContext,
    TranscriptCorrection,
    TranscriptEvent,
)
from interview_assistant.llm_client import STRUCTURED_EXTRA_BODY, get_client, structured_response_format

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_id": {"type": "string"},
                    "corrected_text": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["turn_id", "corrected_text", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["corrections"],
    "additionalProperties": False,
}


def _window_by_time(
    recent_events: list[TranscriptEvent], window_seconds: float
) -> list[TranscriptEvent]:
    """recent_events는 시간순(append-only)이라고 가정 — question_engine.py의 동일
    이름 함수와 같은 로직이지만, 모듈끼리 서로의 private 헬퍼를 import하지 않는다는
    원칙 때문에 별도로 둔다(작은 5줄짜리 중복은 이 프로젝트가 이미 감수하는 수준)."""
    if not recent_events:
        return []
    cutoff = recent_events[-1].timestamp - window_seconds
    return [e for e in recent_events if e.timestamp >= cutoff]


def _build_system_prompt(schema: InterviewSchema, session_context: SessionContext) -> str:
    return (
        "당신은 STT(음성인식)로 자동 전사된 인터뷰 원문을 검토하는 교정자입니다. "
        "아래 [최근 전사]는 완벽하지 않습니다 — 사람 발화를 기계가 실시간으로 받아적은 "
        "것이라, 발음이 비슷한 다른 단어로 잘못 인식되거나 오탈자가 섞여 있을 수 있습니다.\n\n"
        f"도메인: {schema.domain}\n"
        f"이번 인터뷰 목적: {session_context.interview_goal}\n\n"
        "뒤에 나온 다른 턴의 내용을 보면 앞선 어느 턴이 실제로는 다른 단어였을 "
        "가능성이 아주 강하게 확신될 때만, corrections에 그 턴의 turn_id와 고친 문장 "
        "전체(corrected_text)를 담으세요. 전형적인 예: 도메인 전문용어가 발음이 비슷한 "
        "일반 단어로 잘못 적혔다는 게 뒤 문맥으로 명백히 확인되는 경우.\n\n"
        "절대 하지 말아야 할 것:\n"
        "- 조금이라도 애매하면 손대지 마세요 — 대부분의 턴은 고칠 게 없는 게 정상입니다.\n"
        "- 이미 뜻이 통하는 문장을 '더 매끄럽게' 다듬거나 말투를 바꾸지 마세요 — 이건 "
        "교정이지 윤문이 아닙니다.\n"
        "- 뒤에 나온 다른 발언과 명백히 어긋나거나 그 도메인에서 말이 안 되는 단어라는 "
        "강한 근거 없이, 그냥 '이렇게 말했을 것 같다'는 추측만으로 고치지 마세요.\n"
        "- 원문에 없는 내용을 새로 지어내 채우지 마세요 — 실제로 잘못 받아적힌 부분만 "
        "고치고 전체 의미는 바꾸지 마세요.\n"
        "- 화자가 망설이거나 말을 바꾸거나 어색하게 말한 건 오인식이 아닙니다 — 자연스러운 "
        "발화는 그대로 두세요.\n\n"
        "confidence는 그 정정이 맞다는 확신도(0~1)이고, 정말 확실한 경우에만 0.9 이상을 "
        "주세요 — 조금이라도 불확실하면 그 항목을 아예 corrections에 넣지 마세요. reason에는 "
        "어느 뒤 턴의 어떤 내용 때문에 그렇게 판단했는지 한 문장으로 적으세요 — 이유를 한 "
        "문장으로 설명할 수 없다면 애초에 넣지 마세요. 한 번에 corrections는 최대 2개까지만 "
        "제안하세요(그보다 많다면 확신이 부족한 겁니다). 고칠 게 없으면(대부분의 경우) 빈 "
        "배열을 반환하세요."
    )


def _build_user_message(windowed: list[TranscriptEvent]) -> str:
    return "\n".join(f"[{e.turn_id}] {e.text}" for e in windowed)


def review_transcript(
    recent_events: list[TranscriptEvent],
    schema: InterviewSchema,
    session_context: SessionContext,
    *,
    client: OpenAI | None = None,
) -> list[TranscriptCorrection]:
    windowed = _window_by_time(recent_events, TRANSCRIPT_REVIEW_WINDOW_SECONDS)
    if len(windowed) < 2:
        return []  # 비교할 다른 턴이 없으면 LLM 호출 자체를 안 함

    client = client or get_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": _build_system_prompt(schema, session_context)},
            {"role": "user", "content": _build_user_message(windowed)},
        ],
        response_format=structured_response_format("transcript_corrections", _RESPONSE_SCHEMA),
        extra_body=STRUCTURED_EXTRA_BODY,
    )
    data = json.loads(response.choices[0].message.content)
    valid_turn_ids = {e.turn_id for e in windowed}
    return [
        TranscriptCorrection(
            turn_id=c["turn_id"],
            corrected_text=c["corrected_text"],
            confidence=max(0.0, min(1.0, c["confidence"])),
            reason=c["reason"],
        )
        for c in data["corrections"]
        if c["turn_id"] in valid_turn_ids  # 존재하지 않는 turn_id — 할루시네이션 방어
    ]
