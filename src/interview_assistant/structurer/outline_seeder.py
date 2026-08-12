"""
초기 아웃라인 제안 (신규) — 세션 시작 시 한 번, 이번 인터뷰에서 나올 법한 주제를
미리 섹션 제목으로 제안한다("앞으로 나올 것 같은 내용도 미리 항목화, 제목이라도
달아두고" 요구사항). 스키마가 있든 없든 동작한다 — 있으면 그 항목들과 안 겹치는
추가 주제를, 없으면 처음부터 전체 아웃라인을 제안한다.

세션 생성을 블로킹하면 안 되므로(이 프로젝트의 핵심 원칙 — LLM 지연은 사용자 요청
경로에 절대 넣지 않는다, design.md P2) app/web.py가 별도 백그라운드 워커에서 1회
호출한다. structure_synthesizer.py(구조화 틱마다 계속 재종합)와는 별개 관심사(세션
시작 시 1회 vs 매 틱)라 파일을 분리했다 — transcript_reviewer.py가 같은 패키지
안에서 관심사별로 나뉜 것과 같은 전례.

(2026-08-11 추가) 이 호출이 쓰는 모델을 get_model()(경량)에서
get_structuring_model()(고성능, anthropic/claude-sonnet-5)로 바꿨다 — 지식 구조의
초기 골격을 만드는 호출이라 실시간 구조화(structure_synthesizer.py)·최종 정리
(final_document.py)와 같은 급으로 묶어 고성능 모델을 쓰기로 함(config.py 참고).
"""

from __future__ import annotations

import json

from openai import OpenAI

from interview_assistant.config import INITIAL_OUTLINE_MAX_SECTIONS, get_structuring_model
from interview_assistant.contracts import InterviewSchema, SessionContext
from interview_assistant.llm_client import STRUCTURED_EXTRA_BODY, get_client, structured_response_format

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def _build_system_prompt(
    schema: InterviewSchema, session_context: SessionContext, max_sections: int
) -> str:
    existing_desc = "\n".join(f"- {item.label}" for item in schema.items) or "(없음)"
    return (
        "당신은 전문가 인터뷰를 앞두고 예상 목차를 짜는 보조 도구입니다. 아래 정보를 "
        f"보고, 이번 인터뷰에서 다뤄질 것 같은 주제를 섹션 제목으로 최대 {max_sections}개 "
        "제안하세요.\n"
        f"도메인: {schema.domain}\n"
        f"이번 인터뷰 목적: {session_context.interview_goal}\n"
        f"전문가 프로필: {session_context.expert_profile}\n"
        f"포커스: {session_context.focus_notes}\n\n"
        f"이미 정의된 항목(겹치지 않게 새 제목만 제안하세요):\n{existing_desc}\n\n"
        "확신 없으면 억지로 채우지 말고 적은 개수만 반환해도 됩니다. \"기타\", \"추가 "
        "의견\" 같은 모호한 제목은 피하고, 그 자체로 뜻이 통하는 구체적인 제목만 "
        "제안하세요."
    )


def propose_initial_sections(
    schema: InterviewSchema,
    session_context: SessionContext,
    *,
    max_sections: int = INITIAL_OUTLINE_MAX_SECTIONS,
    client: OpenAI | None = None,
) -> list[str]:
    client = client or get_client()
    response = client.chat.completions.create(
        model=get_structuring_model(),
        messages=[
            {
                "role": "system",
                "content": _build_system_prompt(schema, session_context, max_sections),
            },
            {"role": "user", "content": "위 조건에 맞는 섹션 제목을 제안하세요."},
        ],
        response_format=structured_response_format("initial_sections", _RESPONSE_SCHEMA),
        extra_body=STRUCTURED_EXTRA_BODY,
    )
    data = json.loads(response.choices[0].message.content)
    return [label.strip() for label in data["labels"] if label.strip()][:max_sections]
