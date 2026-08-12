"""
인터뷰 종합 정리 문서 작성 — "인터뷰 종료" 시 app/web.py가 딱 1회 호출한다.

(2026-08-05 추가) 진행 중 구조화(`structure_synthesizer.py`)가 이제 매 틱마다 이미
섹션별 summary/overview/unclassified_summary를 계속 재종합하고 있으므로, 이 파일의
역할은 "처음 종합"에서 "마지막 다듬기"로 바뀌었다 — 진행 중 결과를 입력으로 받아
전체를 한 번 더 참고해 일관성을 점검하고 다듬는다. 그래도 여전히 섹션마다 따로
호출하지 않고 한 번에 다 보여주고 한 호출로 받는다(전체를 참고한 종합이 섹션별
호출로는 안 되므로). Structurer를 거치지 않고 app/web.py가 직접 호출한다
(transcript_reviewer와 같은 전례) — Structurer는 턴 배치만 알고 "마지막 다듬기"라는
별개 관심사는 설계 밖이므로, 그 경계를 지키기 위해 여기서 완전히 분리했다.

(2026-08-04, v0.13) 처음엔 각 필드를 줄글 문단(문자열)으로 받았는데, 가독성이
떨어져서 짧은 요점(bullet) 리스트로 바꿨었다. (v0.14) 그런데 표·헤딩·중첩 구조는
배열로 표현할 수 없어서, 다시 문자열로 되돌리고 대신 그 문자열이 실제 마크다운
문서(표/헤딩/목록 포함 가능)가 되도록 프롬프트를 강화했다 — app/web.py가
`markdown` 패키지로 HTML 렌더링해서 보여준다.

(2026-08-10 추가) `unclassified_summary` 필드를 완전히 제거했다 — KnowledgeState
쪽에서 "미분류" 개념 자체가 없어졌다(structure_synthesizer.py/structurer.py 참고).
이제 이 파일은 overview + 섹션별 summary 두 가지만 다듬는다.

(2026-08-10 추가) `_build_system_prompt`에 수치·명칭 같은 구체적 정보를 정확히
보존하라는 지시를 추가했다 — "마지막 다듬기"로 섹션 요약을 한 번 더 다시 쓰는
과정에서 그런 디테일이 뭉개질 위험이 있음(structure_synthesizer.py에도 같은
이유로 동일 지시 추가).

(2026-08-11 추가) 이 호출이 쓰는 모델을 get_model()(경량)에서
get_structuring_model()(고성능, anthropic/claude-sonnet-5)로 바꿨다 — 인터뷰
전체를 참고해 통째로 다시 쓰는 "마지막 다듬기"라 실시간 구조화
(structure_synthesizer.py)·초기 섹션 제안(outline_seeder.py)과 같은 급으로
묶어 고성능 모델을 쓰기로 함(config.py 참고).

(2026-08-11 추가) 위 모델 전환 뒤 structure_synthesizer.py에서 실사용 중 발견된
두 문제(수치를 "(특정) 퍼센트"처럼 얼버무림, summary에 [t009]류 turn_id 인라인
인용)를 이 파일도 같은 모델·같은 "생략 금지" 지시를 쓰고 있어 그대로 안고 있을
수 있다고 보고 선제적으로 같은 금지 문장을 추가했다 — 이 파일에서 실제로 이
문제가 보고된 적은 없음(자세한 내용은 structure_synthesizer.py의 동일 날짜
노트 참고).
"""

from __future__ import annotations

import json

from openai import OpenAI

from interview_assistant.config import FINAL_DOCUMENT_MAX_TOKENS, get_structuring_model
from interview_assistant.contracts import (
    FinalDocument,
    InterviewSchema,
    KnowledgeState,
    SessionContext,
)
from interview_assistant.llm_client import STRUCTURED_EXTRA_BODY, get_client, structured_response_format

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["item_id", "summary"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["overview", "sections"],
    "additionalProperties": False,
}


def _build_system_prompt(schema: InterviewSchema, session_context: SessionContext) -> str:
    return (
        "당신은 방금 끝난 전문가 인터뷰의 기록을 마지막으로 다듬는 편집자입니다. 아래 "
        "[지식 상태]에는 인터뷰 도중 계속 재종합되며 다듬어진 섹션별 요약이 담겨 "
        "있습니다. 이미 잘 정리돼 있을 "
        "가능성이 높으니, 억지로 다 뜯어고치지 말고 전체를 한 번에 참고해 일관성을 "
        "점검하고 다듬으세요.\n\n"
        f"도메인: {schema.domain}\n"
        f"이번 인터뷰 목적: {session_context.interview_goal}\n\n"
        "다음 두 가지를 전부 Markdown 형식의 텍스트로 작성하세요(제목은 ### 이하만 "
        "쓰세요 — 화면에 이미 더 큰 제목이 있습니다). 단순히 사실 문장을 보기 좋게 "
        "나열하는 게 아니라, 실제로 참고하고 활용하기 좋은 잘 정리된 문서처럼 작성해야 "
        "합니다 — 필요하면 부제목으로 하위 구조를 나누고, 사실을 나열한 뒤에는 그게 "
        "왜 중요한지·어떻게 쓰이는지에 대한 짧은 설명을 덧붙이세요. 여러 항목·조건·"
        "케이스를 비교해야 할 때는 Markdown 표를 적극 활용하세요(예: 전체 개요에서 "
        "섹션들을 한눈에 비교하는 표, 섹션 안에서 여러 조건을 나란히 비교하는 표). "
        "문단 사이는 빈 줄로 구분하세요.\n\n"
        "1) overview — 인터뷰 전체를 종합한 개요. 개별 섹션을 따로따로 보지 말고 전체를 "
        "한 번에 참고해서, 핵심 주제·전반적인 그림을 정리하세요. 특정 섹션이 전혀 "
        "다뤄지지 않았다면 그 사실도 언급하세요.\n"
        "2) sections — 요약이 하나라도 있는 섹션마다, 지금 요약을 전체 맥락(다른 섹션·"
        "개요와의 일관성)까지 참고해 마지막으로 한 번 더 다듬어 작성하세요. item_id는 "
        "반드시 [지식 상태]에 나온 것 중에서만 그대로 골라 쓰세요(새로 만들지 마세요). "
        "이미 잘 정리된 내용은 그대로 유지하고, 다른 섹션과 중복되거나 성격이 다른 "
        "내용은 부제목이나 표로 구분하세요. 모순 메모가 있으면 그것도 포함하세요.\n\n"
        "반드시 지켜야 할 것: 주어진 요약에 실제로 담긴 내용만 근거로 삼으세요 — 없는 "
        "내용을 지어내거나 추측으로 채우지 마세요. 표나 구조를 억지로 만들지 마세요 — "
        "실제로 비교할 게 여러 개 있을 때만 표를 쓰고, 그렇지 않으면 문단이나 목록으로 "
        "충분합니다. 다듬는 과정에서 원래 요약에 있던 수치·단위·날짜·고유명사 등 "
        "구체적인 정보를 절대 누락하거나 뭉뚱그리지 마세요. 수치를 뭉뚱그리는 대표적인 "
        "실수: 원래 요약에 '22퍼센트'처럼 정확한 숫자가 있는데 다듬으면서 '(특정) "
        "퍼센트'나 '일정 비율'처럼 바꾸는 것 — 이런 표현은 금지합니다. 원래 있던 숫자는 "
        "그대로 유지하세요. 텍스트 안에 [t009]처럼 turn_id를 대괄호로 인용하지도 "
        "마세요 — 이 문서엔 출처 표시 필드가 없으니, 인용 없이 자연스러운 문장으로만 "
        "쓰세요."
    )


def _build_user_message(state: KnowledgeState) -> str:
    lines = ["[지식 상태]"]
    for item in state.schema_items:
        lines.append(f"\n[섹션 {item.item_id}] {item.label}")
        lines.append(item.summary if item.summary else "(아직 내용 없음)")
        for contra in item.contradictions:
            lines.append(f"  ⚠ 모순: {contra.with_ref}와 상충 — {contra.note}")
    return "\n".join(lines)


def write_final_document(
    state: KnowledgeState,
    schema: InterviewSchema,
    session_context: SessionContext,
    *,
    client: OpenAI | None = None,
) -> FinalDocument:
    if not any(item.summary for item in state.schema_items):
        return FinalDocument()  # 다듬을 내용 자체가 없으면 호출 자체를 스킵

    client = client or get_client()
    response = client.chat.completions.create(
        model=get_structuring_model(),
        messages=[
            {"role": "system", "content": _build_system_prompt(schema, session_context)},
            {"role": "user", "content": _build_user_message(state)},
        ],
        response_format=structured_response_format("final_document", _RESPONSE_SCHEMA),
        extra_body=STRUCTURED_EXTRA_BODY,
        max_tokens=FINAL_DOCUMENT_MAX_TOKENS,
    )
    data = json.loads(response.choices[0].message.content)
    valid_ids = {item.item_id for item in state.schema_items}
    return FinalDocument(
        overview=data["overview"],
        section_summaries={
            s["item_id"]: s["summary"] for s in data["sections"] if s["item_id"] in valid_ids
        },
    )
