"""
질문 생성 트리거 판단(신규) — "지금이 추천 질문을 생성할 때인가"를 정성적으로
판단한다. 1분 주기 백그라운드 루프(app/web.py의 _question_loop)가 "새 턴이 있으면
무조건 재생성"하던 걸 대체한다 — 진짜 사람 인터뷰어가 듣다가 "이건 물어봐야겠다"/
"아직 때가 아니다"를 판단하는 것처럼, 인터뷰 진행 정도·목적·포커스를 함께
고려해서 판단한다.

generate_questions(질문 "내용"을 만듦)와는 관심사가 다르다(이건 "지금인가"만
판단하고, 판단에는 경과 시간·턴 수·커버리지처럼 더 넓은 컨텍스트가 필요함) —
그래서 같은 question_engine 패키지 안에 있지만 별도 파일로 분리했다
(structurer 패키지 안에서 outline_seeder.py가 structurer.py와 분리된 것과 같은
관심사별 분리 전례).

(2026-08-06 추가) 판단 근거로 KnowledgeState 전체를 인자로 받고 있었으면서도
실제로는 "N/M개 섹션이 다각도로 다뤄짐" 카운트 한 줄만 프롬프트에 흘려주고
있었음(사용자가 배경 트리거가 잘 안 뜬다고 느껴 현재 로직을 설명하다가 발견) —
각 섹션이 실제로 뭘 다뤘는지, 전체 종합(overview)이 뭐라고 정리됐는지는 판단
근거에 전혀 없었던 것. `_build_user_message`가 이제 `question_engine.py`의
동명 함수와 같은 모양(섹션별 요약/모순 + 미분류 요약)으로 전체 지식 상태를
[최근 대화] 앞에 함께 넣는다 — `_coverage_summary`는 삭제.

(2026-08-06 추가, 위와 같은 날) `state.overview`는 결국 다시 뺐다 — 실시간
구조화(`structure_synthesizer.py`)가 더 이상 매 틱마다 overview를 안 채우기로
바뀌어서(그 판단에 쓰이지도 않던 필드를 매번 다시 쓰던 낭비였음), 진행 중에는
`state.overview`가 항상 빈 문자열이다. `/end` 이후에야 `write_final_document`가
채우는데, 그 시점엔 `_question_loop`가 이미 멈춰 `judge_trigger`가 다시 불릴 일이
없다 — 그래서 여기 포함시켜도 항상 죽은 코드였을 것.

(2026-08-10 추가) `_build_user_message`의 "미분류" 블록도 삭제했다 — KnowledgeState
쪽에서 "미분류" 개념 자체가 완전히 없어졌다(structure_synthesizer.py/structurer.py
참고). 이제 섹션별 요약/모순만 지식 상태로 넘긴다.

(2026-08-10 추가, 트리거가 너무 보수적이라는 피드백) `_build_system_prompt`의 "초반이면
끼어들 필요 없다" 기준 뒤에 "대부분의 판단 시점에는 그대로 두는 게 맞습니다"라는
절이 붙어있었는데, 이게 "초반" 조건에 안 묶이고 판단 시점 전체에 대한 일반론으로
읽혀서 초반이 지난 뒤에도 모델이 트리거를 계속 억제하는 원인이 됐다. 조건에
명시적으로 종속시키는 문장으로 교체.

(2026-08-11 추가) 트리거 기준을 하나 더 넓혔다 — 기존엔 "설명을 흐리거나
궁금증을 유발했는데 안 파고들어진 지점"만 트리거 이유였는데, 그 정도로 흐름이
끊긴 게 아니어도 흥미로운 지점(구체적 사례·특이한 판단 등)이 있으면 그것도
트리거할 이유가 되도록 기준 문장 추가. 판단이 보는 컨텍스트(최근
TRIGGER_JUDGE_WINDOW_SECONDS초 창)는 안 바꿨으므로, "나중에"는 사실상 이 창이
아직 살아있는 동안으로 제한된다 — 그 창을 넘어서도 기억해두는 것까지는 이번
범위 밖(더 큰 별도 작업).

(2026-08-11 추가, 같은 날 두 번째) "애매하면 트리거하지 마세요 — 너무 자주
트리거하면 인터뷰 흐름을 방해합니다" tie-break 문장을 완전히 삭제했다 —
사용자가 트리거가 너무 안 뜨는 것 같다며 이 문장이 원인으로 보인다고 지적.
(35)에서 스코프 누수(초반 기준이 전체로 읽히는 문제)를 고친 뒤에도 여전히
보수적이라는 피드백이 이어진 것 — 이 문장 자체가 "애매하면 억제"라는 보수적
편향을 명시적으로 지시하고 있었으므로 이번엔 아예 제거.
"""

from __future__ import annotations

import json

from openai import OpenAI

from interview_assistant.config import TRIGGER_JUDGE_MAX_TOKENS, TRIGGER_JUDGE_WINDOW_SECONDS, get_model
from interview_assistant.contracts import (
    KnowledgeState,
    SessionContext,
    TranscriptEvent,
    TriggerDecision,
)
from interview_assistant.llm_client import STRUCTURED_EXTRA_BODY, get_client, structured_response_format

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "should_trigger": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["should_trigger", "reason"],
    "additionalProperties": False,
}


def _window_by_time(
    recent_events: list[TranscriptEvent], window_seconds: float
) -> list[TranscriptEvent]:
    """recent_events는 시간순(append-only)이라고 가정 — question_engine.py의 동일
    이름 함수와 같은 로직이지만, 모듈끼리 서로의 private 헬퍼를 import하지 않는다는
    원칙 때문에 별도로 둔다(transcript_reviewer.py에서도 같은 이유로 중복해둔 전례)."""
    if not recent_events:
        return []
    cutoff = recent_events[-1].timestamp - window_seconds
    return [e for e in recent_events if e.timestamp >= cutoff]


def _build_system_prompt(
    session_context: SessionContext,
    turn_count: int,
    elapsed_seconds: float,
) -> str:
    elapsed_minutes = elapsed_seconds / 60
    return (
        "당신은 지금 진행 중인 전문가 인터뷰를 곁에서 지켜보는 노련한 인터뷰 코디네이터입니다. "
        "질문 내용을 만드는 게 아니라, 인터뷰어에게 '지금 추천 질문을 보여줄 때인가'만 "
        "판단하는 역할입니다. 아래 [지식 상태]와 [최근 대화]를 참고해 판단하세요.\n\n"
        f"이번 인터뷰 목적: {session_context.interview_goal}\n"
        f"포커스: {session_context.focus_notes}\n"
        f"지금까지 진행: 총 {turn_count}턴, 약 {elapsed_minutes:.1f}분 경과\n\n"
        "판단 기준:\n"
        "- 인터뷰가 아직 초반이라 전문가가 이야기를 막 풀어놓는 단계라면, 굳이 끼어들 "
        "필요 없습니다. (이 기준은 초반에만 적용됩니다 — 초반이 지났다면 이 이유로 "
        "트리거를 미루지 마세요.)\n"
        "- 아래 [최근 대화]에서 전문가가 설명을 흐리거나(\"그건 나중에\", \"일단 넘어가고\") "
        "궁금증을 유발할 만한 발언을 했는데 더 파고들어지지 않은 지점이 있다면, 지금이 "
        "물어볼 때입니다.\n"
        "- 그 정도로 흐름이 끊긴 게 아니어도, 구체적인 사례나 특이한 판단처럼 나중에 "
        "더 물어보면 좋겠다 싶은 흥미로운 지점이 있었다면 그것도 트리거할 이유가 됩니다.\n"
        "- 아래 [지식 상태]에 이미 충분히 다뤄진 내용을 또 캐물을 필요는 없습니다.\n\n"
        "should_trigger가 true면 reason에 어떤 발언 때문에 지금이라고 판단했는지 한 문장, "
        "false면 reason은 빈 문자열로 두세요."
    )


def _build_user_message(state: KnowledgeState, windowed: list[TranscriptEvent]) -> str:
    lines = ["[지식 상태]"]
    for item in state.schema_items:
        lines.append(f"- {item.item_id}: {item.label} (status: {item.status.value})")
        if item.summary:
            lines.append(f"    {item.summary}")
        for contra in item.contradictions:
            lines.append(f"    ⚠ 모순: {contra.with_ref}와 상충 — {contra.note}")

    lines.append("\n[최근 대화]")
    if not windowed:
        lines.append("(없음)")
    for e in windowed:
        lines.append(f"[{e.turn_id}] {e.text}")
    return "\n".join(lines)


def judge_trigger(
    state: KnowledgeState,
    recent_events: list[TranscriptEvent],
    session_context: SessionContext,
    elapsed_seconds: float,
    *,
    client: OpenAI | None = None,
) -> TriggerDecision:
    windowed = _window_by_time(recent_events, TRIGGER_JUDGE_WINDOW_SECONDS)
    if len(windowed) < 2:
        return TriggerDecision(should_trigger=False)  # 판단할 대화 자체가 부족 — 호출 스킵

    client = client or get_client()
    response = client.chat.completions.create(
        model=get_model(),
        max_tokens=TRIGGER_JUDGE_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": _build_system_prompt(session_context, len(recent_events), elapsed_seconds),
            },
            {"role": "user", "content": _build_user_message(state, windowed)},
        ],
        response_format=structured_response_format("trigger_decision", _RESPONSE_SCHEMA),
        extra_body=STRUCTURED_EXTRA_BODY,
    )
    data = json.loads(response.choices[0].message.content)
    return TriggerDecision(should_trigger=data["should_trigger"], reason=data["reason"])
