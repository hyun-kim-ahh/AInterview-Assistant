"""
실시간 구조화 재종합(2026-08-05, `structurer/llm_classifier.py` 대체). OpenRouter를
openai SDK(base_url 오버라이드)로 호출한다.

이전엔 턴(들)을 "기존 섹션에 매칭/새 섹션 제안/미분류"로 분류하는 델타 방식이었다.
사용자가 "이제 배치로 보니 분류·미분류 구분 자체가 필요 없다"고 판단해, 매 구조화
틱마다 (a) 지금까지의 전체 구조화 상태(각 섹션의 현재 summary) + (b) 이번 틱에 새로
쌓인 전사 배치를 함께 보고, 영향받는 섹션의 summary를 처음부터 다시 쓰는 방식으로
전환했다 — `structurer/final_document.py`가 인터뷰 종료 시 1회 하던 "전체를 참고한
종합"을 매 틱마다 하는 셈(그래서 그 파일의 어조를 많이 가져왔다). 분류 결과를
쌓아두는 것이 아니라 "최신 상태를 매번 다시 쓰는" 것이므로, 델타(matches/new_items)
대신 sections/new_sections가 그 섹션의 최신 summary/status를 통째로 반환한다.

커버리지(uncovered/partial/covered)와 모순 감지도 이번에 이 호출로 흡수됐다 — 예전엔
`analyzer/coverage.py`(캡처 타입 개수로 계산)와 `analyzer/contradiction.py`(섹션당
별도 LLM 호출)가 따로 있었지만, 이제 이 호출이 전체 맥락을 이미 보고 있으므로 같은
응답에 함께 담아 반환한다(두 모듈 다 삭제).

is_question(순수 질문이라 담긴 정보가 없는 turn_id 판단)은 배치 전환 때 만든
question_turn_ids를 그대로 재사용 — 분류 방식과 무관하게 여전히 필요한 판단.

SectionUpdate/NewSectionProposal/RenameProposal/SynthesisResult는 Structurer 내부에서만
쓰는 타입이라 contracts.py가 아니라 여기 둔다(모듈 간 공유 계약이 아님).

(2026-08-06 추가) `overview`(KnowledgeState의 종합 개요) 생성을 이 실시간 호출에서
뺐다 — "실시간 구조화가 뭘 보고 정리하냐"를 사용자에게 설명하다가, 이 호출도
`final_document.py`(인터뷰 종료 시 1회, 마지막 다듬기)도 `state.overview`(이전에
자신이 쓴 개요)를 입력으로 전혀 안 읽는다는 걸 발견함 — 즉 매 틱(5~10초)마다
"바뀐 게 없으면 그대로 반환하라"는 지시를 실제로 따를 방법 없이 인터뷰 전체를
종합하는 문단을 처음부터 다시 쓰고 있었을 뿐, 최종 정리 품질에는 어차피 아무
영향이 없었다(final_document.py가 섹션 요약만 보고 독립적으로 다시 씀). overview는
이제 `/end`의 `write_final_document`가 딱 1회 채운다 — `KnowledgeState.overview`
필드 자체와 화면 렌더링(진행 중엔 빈 문자열이라 "종합 정리" 박스가 자연히 안 뜸)은
무변경.

(2026-08-10 추가) 구조화 배치가 이제 이번 틱의 새 턴뿐 아니라 최근
`STRUCTURING_LOOKBACK_TURNS`턴도 다시 포함한다(app/web.py의 _run_structuring_pass)
— 이전 틱 경계에서 끊긴 발화가 이번 틱에야 의미가 분명해지는 경우를 위함.
그런데 이 룩백 때문에 "이미 섹션에 반영된 턴을 이번엔 안 인용했다"는 이유로
미분류 안전망이 같은 내용을 중복으로 다시 붙이는 새 부작용이 생길 걸 발견 —
사용자가 그 복잡도를 보고 "미분류 자체를 빼버리자"고 결정. `unclassified_
summary`/`unclassified_new_refs`를 완전히 제거했다 — 질문도 아니고 어떤 섹션
근거로도 안 쓰인 턴은 이제 그냥 구조화 결과 어디에도 안 남는다(원문은
transcript.jsonl에 그대로 있음). 룩백이 경계 문제를 상당 부분 완화해줘서
안전망의 필요성 자체가 줄었다는 판단.

(2026-08-10 추가, 트리거 프롬프트 점검 중 발견) `_build_system_prompt`에 수치·명칭
같은 구체적 정보를 정확히 보존하라는 지시가 없었다 — summary를 매 틱 처음부터
다시 쓰는 방식이라 그런 디테일이 재작성 과정에서 뭉개지거나 생략될 위험이 있음.
"반드시 지켜야 할 것" 문단에 한 문장 추가(같은 이유로 final_document.py에도 추가).

(2026-08-11 추가) 이 호출이 쓰는 모델을 get_model()(경량)에서
get_structuring_model()(고성능, anthropic/claude-sonnet-5)로 바꿨다 — 지식 구조를
매 틱 통째로 다시 쓰는 핵심 호출이라 최종 정리(final_document.py)·초기 섹션
제안(outline_seeder.py)과 같은 급으로 묶어 고성능 모델을 쓰기로 함(config.py 참고).

(2026-08-11 추가, sonnet-5 전환 후 실사용에서 발견) 위 모델 전환 뒤 두 가지 문제가
생김: ①"수치·단위 등은 생략·뭉뚱그리지 말라"는 지시가 이미 있는데도 "22퍼센트"
같은 정확한 숫자를 "(특정) 퍼센트"처럼 얼버무리는 경우가 있었음 — 일반론만으로는
막지 못해 실제 실패 패턴을 반례로 명시. ②summary 문장 안에 [t009]처럼 turn_id를
대괄호로 인용하는 경우가 있었음 — 이런 지시를 준 적이 없는데도 `_build_user_message`가
입력에 쓰는 `[turn_id] text` 표기를 모델이 출력 스타일로 모방한 것으로 추정.
`_build_system_prompt`의 "반드시 지켜야 할 것" 문단에 두 문제를 각각 구체적으로
금지하는 문장 추가(같은 이유로 final_document.py에도 추가). temperature 조정은
프롬프트 수정과 원인을 분리하기 위해 이번엔 보류.

(2026-08-12 추가) summary 본문에 "~에 대한 설명이 이어졌습니다", "~는 아직
언급되지 않았습니다"처럼 인터뷰 진행 상황 자체를 서술하는 문장이 간헐적으로 섞여
나오는 문제 발견 — status 필드로 이미 판단하는 "다뤄졌는지 여부"가 summary 텍스트
서술로도 새어나온 것으로 추정(프롬프트가 "아직 다뤄지지 않았으면 uncovered로
판단하세요" 같은 커버리지 판단 지시를 하고 있어서). "반드시 지켜야 할 것" 문단에
이런 서술 문장을 구체적 반례(나쁜 예/좋은 예)로 명시해 금지하는 문장 추가 — 다만
"왜 중요한지 설명을 덧붙이라"는 기존 지시(내용 자체에 대한 설명)와는 구분해서,
그건 계속 허용된다고 명시.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import OpenAI

from interview_assistant.contracts import (
    Contradiction,
    CoverageStatus,
    InterviewSchema,
    KnowledgeState,
    SessionContext,
    TranscriptEvent,
)
from interview_assistant.config import get_structuring_model
from interview_assistant.llm_client import STRUCTURED_EXTRA_BODY, get_client, structured_response_format

_STATUS_VALUES = [s.value for s in CoverageStatus]

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "status": {"type": "string", "enum": _STATUS_VALUES},
                    "new_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "new_contradictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "with_ref": {"type": "string"},
                                "note": {"type": "string"},
                            },
                            "required": ["with_ref", "note"],
                            "additionalProperties": False,
                        },
                    },
                    "resolved_contradiction_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "item_id", "summary", "status", "new_refs",
                    "new_contradictions", "resolved_contradiction_refs",
                ],
                "additionalProperties": False,
            },
        },
        "new_sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "summary": {"type": "string"},
                    "status": {"type": "string", "enum": _STATUS_VALUES},
                    "refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["label", "summary", "status", "refs"],
                "additionalProperties": False,
            },
        },
        "renames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "new_label": {"type": "string"},
                },
                "required": ["item_id", "new_label"],
                "additionalProperties": False,
            },
        },
        "question_turn_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sections", "new_sections", "renames", "question_turn_ids"],
    "additionalProperties": False,
}


@dataclass
class SectionUpdate:
    item_id: str
    summary: str
    status: CoverageStatus
    new_refs: list[str]
    new_contradictions: list[Contradiction]
    resolved_contradiction_refs: list[str]


@dataclass
class NewSectionProposal:
    label: str
    summary: str
    status: CoverageStatus
    refs: list[str]


@dataclass
class RenameProposal:
    item_id: str
    new_label: str


@dataclass
class SynthesisResult:
    sections: list[SectionUpdate]
    new_sections: list[NewSectionProposal]
    renames: list[RenameProposal] = field(default_factory=list)
    # 화자와 무관, 발화 내용 자체가 (전문가에게 던지는) 질문이라 담긴 정보가 없는
    # turn_id들 — 어떤 섹션의 근거로도 안 쓰인다.
    question_turn_ids: list[str] = field(default_factory=list)


def _build_user_message(events: list[TranscriptEvent], state: KnowledgeState) -> str:
    lines = ["[현재까지의 구조화 상태]"]
    if not state.schema_items:
        lines.append("(아직 섹션 없음)")
    for item in state.schema_items:
        lines.append(f"\n[섹션 {item.item_id}] {item.label} (status: {item.status.value})")
        lines.append(item.summary if item.summary else "(아직 내용 없음)")
        for contra in item.contradictions:
            lines.append(f"  ⚠ 모순: {contra.with_ref}와 상충 — {contra.note}")
    lines.append("\n[이번 구조화 주기에 새로 들어온 발화]")
    for e in events:
        lines.append(f"[{e.turn_id}] {e.text}")
    return "\n".join(lines)


def _build_system_prompt(schema: InterviewSchema, session_context: SessionContext) -> str:
    items_desc = (
        "\n".join(
            f"- {item.item_id}: {item.label} (기준: {item.criteria})"
            for item in schema.items
        )
        or "(없음)"
    )
    return (
        "당신은 진행 중인 전문가 인터뷰의 지식 구조를 실시간으로 편집하는 편집자입니다. "
        "목표는 스키마 항목을 채웠는지 확인하는 게 아니라, 지금까지 들은 내용 전체를 "
        "빠짐없이 붙잡아 실제로 참고하고 활용하기 좋은 잘 정리된 문서로 계속 다시 "
        "정리해나가는 것입니다.\n"
        f"도메인: {schema.domain}\n"
        f"이번 인터뷰 목적: {session_context.interview_goal}\n"
        f"전문가 프로필: {session_context.expert_profile}\n"
        f"포커스: {session_context.focus_notes}\n\n"
        f"사전 정의된 스키마 항목(참고용, 강제 아님):\n{items_desc}\n\n"
        "[현재까지의 구조화 상태]에는 지금까지 만들어진 섹션들과 그 최신 요약이 담겨 "
        "있고, [이번 구조화 주기에 새로 들어온 발화]에는 이번에 새로 반영해야 할 원문이 "
        "담겨 있습니다. 짧게 끊긴 여러 턴이라도 순서대로 이어지는 하나의 맥락으로 "
        "읽으세요(실제 STT는 문장을 잘라서 인식하는 경우가 많습니다).\n\n"
        "먼저 각 turn_id별로 그 발화 자체가 (전문가에게 무언가를 요청·질문하는) 질문이고 "
        "새로운 정보를 담고 있지 않은지 판단하세요 — 그렇다면 그 turn_id를 "
        "question_turn_ids에 담고, 아래 어떤 근거로도 쓰지 마세요. 전문가가 스스로 던지는 "
        "질문처럼 그 자체로 정보나 통찰을 담고 있다면 question_turn_ids에 넣지 마세요. "
        "발화를 누가 말했는지는 알 수 없으니 오직 내용만 보고 판단하세요. 애매하면 넣지 "
        "마세요 — 대부분의 발화는 정보를 담은 답변입니다.\n\n"
        "question_turn_ids에 안 들어간 발화들을 대상으로:\n"
        "1) 새 발화가 기존 섹션과 관련 있으면, 그 섹션의 요약을 이번 내용까지 반영해 "
        "처음부터 다시 쓰세요(그냥 끝에 덧붙이지 말고, 필요하면 전체를 재구성) — "
        "sections에 {item_id, summary, status, new_refs, new_contradictions, "
        "resolved_contradiction_refs}로 담으세요. 단순히 사실을 나열하지 말고, 부제목으로 "
        "하위 구조를 나누거나 왜 중요한지 설명을 덧붙이는 등 잘 정리된 문서처럼 "
        "쓰세요(Markdown, 제목은 #### 이하만). 비교할 게 여러 개일 땐 표를 활용하되 "
        "억지로 만들지 마세요. new_refs에는 이번에 새로 반영한 turn_id를 전부(여러 턴에 "
        "걸쳐도 전부) 담으세요.\n"
        "2) status는 이 섹션이 아직 다뤄지지 않았으면 uncovered, 한 가지 관점에서만 "
        "다뤄졌으면 partial, 여러 관점(기준/휴리스틱/예외/근거 등)에서 다뤄졌으면 covered로 "
        "판단하세요.\n"
        "3) new_contradictions에는 이 섹션의 현재 요약과 새 발화 사이(또는 새 발화들 "
        "사이)에 실제로 내용이 어긋나는 지점이 있을 때만 {with_ref, note}를 담으세요 — "
        "with_ref는 이번 새 발화의 turn_id뿐 아니라 이 섹션이 지금까지 근거로 삼은 어떤 "
        "turn_id도 가능합니다. 단순히 관점이 다르거나 보완적인 내용은 모순이 아니니 "
        "억지로 지어내지 마세요. resolved_contradiction_refs에는 [현재까지의 구조화 "
        "상태]에 이미 나온 모순 중, 새 발화로 실제로 해소됐다고 아주 뚜렷하게 판단될 "
        "때만 그 with_ref를 담으세요 — 애매하면 비워두세요.\n"
        "4) 어느 기존 섹션과도 안 맞는 새로운 주제라면 new_sections에 새 섹션 제목과 "
        "함께 담으세요(summary/status/refs는 sections와 같은 방식). 섹션 목록에 비슷한 "
        "제목이 이미 있으면 새로 만들지 말고 sections로 그 섹션에 반영하세요. 이번 배치당 "
        "new_sections는 최대 2~3개만 제안하세요.\n"
        "5) 섹션 제목은 한 번 정해졌다고 무조건 그대로 두는 게 아닙니다 — 기존 섹션 "
        "제목이 지금까지 쌓인 내용과 더 이상 안 맞는다고 아주 뚜렷하게 판단될 때만 "
        "renames에 {item_id, new_label}을 담으세요. 대부분의 경우 제목은 그대로 두는 게 "
        "맞습니다. 애매하면 renames를 빈 배열로 두세요.\n\n"
        "반드시 지켜야 할 것: 실제로 언급된 내용만 근거로 삼으세요 — 없는 내용을 지어내지 "
        "마세요. 이번 새 발화와 무관한 섹션은 sections에 아예 포함하지 마세요(건드리지 "
        "않은 섹션은 지금 상태 그대로 유지됩니다). 수치·단위·날짜·고유명사·제품명 등 "
        "구체적인 정보는 절대 생략하거나 뭉뚱그려 쓰지 말고 발화에 나온 그대로 정확히 "
        "옮기세요. 수치를 뭉뚱그리는 대표적인 실수: 발화에 '22퍼센트'처럼 정확한 숫자가 "
        "나왔는데 요약엔 '(특정) 퍼센트'나 '일정 비율'처럼 얼버무려 쓰는 것 — 이런 표현은 "
        "금지합니다. 정확한 수치가 발화에 있으면 그 숫자를 그대로 쓰고, 정확한 수치가 "
        "없으면 수치 언급 자체를 생략하거나 '정확한 숫자는 언급되지 않음'처럼 있는 "
        "그대로만 쓰세요 — '(특정)' 같은 괄호 placeholder는 쓰지 마세요. summary 텍스트 "
        "안에 [t009]처럼 turn_id를 대괄호로 인용하지도 마세요 — 출처 추적은 new_refs "
        "필드로 이미 충분하니, 문장은 인용 표시 없이 자연스러운 정리문으로만 쓰세요. "
        "summary 본문에 인터뷰가 어떻게 진행되고 있는지를 서술하지도 마세요 — "
        "'~에 대한 설명이 이어졌습니다', '~는 아직 이야기되지 않았습니다', "
        "'~라고 말했습니다'처럼 대화의 진행 상황이나 발화 행위 자체를 묘사하는 문장은 "
        "금지합니다. 다뤄졌는지 여부는 status 필드로만 표현하고, summary는 대화록이 "
        "아니라 완성된 지식 문서처럼 실제 내용(사실·기준·판단)만 곧바로 쓰세요. 나쁜 예: "
        "'로스팅 시간에 대한 설명이 이어졌고, 온도 기준은 아직 언급되지 않았습니다.' "
        "좋은 예: '로스팅 시간은 참고 지표로만 쓰이며 절대 기준은 아니다.'(다뤄지지 않은 "
        "내용은 언급하지 말고 그 섹션 자체를 이번엔 sections에서 빼세요.) 다만 내용 "
        "자체에 대한 설명(왜 중요한지·어떻게 쓰이는지)은 계속 덧붙이세요 — 금지하는 "
        "것은 인터뷰가 어떻게 흘러갔는지에 대한 서술뿐입니다."
    )


def synthesize_structure(
    events: list[TranscriptEvent],
    schema: InterviewSchema,
    session_context: SessionContext,
    state: KnowledgeState,
    *,
    client: OpenAI | None = None,
) -> SynthesisResult:
    client = client or get_client()
    response = client.chat.completions.create(
        model=get_structuring_model(),
        messages=[
            {"role": "system", "content": _build_system_prompt(schema, session_context)},
            {"role": "user", "content": _build_user_message(events, state)},
        ],
        response_format=structured_response_format("structure_synthesis", _RESPONSE_SCHEMA),
        extra_body=STRUCTURED_EXTRA_BODY,
    )
    data = json.loads(response.choices[0].message.content)
    sections = [
        SectionUpdate(
            item_id=s["item_id"],
            summary=s["summary"],
            status=CoverageStatus(s["status"]),
            new_refs=s["new_refs"],
            new_contradictions=[
                Contradiction(with_ref=c["with_ref"], note=c["note"]) for c in s["new_contradictions"]
            ],
            resolved_contradiction_refs=s["resolved_contradiction_refs"],
        )
        for s in data["sections"]
    ]
    new_sections = [
        NewSectionProposal(
            label=n["label"], summary=n["summary"], status=CoverageStatus(n["status"]), refs=n["refs"]
        )
        for n in data["new_sections"]
    ]
    renames = [
        RenameProposal(item_id=r["item_id"], new_label=r["new_label"]) for r in data["renames"]
    ]
    return SynthesisResult(
        sections=sections,
        new_sections=new_sections,
        renames=renames,
        question_turn_ids=data["question_turn_ids"],
    )
