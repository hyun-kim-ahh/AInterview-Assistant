"""
데이터 계약 (contracts) — 모든 모듈을 잇는 접착제.

모듈 간 통신은 오직 이 파일의 타입으로만 이루어진다. 어떤 모듈도 다른 모듈의
내부를 import하지 않는다. 계약을 바꿔야 하면 여기부터 고치고 하류로 전파한다.

※ 시드(seed)다. dev-plan 1단계에서 검토·확정하고, 필요하면 pydantic 등으로 교체해도 된다.
  design.md §4.3의 JSON 스키마와 대응한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 1. 전사 이벤트 — 모든 TranscriptSource 구현이 방출하는 공통 계약
# ─────────────────────────────────────────────────────────────

class Speaker(str, Enum):
    INTERVIEWER = "interviewer"
    EXPERT = "expert"


@dataclass
class TranscriptEvent:
    """STT / 붙여넣기 / 파일재생 어느 Source든 이것만 방출한다.

    기본은 순수 전사다. speaker는 선택(기본 None) — 화자분리를 지원하는 Source가
    알려주면 힌트로 쓰고, 몰라도 하류는 그대로 동작한다.
    """
    text: str
    timestamp: float          # 세션 시작으로부터 초
    turn_id: str              # 예: "t150" — verbatim 역참조 키
    speaker: Optional[Speaker] = None   # 선택: 알면 채우고, 모르면 None
    corrected: bool = False   # STT 오인식 교정이 적용됐는지(structurer/transcript_reviewer.py) — 표시용, 트리거 아님


@dataclass
class TranscriptCorrection:
    """전사 재검토(structurer/transcript_reviewer.py)가 제안하는 교정 — 뒤 턴의
    내용으로 미뤄봤을 때 앞선 턴이 STT 오인식됐다고 강하게 확신될 때만 나온다.
    Structurer 내부 타입이 아니라 app/web.py가 소비하는 모듈 간 계약이라 여기 둔다."""
    turn_id: str               # 교정 대상 턴
    corrected_text: str
    confidence: float = 0.0    # 0~1, 이 교정이 맞다는 확신도
    reason: str = ""           # 왜 그렇게 판단했는지 한 문장


@dataclass
class FinalDocument:
    """인터뷰 종료 시 1회(structurer/final_document.py)가 생성하는 종합 정리 결과 —
    Structurer 내부 타입이 아니라 app/web.py가 KnowledgeState에 적용하는 모듈 간
    계약이라 여기 둔다(TranscriptCorrection과 같은 이유). 각 필드는 마크다운 문서
    텍스트(v0.14 — 표/헤딩/목록을 표현하려고 요점 배열에서 되돌림)."""
    overview: str = ""
    section_summaries: dict[str, str] = field(default_factory=dict)  # item_id -> 마크다운 요약


@dataclass
class TriggerDecision:
    """question_engine/trigger_judge.py가 판단하는 결과 — "지금 추천 질문을 생성할
    때인가"를 인터뷰 진행 정도·목적·포커스를 함께 고려해 정성적으로 판단한다.
    app/web.py의 백그라운드 루프가 소비하는 모듈 간 계약이라 여기 둔다
    (TranscriptCorrection과 같은 이유)."""
    should_trigger: bool
    reason: str = ""  # 트리거할 때만 어떤 발언 때문인지 한 문장(생성 쪽에 힌트로 전달됨)


# ─────────────────────────────────────────────────────────────
# 2. 지식 상태 — 상시 갱신되는 인터뷰의 "현재 이해"
# ─────────────────────────────────────────────────────────────

class CoverageStatus(str, Enum):
    UNCOVERED = "uncovered"
    PARTIAL = "partial"
    COVERED = "covered"


@dataclass
class Contradiction:
    with_ref: str              # 충돌하는 원문 turn_id
    note: str = ""


@dataclass
class SchemaItem:
    """인터뷰 스키마의 한 항목 + 그 항목에 대해 지금까지 종합된 것.

    (2026-08-05 추가) 턴 하나씩 분류해 캡처를 쌓는 방식(Capture 리스트)에서, 매
    구조화 틱마다 현재 전체 상태 + 이번 배치를 보고 이 섹션의 요약을 처음부터 다시
    쓰는 방식으로 전환 — "분류/미분류" 개념 자체를 없앴다(사용자 요청). summary는
    더 이상 인터뷰 종료 시 1회만 채워지는 필드가 아니라 진행 중에도 매 틱마다 계속
    갱신된다(final_document.py는 이제 "처음 종합"이 아니라 "마지막 다듬기" 역할).
    """
    item_id: str
    label: str
    status: CoverageStatus = CoverageStatus.UNCOVERED
    contradictions: list[Contradiction] = field(default_factory=list)
    # 마크다운 문서 텍스트(표/헤딩/목록 포함 가능, app/web.py가 HTML로 렌더링) — 이
    # 섹션에 대해 지금까지 나온 발화를 종합한 최신 상태. 진행 중엔 빈 문자열.
    summary: str = ""
    # 지금까지 이 summary에 반영된 turn_id 전체(감사 추적용, 누적·중복 없음) — 예전
    # Capture.verbatim_ref 목록의 경량화 버전. 개별 발언 내용은 더 안 남기고, "어느
    # 발화들이 근거였는지"만 유지한다.
    source_refs: list[str] = field(default_factory=list)


@dataclass
class KnowledgeState:
    session_id: str
    schema_items: list[SchemaItem] = field(default_factory=list)
    # 마크다운 문서 텍스트, 진행 중에도 매 틱마다 계속 갱신(SchemaItem.summary와 같은
    # 전환) — 전체를 종합한 개요. 진행 중에는 빈 문자열로 시작.
    overview: str = ""


# ─────────────────────────────────────────────────────────────
# 3. 질문 후보 — 트리거 시 표출되는 캐시 대상
# ─────────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    PROBE = "probe"                 # 파고들기: 방금 답변의 근거를 더 캐기
    CONTRADICTION = "contradiction" # 모순확인: 앞서와 어긋나는 지점 짚기
    GAP = "gap"                     # 누락: 아직 안 다룬 스키마 항목
    EXPAND = "expand"               # 확장: 새 주제 열기


@dataclass
class QuestionCandidate:
    type: QuestionType
    text: str
    target_item: Optional[str] = None   # 관련 SchemaItem.item_id (없으면 None)
    refs: list[str] = field(default_factory=list)  # 관련 원문 turn_id들


@dataclass
class QuestionCandidates:
    generated_at: str                   # 생성 시점의 turn_id
    candidates: list[QuestionCandidate] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# 4. 인터뷰 스키마 (입력) — 대상 도메인을 결정하는 세션별 주입물
# ─────────────────────────────────────────────────────────────

@dataclass
class SchemaItemDef:
    item_id: str
    label: str
    criteria: str = ""          # 이 항목에서 무엇을 확인해야 하는지


@dataclass
class InterviewSchema:
    domain: str                 # 사람이 읽는 도메인 라벨 (자유 문자열)
    items: list[SchemaItemDef] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# 5. 세션 컨텍스트 (입력) — "이번" 인터뷰의 목적·배경, 세션마다 새로 입력
#    (InterviewSchema와 별개: 스키마는 도메인 항목 골격으로 여러 세션에 재사용
#     가능하지만, 세션 컨텍스트는 매 인터뷰마다 다시 채운다.)
# ─────────────────────────────────────────────────────────────

@dataclass
class SessionContext:
    """이번 인터뷰에 한정된 설정. Structurer/Question Engine에 그라운딩으로 전달."""
    session_id: str
    interview_goal: str = ""    # 이번 인터뷰에서 구체적으로 얻고자 하는 것
    expert_profile: str = ""    # 전문가 소개/배경 (그라운딩용)
    focus_notes: str = ""       # 이번 세션에서 특히 파고들고 싶은 주제·우려사항


# ─────────────────────────────────────────────────────────────
# 6. 채택/무시/편집 로그 (F11) — 질문 카드에 대한 인터뷰어의 반응, 학습 신호용
# ─────────────────────────────────────────────────────────────

class AdoptionAction(str, Enum):
    ADOPTED = "adopted"      # 원문 그대로 채택
    DISMISSED = "dismissed"  # 무시
    EDITED = "edited"        # 편집 후 채택


@dataclass
class AdoptionEvent:
    generated_at: str            # 그 후보가 생성된 시점 turn_id(QuestionCandidates.generated_at과 동일 관례)
    question_type: QuestionType
    question_text: str           # 원래 추천 텍스트
    action: AdoptionAction
    edited_text: str = ""        # action=EDITED일 때만 실제로 물어본 텍스트
    target_item: Optional[str] = None
