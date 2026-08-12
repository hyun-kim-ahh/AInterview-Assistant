"""
최소 UI (dev-plan 10단계) — localhost 웹 한 창: 전사 페인 + 트리거 버튼 + 질문 카드,
세션 시작 전 F0 폼(스키마 선택 + 세션 컨텍스트 입력).

서버 렌더링 Jinja2 + 순수 HTML 폼(POST/Redirect/GET)만 쓴다 — JS 프레임워크·AJAX·
WebSocket 없음. 전사 진행은 CLI의 Enter키를 "다음 턴" 버튼으로 옮긴 것뿐이다.

개인용·단일 세션이라 전역 변수 하나로 상태를 들고 있는다. Flask 요청 스레드는 기본
단일 스레드라 요청끼리는 자연히 직렬화되지만, 세션당 백그라운드 데몬 스레드 두 개
(구조화 `_structuring_loop`, 목차 시딩 `_seed_outline_once`)가 추가로 동시에 돈다 —
`_state`를 주고받는 유일한 동시성 지점이라 identity 체크(락 대신)로 최소한만
방어한다. 상세 이유는 아래 `_structuring_loop` 참고.

추천 질문 생성은 dev-plan 13단계(P2: 생성/표출 분리)에선 3초 고정 배경 루프였지만,
"어차피 안 볼 수도 있는데 매번 LLM을 부르는 건 토큰 낭비"라는 이유로 되돌렸다 —
`/ask` 버튼을 누른 시점에 그 자리에서 동기 생성한다(2026-07-30). 구조화(스키마
매핑/새 섹션 생성)는 이 결정과 무관하게 계속 백그라운드 상시 루프다 — "문서를
계속 쌓아나간다"는 이 도구의 핵심 가치라 별개로 유지.

(2026-08-04 추가) "버튼을 안 눌러도 시스템이 알아서 추천해주는 것도 필요하다"는
요청으로, 고정 주기(`_question_loop`) 백그라운드 보완 생성을 다시 들였다 —
다만 새 턴이 하나도 안 쌓였으면 호출을 건너뛰어(`_has_new_turns`류 가드) 13단계를
되돌린 이유(토큰 낭비)가 재발하지 않게 한다. "인터뷰 종료"(`/end`)도 같은 즉시-생성
트리거의 하나이자, 입력 경로(마이크/자동 전사 수집)를 멈추는 유일한 방법이다 —
"새 세션 시작"(`/new`)과 달리 화면은 그대로 유지된다.

(2026-08-05 추가) 백그라운드 주기(현재 10초)를 "새 턴 있으면 무조건 생성"에서 "새 턴
있으면 판단(`question_engine/trigger_judge.py`)부터 하고, 판단이 통과해야만
생성"으로 한 단계 더 정교화했다 — 진짜 인터뷰어가 대화를 들으며 "이건 물어봐야겠다"/
"아직 때가 아니다"를 정성적으로 판단하는 것을 흉내낸다. 판단에는 인터뷰 목적·
포커스·진행 정도(경과 시간·턴 수)·커버리지가 컨텍스트로 들어가고, 판단이 트리거를
결정하면 그 이유가 `generate_questions`에 힌트로 전달돼 실제로 그 지점을 파고드는
질문이 나오게 한다. 버튼(`/ask`)·종료(`/end`)는 판단 없이 항상 즉시 생성(P1 풀
방식 — 사람이 명시적으로 누른 트리거는 그대로 존중).

구조화(턴 단위 분류) 자체도 "진행 중엔 raw 캡처만 계속 쌓이고 정리된 문서가 아니다"
라는 지적을 받았지만, 매 구조화 틱마다 다시 종합하면 토큰 낭비가 심하므로 진행
중에는 그대로 두고, `/end`가 백그라운드로(`_write_final_document`) 딱 1회 개요+
섹션별 요약+미분류 요약을 한 번의 호출로 작성해 `KnowledgeState`에 얹는다
(`structurer/final_document.py`).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import markdown as _markdown_lib
from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

from interview_assistant.config import (
    QUESTION_GENERATION_INTERVAL_SECONDS,
    STRUCTURING_LOOKBACK_TURNS,
    TRANSCRIPT_CORRECTION_MIN_CONFIDENCE,
)
from interview_assistant.contracts import (
    AdoptionAction,
    AdoptionEvent,
    CoverageStatus,
    InterviewSchema,
    QuestionCandidates,
    QuestionType,
    SchemaItemDef,
    SessionContext,
    Speaker,
    TranscriptEvent,
)
from interview_assistant.question_engine.question_engine import generate_questions
from interview_assistant.question_engine.trigger_judge import judge_trigger
from interview_assistant.schema_loader import load_schema, save_schema
from interview_assistant.sources.file_replay import FileReplaySource
from interview_assistant.sources.stt.deepgram_live_source import DeepgramLiveMicSTTSource
from interview_assistant.sources.stt.deepgram_source import DeepgramFileSTTSource
from interview_assistant.storage.adoption_log import AdoptionLogWriter, read_adoption_log
from interview_assistant.storage.knowledge_export import (
    KnowledgeStateExporter,
    knowledge_state_to_dict,
    load_exported_state,
)
from interview_assistant.storage.session_log import SessionLogWriter
from interview_assistant.storage.session_meta import (
    SessionMeta,
    delete_session,
    list_sessions,
    read_session_meta,
    write_session_meta,
)
from interview_assistant.structurer.final_document import write_final_document
from interview_assistant.structurer.structurer import Structurer
from interview_assistant.structurer.transcript_reviewer import review_transcript

REPO_ROOT = Path(__file__).resolve().parents[3]
TRANSCRIPT_PATH = REPO_ROOT / "fixtures" / "fake_transcript.jsonl"
# 스키마 파일 전용 디렉터리(2026-08-07) — F0에서 경로를 직접 붙여넣지 않고, 이 안의
# 파일들을 드롭다운으로 보여줘 고르기만 하면 바로 편집 그리드에 뜨게 한다(_list_schema_files).
SCHEMAS_DIR = REPO_ROOT / "schemas"
# 테스트가 "실제로 존재하는 스키마 파일 경로"로 참조하는 상수 — UI가 보여주는 기본
# 선택값(index()가 그때그때 SCHEMAS_DIR의 첫 파일을 고름)과는 별개다(다른 UI
# 기본값들과 마찬가지로 "UI 기본값 vs 테스트용 내부 상수" 분리 원칙).
DEFAULT_SCHEMA_PATH = SCHEMAS_DIR / "example_schema.json"
# 오디오 파일 테스트 전용 디렉터리(2026-08-07) — schemas/와 같은 이유로 fixtures/와
# 분리했다: fixtures/의 오디오 파일들은 tests/test_stt_deepgram.py(실 API 라이브
# 테스트)/scripts/compare_stt_vendors.py가 그대로 참조하고 있어 건드리면 안 된다.
AUDIO_DIR = REPO_ROOT / "audio"
_AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.m4a", "*.mp4", "*.flac", "*.ogg", "*.webm")
SESSIONS_DIR = REPO_ROOT / "sessions"
DEFAULT_STRUCTURING_INTERVAL_SECONDS = 5.0

app = Flask(__name__)


def _render_markdown(text: str) -> str:
    """종합 정리 문서(structurer/final_document.py)가 만드는 마크다운 텍스트를
    화면에 보여줄 HTML로 변환한다. 표(테이블)를 적극 활용하라고 프롬프트에서
    요청하므로 tables 확장을 켠다. 원본 마크다운은 KnowledgeState/저장 파일에
    그대로 남고, 이 함수는 표시 시점에만 쓰인다(Jinja 필터 + /status JSON)."""
    if not text:
        return ""
    return _markdown_lib.markdown(text, extensions=["tables"])


app.jinja_env.filters["markdown"] = _render_markdown


def _format_elapsed(seconds: int) -> str:
    """경과 시간 표시 포맷(2026-08-10) — 60초 넘어가면 "n분 m초", 그 전엔 "m초"만.
    session.html의 JS `formatElapsed`가 실시간 틱에서 같은 규칙을 중복 구현한다
    (Python/JS 런타임이 분리돼 있어 공유 불가 — SSR과 JS 틱 둘 다 이 규칙을 따르게
    나란히 유지)."""
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}분 {secs}초" if minutes else f"{secs}초"


app.jinja_env.filters["format_elapsed"] = _format_elapsed


@dataclass
class AppState:
    schema: InterviewSchema
    session_context: SessionContext
    structurer: Structurer
    events: Iterator[TranscriptEvent]
    adoption_log: AdoptionLogWriter
    transcript_writer: SessionLogWriter
    knowledge_exporter: KnowledgeStateExporter
    transcript_log: list[TranscriptEvent] = field(default_factory=list)
    latest_candidates: QuestionCandidates | None = None
    # index()가 SSR 렌더 때마다 "이 배치를 이미 새 것으로 한 번 보여줬는지" 판단하는
    # 기준값(2026-08-10) — /adopt 등 POST 후 redirect로 같은 배치를 다시 그릴 때
    # card-flash가 매번 재생되는 걸 막는다. latest_candidates.generated_at과 값이
    # 같아지면 "이미 보여줬다"는 뜻.
    flashed_candidates_generated_at: str | None = None
    exhausted: bool = False
    started_at: datetime = field(default_factory=datetime.now)
    # 15단계 Milestone 3: 오디오 파일/실시간 마이크 모드는 이벤트가 예측 불가한 속도로
    # 도착하므로 "다음 턴" 버튼 대신 백그라운드 스레드가 자동으로 ingest한다(_ingestion_loop).
    # fixture 모드(기본값)는 지금처럼 버튼으로 수동 진행.
    auto_ingest: bool = False
    # fixture/audio_file/live_mic — auto_ingest만으로는 audio_file과 live_mic을 구분 못 해
    # 화면의 오디오 플레이어 표시 여부(audio_file일 때만)를 판단하려면 따로 필요하다.
    input_mode: str = "fixture"
    # audio_file 모드일 때만 채워짐 — /audio 라우트가 이 경로의 파일을 그대로 서빙한다.
    audio_path: str = ""
    # 실시간 마이크 소스는 세션 종료 시 정리(stop())해야 하는 리소스(마이크 스트림+WebSocket)를
    # 들고 있어서 참조를 남겨둔다 — fixture/audio_file 모드에서는 None.
    stt_source: DeepgramLiveMicSTTSource | None = None
    # "추천 질문 생성" 버튼을 다시 누르면 latest_candidates가 통째로 새로 그려지므로
    # "채택"한 질문은 그것과 별개로 여기 쌓아둔다 — 재생성 후에도 이미 채택한
    # 질문은 안 사라진다.
    kept_questions: list[AdoptionEvent] = field(default_factory=list)
    # F0에서 사용자가 고른 구조화 주기(초) — _structuring_loop가 이 간격으로 깨어난다.
    structuring_interval_seconds: float = DEFAULT_STRUCTURING_INTERVAL_SECONDS
    # transcript_log 중 이미 _run_structuring_pass로 처리한 개수(커서) — 다음 구조화
    # 틱에서 이 인덱스부터 이어서 처리한다.
    structured_count: int = 0
    # 이미 전사 교정이 적용된 turn_id — 한 턴당 세션에서 딱 1번만 교정해서 플립플롭 방지.
    corrected_turn_ids: set[str] = field(default_factory=set)
    # 추천 질문을 마지막으로 생성했을 때의 transcript_log 길이 — 버튼/주기적 백그라운드/
    # 종료 트리거 셋이 이 값을 공유해서, 백그라운드 루프가 방금 다른 트리거로 이미
    # 생성된 걸 또 반복 생성하는 낭비를 피한다.
    last_question_turn_count: int = 0
    # "인터뷰 종료" 버튼으로 입력 경로(마이크/자동 전사 수집)가 멈춘 상태 — 화면은
    # 그대로 유지되고 조회·채택·질문 재생성 등 나머지 기능은 계속 동작한다.
    ended: bool = False
    # /end 이후 종합 정리 문서를 백그라운드로 작성 중일 때만 True(화면에 "정리 중"
    # 표시용) — _write_final_document가 끝나면(성공/실패 무관) 다시 False가 된다.
    summarizing: bool = False
    # (2026-08-11 추가) 세션당 "구조화는 한 번에 하나만, 질문 생성도 한 번에
    # 하나만" 규칙을 강제하는 락 — 배경 루프(_structuring_loop/_question_loop)와
    # /end의 마지막 처리가 같은 세션에서 동시에 같은 종류의 LLM 호출을 두 번
    # 돌리다가 늦게 끝나는 쪽이 결과를 덮어쓰는 경합을 막는다. CLAUDE.md의
    # "무관한 요청끼리 서로 안 막히게 락을 안 쓴다"는 원칙과는 별개다 — 그건
    # /next·/status처럼 서로 상관없는 요청을 막지 말자는 취지였고, 이 락은
    # 정확히 같은 작업(구조화/질문 생성)이 두 곳에서 겹치는 것만 좁게 막는다.
    structuring_lock: threading.Lock = field(default_factory=threading.Lock)
    question_generation_lock: threading.Lock = field(default_factory=threading.Lock)


_state: AppState | None = None


def _regenerate_once(app_state: AppState, *, trigger_hint: str | None = None) -> QuestionCandidates:
    # generate_questions를 기본 인자가 아니라 이름으로 직접 호출한다 — Structurer의
    # classify 기본 인자와 같은 late-binding 함정을 피하기 위함. 이렇게 해야
    # monkeypatch.setattr(web, "generate_questions", fake)가 테스트에서 그대로 먹힌다.
    # /ask 라우트가 버튼을 누를 때마다 직접 호출한다(더 이상 배경 루프가 없음).
    # trigger_hint는 _question_loop가 judge_trigger의 판단 이유를 넘길 때만 채워진다.
    return generate_questions(
        app_state.structurer.state,
        app_state.transcript_log,
        app_state.session_context,
        trigger_hint=trigger_hint,
    )


def _refresh_candidates(app_state: AppState, *, trigger_hint: str | None = None) -> None:
    """버튼(/ask)·주기적 백그라운드(판단 통과 시)·인터뷰 종료(/end, 2026-08-11부터
    백그라운드), 세 트리거가 공유하는 생성 호출 — 생성과 동시에 "몇 턴까지 반영했는지"를
    기록해, 백그라운드 루프가 다른 트리거가 방금 만든 결과를 곧바로 또 재생성하는
    낭비를 피할 수 있게 한다.

    백그라운드 스레드(_question_loop, _start_final_question_worker)에서도 호출되므로,
    호출 사이 세션이 바뀌었으면(예: /new) 결과를 조용히 버린다 — _write_final_document와
    같은 패턴. (2026-08-11 추가) question_generation_lock으로 세 트리거가 동시에
    돌지 않게 직렬화한다 — 배경 루프의 호출이 이미 진행 중일 때 /end가 마지막
    질문을 생성하려 하면, 배경 호출이 끝날 때까지 기다렸다가 그 다음에 자기
    몫(그 시점의 최신 상태 기준)을 실행한다. 그래야 늦게 끝나는 쪽이 이겨서
    "최종" 결과를 옛 배경 결과가 덮어쓰는 경합이 없다."""
    with app_state.question_generation_lock:
        candidates = _regenerate_once(app_state, trigger_hint=trigger_hint)
        if _state is not app_state:
            return
        app_state.latest_candidates = candidates
        app_state.last_question_turn_count = len(app_state.transcript_log)


def _seed_outline_once(app_state: AppState) -> None:
    """세션 시작 시 1회 — 예상 목차를 미리 채운다(Structurer.seed_outline 참고).
    세션이 그 사이 바뀌었으면(빠르게 /new 눌림 등) 결과를 조용히 버린다."""
    app_state.structurer.seed_outline()
    if _state is app_state:
        app_state.knowledge_exporter.export(app_state.structurer.state)


def _start_outline_seeding_worker(app_state: AppState) -> None:
    threading.Thread(target=_seed_outline_once, args=(app_state,), daemon=True).start()


def _write_final_document(app_state: AppState) -> None:
    """인터뷰 종료 시 1회 — 지금까지 쌓인 캡처 전체(섹션별 + 미분류)를 한 번에 보고
    종합 개요 + 섹션별 요약 + 미분류 요약을 작성한다. 세션이 그 사이 바뀌었으면
    결과를 조용히 버린다(_seed_outline_once와 같은 패턴). 입력이 인터뷰 전체
    분량이라 몇 초~수십 초 걸릴 수 있어 백그라운드 스레드로 돌린다(/end 응답을
    막지 않음). app_state.summarizing은 화면에 "정리 중" 표시용 — 실제 API 호출이
    실패해도(드묾) 표시가 영원히 안 꺼지지 않게 finally로 항상 되돌린다."""
    try:
        document = write_final_document(app_state.structurer.state, app_state.schema, app_state.session_context)
        if _state is not app_state:
            return
        state = app_state.structurer.state
        state.overview = document.overview
        for item in state.schema_items:
            if item.item_id in document.section_summaries:
                item.summary = document.section_summaries[item.item_id]
        app_state.knowledge_exporter.export(state)
    finally:
        app_state.summarizing = False


def _start_summary_worker(app_state: AppState) -> None:
    threading.Thread(target=_write_final_document, args=(app_state,), daemon=True).start()


def _start_final_question_worker(app_state: AppState) -> None:
    """/end 전용(2026-08-11) — 마지막 질문 생성을 메인 스레드에서 동기로 기다리지
    않고 종합 정리(_start_summary_worker)와 나란히 백그라운드로 돌린다. 이전엔
    /end 응답(redirect) 자체가 이 호출까지 끝나야 나가서, "정리 중" 표시(summarizing)를
    화면에 보여줄 방법(그 redirect나 /status 폴링) 자체가 이 세 번째 LLM 호출까지
    다 끝나야 열렸다 — 이제 응답이 구조화 flush+전사교정 뒤 바로 나간다.
    (2026-08-11, _finalize_interview로 흡수) 구조화 flush도 백그라운드로 옮겨서
    이제 이 함수도 _finalize_interview 안에서 flush가 끝난 뒤 호출된다."""
    threading.Thread(target=_refresh_candidates, args=(app_state,), daemon=True).start()


def _finalize_interview(app_state: AppState) -> None:
    """/end 전용(2026-08-11 신설) — 남은 원문 턴 flush(_run_structuring_pass) 후
    종합 정리+마지막 질문 생성을 병렬로 시작한다. 이전엔 이 flush가 /end 요청을
    처리하는 메인 스레드에서 동기로 돌아서(구조화 호출이라 몇 초~수십 초 걸릴 수
    있음) redirect 자체가 그만큼 늦게 나갔고, 그 결과 "정리 중" 표시(summarizing)가
    화면에 뜨는 시점도 체감상 한참 늦어졌다 — end_interview()가 summarizing=True를
    세우자마자(이 워커를 띄우기 *전에*) 곧바로 redirect하도록 바꿔서, 이제 flush까지
    포함해 이 함수 전체가 백그라운드에서 돈다."""
    if _state is not app_state:
        return
    _run_structuring_pass(app_state)
    if _state is not app_state:
        return
    _start_summary_worker(app_state)
    _start_final_question_worker(app_state)


def _start_finalize_worker(app_state: AppState) -> None:
    threading.Thread(target=_finalize_interview, args=(app_state,), daemon=True).start()


def _append_raw_event(app_state: AppState, event: TranscriptEvent) -> None:
    """`/next`와 `_ingestion_loop`가 공유하는 원문 처리 — 구조화는 여기서 하지 않는다.

    구조화(LLM 분류)는 더 이상 턴이 들어오자마자 동기 실행되지 않고, 사용자가 F0에서
    고른 주기로 도는 `_structuring_loop`가 전담한다(아래 참고). 여기서는 원문을
    `transcript_log`(화면 표시용)와 `transcript_writer`(디스크, F2)에만 즉시 반영한다 —
    이 두 가지는 비싸지 않아 실시간으로 유지해도 무방하다.
    """
    app_state.transcript_log.append(event)
    app_state.transcript_writer.append(event)


def _apply_transcript_corrections(app_state: AppState) -> None:
    """구조화 틱 하나당 1회 — review_transcript는 Structurer를 거치지 않는 독립
    함수라서 _run_structuring_pass가 ingest_batch 다음에 직접 호출한다. 코드 레벨
    방어(confidence 임계값, 이미 교정된 turn_id 재교정 방지)는 프롬프트 지시와
    별개로 여기서 강제한다."""
    corrections = review_transcript(app_state.transcript_log, app_state.schema, app_state.session_context)
    if not corrections:
        return
    by_turn_id = {e.turn_id: e for e in app_state.transcript_log}
    applied = False
    for corr in corrections:
        if corr.confidence < TRANSCRIPT_CORRECTION_MIN_CONFIDENCE:
            continue
        if corr.turn_id in app_state.corrected_turn_ids:
            continue
        event = by_turn_id.get(corr.turn_id)
        new_text = corr.corrected_text.strip()
        if event is None or not new_text or new_text == event.text:
            continue
        event.text = new_text
        event.corrected = True
        app_state.corrected_turn_ids.add(corr.turn_id)
        applied = True
    if applied:
        app_state.transcript_writer.export_corrected_snapshot(app_state.transcript_log)


def _run_structuring_pass(app_state: AppState) -> bool:
    """대기 중인(아직 구조화 안 된) 원문 턴 전체를 한 번에 structurer.ingest_batch로
    재종합한다(2026-08-05, 델타 분류 → 전체 재종합 전환 — 커버리지 계산·모순 감지도
    이제 이 호출 하나로 흡수됨, analyzer/ 모듈 삭제).

    (2026-08-10 추가) ingest_batch에 넘기는 배치는 "새로 쌓인 턴"뿐 아니라 그
    앞의 STRUCTURING_LOOKBACK_TURNS만큼도 다시 포함한다 — 이전 틱 끝에서 시작된
    발화가 이번 틱에야 의미가 분명해지는 경계 케이스 대응. structured_count(다음
    틱의 "새 턴" 기준점)의 의미 자체는 그대로다.

    처리한 턴이 있으면 구조화 결과를 파일로 내보내고(F6) True를 반환한다.

    (2026-08-11 추가) structuring_lock으로 배경 루프(_structuring_loop)와 /end의
    마지막 flush가 동시에 돌지 않게 직렬화한다. 배경 틱이 이미 진행 중일 때 /end가
    호출하면, 그 틱이 끝날 때까지 기다렸다가(=지금도 /end는 자기 호출이 끝나길
    기다리고 있었으니 체감상 더 느려지지 않음) pending을 그 시점 기준으로 다시
    계산한다 — 배경 틱이 이미 다 처리해놨으면 pending이 비어 즉시 False로
    끝나서 중복 API 호출 자체가 안 생긴다. pending을 락 밖에서 먼저 계산하면
    안 된다 — 기다리는 동안 다른 쪽이 structured_count를 이미 옮겨놨을 수 있어서.
    """
    with app_state.structuring_lock:
        pending = app_state.transcript_log[app_state.structured_count :]
        if not pending:
            return False
        lookback_start = max(0, app_state.structured_count - STRUCTURING_LOOKBACK_TURNS)
        app_state.structurer.ingest_batch(app_state.transcript_log[lookback_start:])
        if _state is not app_state:
            return False
        app_state.structured_count = len(app_state.transcript_log)
        _apply_transcript_corrections(app_state)
        app_state.knowledge_exporter.export(app_state.structurer.state)
        return True


def _structuring_loop(app_state: AppState) -> None:
    """세션당 하나, F0에서 고른 주기(structuring_interval_seconds)마다 깨어난다.

    입력 방식(fixture/audio_file/live_mic)과 무관하게 항상 돈다 — fixture 모드도
    이제 이 루프가 구조화한다("/next"는 원문만 append하고 구조화는 안 함).
    락을 걸면 LLM 호출(ingest_batch) 동안 다른 요청이 막혀버리므로, 대신 락 없이
    identity 체크(`_state is app_state`)만으로 세션이 그 사이 바뀌었는지 방어한다.
    인터뷰가 종료(ended)되면 더 들어올 내용이 없으니 루프도 멈춘다.
    """
    while _state is app_state and not app_state.ended:
        time.sleep(app_state.structuring_interval_seconds)
        if _state is not app_state or app_state.ended:
            return
        _run_structuring_pass(app_state)


def _start_structuring_worker(app_state: AppState) -> None:
    threading.Thread(target=_structuring_loop, args=(app_state,), daemon=True).start()


def _question_loop(app_state: AppState) -> None:
    """세션당 하나, QUESTION_GENERATION_INTERVAL_SECONDS(현재 10초)마다 깨어난다. 3단
    게이팅: (1) 마지막 생성 이후 새 턴이 하나도 안 쌓였으면 공짜로 스킵(기존
    _has_new_turns 취지 그대로) — (2) 새 턴이 있으면 judge_trigger로 "지금이 물어볼
    때인가"를 정성적으로 판단(인터뷰 목적·진행 정도·커버리지 참고) — (3) 판단이
    통과해야만 실제 생성(_refresh_candidates)을 호출한다. 진짜 인터뷰어가 듣다가
    "이건 물어봐야겠다"를 판단하는 것과 같은 결을 노려서, "새 턴 있으면 무조건
    재생성"이던 이전 방식보다 더 적절한 시점에만 생성되게 한다. 인터뷰가 종료
    (ended)됐으면 더 이상 돌 이유가 없어 멈춘다."""
    while _state is app_state and not app_state.ended:
        time.sleep(QUESTION_GENERATION_INTERVAL_SECONDS)
        if _state is not app_state or app_state.ended:
            return
        if len(app_state.transcript_log) == app_state.last_question_turn_count:
            continue
        elapsed_seconds = (datetime.now() - app_state.started_at).total_seconds()
        decision = judge_trigger(
            app_state.structurer.state,
            app_state.transcript_log,
            app_state.session_context,
            elapsed_seconds,
        )
        if not decision.should_trigger:
            continue
        _refresh_candidates(app_state, trigger_hint=decision.reason)


def _start_question_worker(app_state: AppState) -> None:
    threading.Thread(target=_question_loop, args=(app_state,), daemon=True).start()


def _ingestion_loop(app_state: AppState) -> None:
    """오디오 파일/실시간 마이크 모드 전용 — 이벤트가 도착하는 대로 자동으로 원문을 담는다.

    fixture 모드의 "/next" 버튼(사람이 직접 한 턴씩 진행)을 대신한다. 구조화는 더 이상
    여기서 하지 않는다 — `_structuring_loop`가 별도 주기로 담당(위 참고).

    "인터뷰 종료"(ended)가 눌리면 새로 도착하는 이벤트를 더 이상 담지 않고 멈춘다 —
    live_mic는 `stt_source.stop()`으로 스트림 자체가 끊기지만, audio_file 모드는
    `stt_source`가 없어(None) 이 체크가 유일한 정지 수단이다.
    """
    for event in app_state.events:
        if _state is not app_state or app_state.ended:
            return
        _append_raw_event(app_state, event)
        if _state is not app_state or app_state.ended:
            return
    if _state is app_state and not app_state.ended:
        app_state.exhausted = True


def _start_ingestion_worker(app_state: AppState) -> None:
    threading.Thread(target=_ingestion_loop, args=(app_state,), daemon=True).start()


def _coverage_summary(app_state: AppState) -> tuple[int, int]:
    items = app_state.structurer.state.schema_items
    gap_count = sum(1 for i in items if i.status != CoverageStatus.COVERED)
    contradiction_count = sum(1 for i in items if i.contradictions)
    return gap_count, contradiction_count


def _structured_state_for_status(app_state: AppState) -> dict:
    """/status JSON 전용 — knowledge_state_to_dict의 원본 마크다운은 그대로 두고
    (파일 export와 같은 소스), *_html 필드를 추가로 얹어서 폴링 JS(rebuildStructurePanel)가
    별도 마크다운 파서 없이 바로 innerHTML에 꽂을 수 있게 한다."""
    data = knowledge_state_to_dict(app_state.structurer.state)
    data["overview_html"] = _render_markdown(data["overview"])
    for item_dict in data["schema_items"]:
        item_dict["summary_html"] = _render_markdown(item_dict["summary"])
    return data


def _candidates_for_status(app_state: AppState) -> dict | None:
    """/status JSON 전용 — _structured_state_for_status와 같은 이유(폴링 JS
    rebuildQuestionPanel이 배경 생성 결과를 새로고침 없이 반영할 수 있게 함)."""
    if app_state.latest_candidates is None:
        return None
    return {
        "generated_at": app_state.latest_candidates.generated_at,
        "candidates": [
            {"type": c.type.value, "text": c.text, "target_item": c.target_item or ""}
            for c in app_state.latest_candidates.candidates
        ],
    }


def _items_from_form(form) -> list[dict]:
    """화면에 그대로 다시 보여줄 목적 — 빈 줄도 안 거르고 그대로 병렬 zip한다."""
    labels = form.getlist("item_label[]")
    criteria = form.getlist("item_criteria[]")
    return [{"label": label, "criteria": crit} for label, crit in zip(labels, criteria)]


def _build_schema_from_form(form) -> InterviewSchema:
    """실제 저장용 — 라벨 빈 줄은 버리고 item_id를 순서대로 새로 부여한다."""
    items = [
        SchemaItemDef(
            item_id=f"item_{i:02d}",
            label=row["label"].strip(),
            criteria=row["criteria"].strip(),
        )
        for i, row in enumerate(_items_from_form(form), start=1)
        if row["label"].strip()
    ]
    return InterviewSchema(domain=form.get("domain", "").strip(), items=items)


def _schema_items_for_template(schema: InterviewSchema) -> list[dict]:
    return [{"label": i.label, "criteria": i.criteria} for i in schema.items]


def _list_schema_files() -> list[Path]:
    """F0의 스키마 드롭다운이 보여줄 목록 — SCHEMAS_DIR의 *.json을 이름순으로.
    디렉터리가 없으면(초기 설치 등) 빈 목록."""
    if not SCHEMAS_DIR.is_dir():
        return []
    return sorted(SCHEMAS_DIR.glob("*.json"))


def _list_audio_files() -> list[Path]:
    """F0의 "오디오 파일 테스트" 드롭다운이 보여줄 목록 — AUDIO_DIR의 흔한
    오디오/영상 컨테이너 확장자를 이름순으로. 디렉터리가 없으면 빈 목록."""
    if not AUDIO_DIR.is_dir():
        return []
    return sorted(p for ext in _AUDIO_EXTENSIONS for p in AUDIO_DIR.glob(ext))


def _render_setup(status_code: int = 200, **kwargs):
    # setup.html은 6곳(index/load_schema_route/save_schema_route x2/create_session x2)에서
    # 렌더링되는데, 매번 반복되던 default_schema_path 보일러플레이트를 걷어내고
    # 지난 세션 목록(sessions)이 어디서든 일관되게 보이도록 공용 헬퍼로 모은다.
    kwargs.setdefault("default_schema_path", str(DEFAULT_SCHEMA_PATH))
    kwargs.setdefault("sessions", list_sessions(SESSIONS_DIR))
    # 방금 저장한 새 파일도 다음 렌더에 바로 드롭다운에 뜨도록 매번 새로 훑는다.
    kwargs.setdefault("available_schema_files", _list_schema_files())
    kwargs.setdefault("available_audio_files", _list_audio_files())
    return render_template("setup.html", **kwargs), status_code


@app.route("/")
def index():
    if _state is None:
        # F0을 처음 열 때 아무것도 자동으로 채우지 않는다 — 스키마/오디오
        # 드롭다운도 "선택 안 함"이 기본이고(setup.html), 세션 이름도 매번
        # 직접 입력한다(2026-08-07, 재확인 — 한 번 자동 로드를 넣었다가
        # "세션 이름은 비워두라고 했다"는 피드백으로 되돌림). 드롭다운 목록
        # 자체(available_schema_files/available_audio_files)는 _render_setup이
        # 계속 채워주므로 고를 것은 그대로 보인다 — 다만 아무것도 미리
        # 골라놓지 않을 뿐이다.
        return _render_setup(schema_domain="", schema_path_value="", schema_items=[])
    elapsed = int((datetime.now() - _state.started_at).total_seconds())
    gap_count, contradiction_count = _coverage_summary(_state)
    candidates_are_new = (
        _state.latest_candidates is not None
        and _state.latest_candidates.generated_at != _state.flashed_candidates_generated_at
    )
    if _state.latest_candidates is not None:
        _state.flashed_candidates_generated_at = _state.latest_candidates.generated_at
    return render_template(
        "session.html",
        state=_state,
        elapsed=elapsed,
        gap_count=gap_count,
        contradiction_count=contradiction_count,
        candidates_are_new=candidates_are_new,
    )


def _kept_questions_for_status(app_state: AppState) -> list[dict]:
    """/status·/adopt JSON 전용 — 카드가 아니라 "킵한 질문" 목록(session.html의
    kept-questions-panel과 같은 모양)을 폴링/fetch로 반영할 수 있게 한다."""
    return [
        {
            "question_type": k.question_type.value,
            "text": k.edited_text if k.action == AdoptionAction.EDITED else k.question_text,
        }
        for k in app_state.kept_questions
    ]


def _status_payload(app_state: AppState) -> dict:
    """`/status` 폴링과 `/ask`(추천 질문 생성)·`/adopt`(채택, 둘 다 fetch로 전환됨 —
    전체 페이지 리로드가 audio_file 모드의 <audio> 태그를 재생성해 재생을 끊기게
    하는 문제 때문) 셋이 공유하는 응답 모양. 세션 상태를 바꾸지 않는 순수 조회."""
    gap_count, contradiction_count = _coverage_summary(app_state)
    return {
        "active": True,
        "exhausted": app_state.exhausted,
        "auto_ingest": app_state.auto_ingest,
        "summarizing": app_state.summarizing,
        "transcript": [
            {
                "turn_id": e.turn_id,
                "speaker": e.speaker.value if e.speaker else None,
                "text": e.text,
                "corrected": e.corrected,
            }
            for e in app_state.transcript_log
        ],
        "gap_count": gap_count,
        "contradiction_count": contradiction_count,
        "structured_state": _structured_state_for_status(app_state),
        "candidates": _candidates_for_status(app_state),
        "kept_questions": _kept_questions_for_status(app_state),
    }


@app.route("/status")
def status():
    # 폴링 전용 조회 라우트(웹 UI 실시간화) — 세션 상태를 바꾸지 않는다.
    if _state is None:
        return jsonify({"active": False})
    return jsonify(_status_payload(_state))


@app.route("/schema/load", methods=["POST"])
def load_schema_route():
    # F0 화면의 "불러오기" — 스키마 그리드만 채운다(세션 시작 아님, 상태 불변).
    schema_path = request.form.get("schema_path", "").strip()
    info = None
    try:
        schema = load_schema(schema_path)
    except (OSError, json.JSONDecodeError, KeyError):
        schema = InterviewSchema(domain="", items=[])
        info = f"'{schema_path}' 경로에 파일이 없습니다 — 새로 작성해서 저장할 수 있습니다."
    return _render_setup(
        schema_path_value=schema_path,
        schema_domain=schema.domain,
        schema_items=_schema_items_for_template(schema),
        info=info,
        form=request.form,
    )


@app.route("/schema/save", methods=["POST"])
def save_schema_route():
    # /session은 그대로 두고(계약 무변경), 스키마를 파일로 저장만 한다 — "세션 시작"은
    # 그 뒤 이 파일을 읽어서 진행된다(두 단계 플로우, 아래 setup.html 안내 문구 참고).
    schema_path = request.form.get("schema_path", "").strip()
    schema = _build_schema_from_form(request.form)
    if not schema_path or not schema.domain:
        return _render_setup(
            400,
            schema_path_value=schema_path,
            schema_domain=schema.domain,
            schema_items=_items_from_form(request.form),
            error="저장하려면 경로와 도메인이 모두 필요합니다.",
            form=request.form,
        )
    save_schema(schema_path, schema)
    return _render_setup(
        schema_path_value=schema_path,
        schema_domain=schema.domain,
        schema_items=_schema_items_for_template(schema),
        info=f"'{schema_path}'에 저장했습니다.",
        form=request.form,
    )


@app.route("/session", methods=["POST"])
def create_session():
    global _state
    use_schema = request.form.get("use_schema", "yes") == "yes"
    if use_schema:
        schema_path = request.form.get("schema_path", "").strip()
        try:
            schema = load_schema(schema_path)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            return _render_setup(400, error=f"스키마를 불러올 수 없습니다: {exc}", form=request.form)
    else:
        # 스키마 없이 진행 — 도메인 라벨만 세션 구분용으로 남기고 항목은 비운다.
        domain = request.form.get("domain", "").strip() or "자유 형식 인터뷰"
        schema = InterviewSchema(domain=domain, items=[])

    session_context = SessionContext(
        session_id=f"web_{datetime.now():%Y%m%d_%H%M%S}",
        interview_goal=request.form.get("interview_goal", ""),
        expert_profile=request.form.get("expert_profile", ""),
        focus_notes=request.form.get("focus_notes", ""),
    )

    input_mode = request.form.get("input_mode", "fixture")
    stt_source: DeepgramLiveMicSTTSource | None = None
    audio_path = ""
    auto_ingest = input_mode != "fixture"
    if input_mode == "audio_file":
        # 절대 경로로 정규화 — 이 뒤로는 STT도, /audio 라우트의 send_file도 같은
        # 경로를 쓴다. send_file은 상대 경로를 (프로세스 CWD가 아니라) Flask 앱의
        # root_path 기준으로 찾아서, 상대 경로를 그대로 두면 STT는 열리는데
        # /audio는 FileNotFoundError가 나는 불일치가 생긴다.
        audio_path = str(Path(request.form.get("audio_path", "").strip()).resolve())
        try:
            events = DeepgramFileSTTSource(audio_path, realtime=True).stream()
        except (OSError, KeyError) as exc:
            return _render_setup(400, error=f"오디오 파일을 전사할 수 없습니다: {exc}", form=request.form)
    elif input_mode == "live_mic":
        stt_source = DeepgramLiveMicSTTSource()
        events = stt_source.stream()
    else:
        events = FileReplaySource(TRANSCRIPT_PATH).stream()

    try:
        structuring_interval_seconds = float(
            request.form.get("structuring_interval_seconds", DEFAULT_STRUCTURING_INTERVAL_SECONDS)
        )
    except (TypeError, ValueError):
        structuring_interval_seconds = DEFAULT_STRUCTURING_INTERVAL_SECONDS

    session_dir = SESSIONS_DIR / session_context.session_id
    structurer = Structurer(schema, session_context)
    adoption_log = AdoptionLogWriter(session_dir / "adoptions.jsonl")
    transcript_writer = SessionLogWriter(session_dir / "transcript.jsonl")
    knowledge_exporter = KnowledgeStateExporter(session_dir, schema.domain)
    write_session_meta(
        session_dir,
        SessionMeta(
            session_id=session_context.session_id,
            domain=schema.domain,
            has_schema=bool(schema.items),
            input_mode=input_mode,
            interview_goal=session_context.interview_goal,
            started_at=datetime.now().isoformat(),
        ),
    )
    _state = AppState(
        schema=schema,
        session_context=session_context,
        structurer=structurer,
        events=events,
        adoption_log=adoption_log,
        transcript_writer=transcript_writer,
        knowledge_exporter=knowledge_exporter,
        auto_ingest=auto_ingest,
        stt_source=stt_source,
        structuring_interval_seconds=structuring_interval_seconds,
        input_mode=input_mode,
        audio_path=audio_path,
    )
    _start_structuring_worker(_state)
    _start_outline_seeding_worker(_state)
    _start_question_worker(_state)
    if auto_ingest:
        _start_ingestion_worker(_state)
    return redirect(url_for("index"))


@app.route("/audio")
def session_audio():
    # audio_file 모드에서 세션 화면이 오디오를 재생하기 위한 소스. 세션 id/경로를
    # 쿼리로 안 받고 항상 현재 _state.audio_path만 서빙 — 클라이언트가 임의 경로를
    # 지정할 수 없어 path traversal 여지 자체가 없다(STT가 이미 같은 경로를 서버
    # 로컬에서 신뢰하고 여는 것과 같은 신뢰 경계, 새 위험 추가 아님).
    if _state is None or _state.input_mode != "audio_file" or not _state.audio_path:
        abort(404)
    return send_file(_state.audio_path)


@app.route("/next", methods=["POST"])
def next_turn():
    # auto_ingest 모드(오디오 파일/실시간 마이크)는 백그라운드 스레드(_ingestion_loop)가
    # 알아서 진행하므로 이 버튼은 UI에서 숨겨지지만, 혹시 눌려도 안전하게 무시한다.
    # 구조화는 더 이상 여기서 동기 실행하지 않는다 — _structuring_loop가 주기적으로 담당.
    if _state is not None and not _state.exhausted and not _state.auto_ingest and not _state.ended:
        event = next(_state.events, None)
        if event is None:
            _state.exhausted = True
        else:
            _append_raw_event(_state, event)
    return redirect(url_for("index"))


@app.route("/ask", methods=["POST"])
def ask():
    # 트리거 = 생성 트리거(2026-07-30 되돌림, 토큰 효율). 버튼을 누른 시점의 구조화
    # 결과 + 최근 전사(question_engine의 60초 시간창)로 그 자리에서 생성한다. 매번
    # 새로 생성한다 — "새 턴이 있을 때만" 같은 조건 없이 누를 때마다 최신 상태 반영.
    # (2026-08-11 추가) 세션이 있을 때는 redirect(전체 페이지 리로드) 대신 /status와
    # 같은 JSON을 반환 — 전체 리로드가 audio_file 모드의 <audio> 태그를 매번
    # 재생성해 재생을 끊기게 하던 문제 때문(session.html이 fetch로 받아 카드만
    # innerHTML 교체). 세션이 아직 없을 때(_state is None)는 갱신할 패널 자체가
    # 없으니 기존 그대로 setup 화면으로 redirect.
    if _state is not None:
        _refresh_candidates(_state)
        return jsonify(_status_payload(_state))
    return redirect(url_for("index"))


@app.route("/adopt", methods=["POST"])
def adopt_question():
    # 카드 액션이 "채택" 하나로 단순화됨 — 텍스트를 안 고치면 ADOPTED, 고쳤으면 EDITED.
    # 채택된 질문은 adoption_log(파일, 학습 신호용 기록)뿐 아니라 kept_questions(화면에
    # 계속 보여줄 목록)에도 남는다 — latest_candidates는 몇 초 뒤 배경 생성으로
    # 통째로 바뀌어도 kept_questions는 그대로 남아야 하기 때문.
    # (2026-08-11 추가) 세션이 있을 때는 redirect(전체 페이지 리로드) 대신 /ask와
    # 같은 이유로 JSON을 반환 — 전체 리로드가 audio_file 모드의 <audio> 태그를
    # 매번 재생성해 재생을 끊기게 하던 문제. 세션이 없을 때(_state is None)는
    # 갱신할 패널 자체가 없으니 기존 그대로 setup 화면으로 redirect.
    if _state is None:
        return redirect(url_for("index"))
    original_text = request.form.get("original_text", "")
    text = request.form.get("text", original_text)
    action = (
        AdoptionAction.EDITED
        if text.strip() != original_text.strip()
        else AdoptionAction.ADOPTED
    )
    event = AdoptionEvent(
        generated_at=_state.latest_candidates.generated_at if _state.latest_candidates else "",
        question_type=QuestionType(request.form.get("question_type", QuestionType.PROBE.value)),
        question_text=original_text,
        action=action,
        edited_text=text if action == AdoptionAction.EDITED else "",
        target_item=request.form.get("target_item") or None,
    )
    _state.adoption_log.append(event)
    _state.kept_questions.append(event)
    return jsonify(_status_payload(_state))


@app.route("/new", methods=["POST"])
def new_session():
    global _state
    if _state is not None:
        _run_structuring_pass(_state)  # 남은 원문 턴 flush — 저장된 결과를 최신으로 마무리
        if _state.stt_source is not None:
            _state.stt_source.stop()  # 마이크 스트림+WebSocket 정리
    _state = None
    return redirect(url_for("index"))


@app.route("/end", methods=["POST"])
def end_interview():
    # "새 세션 시작"(/new)과 다르다 — _state를 None으로 안 바꾸고 화면(session.html)도
    # 그대로 유지한 채, 입력 경로만 멈추고 마무리 질문을 한 번 더 생성한다. 이미
    # 종료된 세션에 다시 눌려도 안전하게 아무 일도 안 한다(idempotent).
    if _state is None or _state.ended:
        return redirect(url_for("index"))
    _state.ended = True
    if _state.stt_source is not None:
        _state.stt_source.stop()  # 마이크 스트림+WebSocket 정리(/new와 동일 호출)
    # (2026-08-11) 남은 원문 턴 flush(_run_structuring_pass, 구조화 LLM 호출이라
    # 몇 초~수십 초 걸릴 수 있음)까지 통째로 _finalize_interview로 옮겨 백그라운드에서
    # 돈다 — 이전엔 이 flush를 여기서 동기로 기다린 뒤에야 summarizing=True를 세우고
    # redirect했어서, "정리 중" 표시가 화면에 뜨는 시점 자체가 flush가 끝날 때까지
    # 늦어졌다(체감상 "한참 있다가 뜬다"). 이제 summarizing=True를 세우자마자(백그라운드
    # 워커를 띄우기도 전에) 바로 redirect하므로, 다음 페이지 로드/폴링에서 곧바로
    # "정리 중" 표시가 보인다.
    _state.summarizing = True  # 화면에 "정리 중" 표시 — 워커 시작 전에 세워야 함
    _start_finalize_worker(_state)  # flush → 종합 정리+마지막 질문 생성(병렬)을 전부 백그라운드로
    return redirect(url_for("index"))


def _load_corrected_transcript(path: Path) -> list[TranscriptEvent]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        TranscriptEvent(
            text=r["text"],
            timestamp=r["timestamp"],
            turn_id=r["turn_id"],
            speaker=Speaker(r["speaker"]) if r["speaker"] else None,
            corrected=r.get("corrected", False),
        )
        for r in data
    ]


@app.route("/sessions/<session_id>")
def session_detail(session_id: str):
    # 지난 세션 결과 조회(읽기 전용) — 디스크에 남은 파일만 읽는다, 실시간 세션 상태와 무관.
    session_dir = SESSIONS_DIR / session_id
    meta = read_session_meta(session_dir)
    if meta is None:
        return "세션을 찾을 수 없습니다.", 404
    corrected_path = session_dir / "transcript_corrected.json"
    transcript_path = session_dir / "transcript.jsonl"
    if corrected_path.exists():
        # 전사 교정이 반영된 최신 스냅샷 — 있으면 우선 사용.
        transcript = _load_corrected_transcript(corrected_path)
    elif transcript_path.exists():
        transcript = list(FileReplaySource(transcript_path).stream())
    else:
        transcript = []
    structured_state = load_exported_state(session_dir / "structured_state.json")
    adoptions = read_adoption_log(session_dir / "adoptions.jsonl")
    return render_template(
        "session_detail.html",
        meta=meta,
        transcript=transcript,
        structured_state=structured_state,
        adoptions=adoptions,
    )


@app.route("/sessions/<session_id>/delete", methods=["POST"])
def delete_session_route(session_id: str):
    # F0(setup.html)의 세션 목록은 _state가 None일 때만 그려지므로, 여기서 지워지는
    # 세션이 진행 중인 세션일 가능성 자체가 없다.
    delete_session(SESSIONS_DIR, session_id)
    return redirect(url_for("index"))
