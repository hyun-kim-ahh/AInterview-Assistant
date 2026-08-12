"""
CLI 관통 진입점 (dev-plan 7단계 마일스톤).

FileReplaySource(4) → Structurer 바보버전(6) → QuestionEngine 바보버전(7)을 연결한다.
Enter=다음 턴, q=추천 질문(비파괴적, 턴을 소비하지 않음), x 또는 Ctrl-D=종료.

실시간 페이싱+동시 트리거 감지는 스레드가 필요해 이 단계의 anti-goal("얇은 관통
전에 실시간·백그라운드 기능 착수 금지")에 걸린다. 대신 사람이 Enter로 직접 페이스를
조절하는 줄 단위 REPL로 "훑어보다 원할 때 트리거"라는 풀(pull) 방식 경험을 대체한다.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

from interview_assistant.contracts import (
    CoverageStatus,
    KnowledgeState,
    QuestionCandidates,
    SessionContext,
    TranscriptEvent,
)
from interview_assistant.question_engine.question_engine import generate_questions
from interview_assistant.schema_loader import load_schema
from interview_assistant.sources.file_replay import FileReplaySource
from interview_assistant.structurer.structurer import Structurer

GenerateFn = Callable[[KnowledgeState, list[TranscriptEvent], SessionContext], QuestionCandidates]

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "example_schema.json"
SESSION_CONTEXT_PATH = REPO_ROOT / "fixtures" / "example_session_context.json"
TRANSCRIPT_PATH = REPO_ROOT / "fixtures" / "fake_transcript.jsonl"


def _load_session_context(path: Path) -> SessionContext:
    return SessionContext(**json.loads(path.read_text(encoding="utf-8")))


def run_session(
    structurer: Structurer,
    events: Iterator[TranscriptEvent],
    session_context: SessionContext,
    generate: GenerateFn = generate_questions,
) -> None:
    """대화형 루프: Enter=다음 턴, q=추천 질문(비파괴적), x=종료, EOF(Ctrl-D)=종료."""
    recent_events: list[TranscriptEvent] = []
    exhausted = False
    while True:
        prompt = "(Enter=다음 턴, q=추천 질문, x=종료) > " if not exhausted else "(q=추천 질문, x=종료) > "
        try:
            command = input(prompt).strip().lower()
        except EOFError:
            print("\n=== 세션 종료 ===")
            return

        if command == "x":
            print("=== 세션 종료 ===")
            return
        if command == "q":
            candidates = generate(structurer.state, recent_events, session_context)
            print("--- 추천 질문 ---")
            for c in candidates.candidates:
                print(f"[{c.type.value}] {c.text}")
            print("-----------------")
            continue
        if command != "":
            print("(알 수 없는 명령 — Enter/q/x만 지원)")
            continue
        if exhausted:
            print("(더 이상 턴이 없습니다 — q로 추천 질문, x로 종료)")
            continue

        try:
            event = next(events)
        except StopIteration:
            exhausted = True
            print("(전사 끝 — q로 추천 질문을 보거나 x로 종료하세요)")
            continue
        structurer.ingest_batch([event])
        recent_events.append(event)
        speaker = event.speaker.value if event.speaker else "?"
        print(f"[{event.turn_id} · {speaker}] {event.text}")
        covered = sum(
            1 for i in structurer.state.schema_items if i.status == CoverageStatus.COVERED
        )
        total = len(structurer.state.schema_items)
        contradiction_count = sum(1 for i in structurer.state.schema_items if i.contradictions)
        print(f"    (커버리지 {covered}/{total} 항목 · 모순 {contradiction_count}건)")


def main() -> None:
    schema = load_schema(SCHEMA_PATH)
    session_context = _load_session_context(SESSION_CONTEXT_PATH)
    structurer = Structurer(schema, session_context)
    events = FileReplaySource(TRANSCRIPT_PATH).stream()
    print(f"=== 세션 시작 · {schema.domain} ===")
    run_session(structurer, events, session_context)


if __name__ == "__main__":
    main()
