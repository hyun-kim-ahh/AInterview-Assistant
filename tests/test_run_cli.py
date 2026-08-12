"""run_cli.py CLI 관통 마일스톤 계약 확인 테스트 (dev-plan 7단계).

scripts/run_cli.py는 패키지가 아니라 독립 스크립트라 importlib로 파일 경로에서
직접 로드한다(if __name__=="__main__" 가드 덕분에 임포트만으로는 main()이 실행되지 않음).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from interview_assistant.contracts import (
    InterviewSchema,
    QuestionCandidate,
    QuestionCandidates,
    QuestionType,
    SchemaItemDef,
    SessionContext,
    Speaker,
    TranscriptEvent,
)
from interview_assistant.structurer.structure_synthesizer import SynthesisResult
from interview_assistant.structurer.structurer import Structurer

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_cli.py"
_spec = importlib.util.spec_from_file_location("run_cli", SCRIPT_PATH)
run_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_cli)


def _make_schema() -> InterviewSchema:
    return InterviewSchema(
        domain="숙련 커피 로스터 (테스트용)",
        items=[
            SchemaItemDef(item_id="item_01", label="로스팅 종료 시점 판단"),
            SchemaItemDef(item_id="item_02", label="생두 품질 선별"),
        ],
    )


def _make_events() -> list[TranscriptEvent]:
    return [
        TranscriptEvent(
            text="로스팅은 언제 멈추세요?", timestamp=1.0, turn_id="t001", speaker=Speaker.INTERVIEWER
        ),
        TranscriptEvent(
            text="소리보다 향이 먼저예요", timestamp=2.0, turn_id="t002", speaker=Speaker.EXPERT
        ),
        TranscriptEvent(
            text="타이머는 안 보세요?", timestamp=3.0, turn_id="t003", speaker=Speaker.INTERVIEWER
        ),
    ]


def _no_updates(events, schema, session_context, state):
    return SynthesisResult(sections=[], new_sections=[])


def _fake_generate(state, recent_events, session_context):
    return QuestionCandidates(
        generated_at=recent_events[-1].turn_id if recent_events else "",
        candidates=[QuestionCandidate(type=QuestionType.PROBE, text="테스트용 추천 질문")],
    )


def _fake_input(commands):
    it = iter(commands)

    def _input(prompt=""):
        return next(it)

    return _input


def test_advances_and_prints_transcript_then_exits(monkeypatch, capsys):
    events = _make_events()
    structurer = Structurer(
        _make_schema(), SessionContext(session_id="test_session_01"), synthesize=_no_updates
    )
    monkeypatch.setattr("builtins.input", _fake_input(["", "", "", "x"]))

    run_cli.run_session(
        structurer,
        iter(events),
        SessionContext(session_id="test_session_01"),
        generate=_fake_generate,
    )

    out = capsys.readouterr().out
    assert out.index(events[0].text) < out.index(events[1].text) < out.index(events[2].text)
    assert "세션 종료" in out


def test_query_after_expert_turn_shows_probe_question(monkeypatch, capsys):
    events = _make_events()
    structurer = Structurer(
        _make_schema(), SessionContext(session_id="test_session_01"), synthesize=_no_updates
    )
    monkeypatch.setattr("builtins.input", _fake_input(["", "", "q", "x"]))

    run_cli.run_session(
        structurer,
        iter(events),
        SessionContext(session_id="test_session_01"),
        generate=_fake_generate,
    )

    out = capsys.readouterr().out
    assert "추천 질문" in out
    assert "[probe]" in out
    assert "테스트용 추천 질문" in out


def test_repeated_query_before_advancing_does_not_consume_turn(monkeypatch, capsys):
    events = _make_events()
    structurer = Structurer(
        _make_schema(), SessionContext(session_id="test_session_01"), synthesize=_no_updates
    )
    monkeypatch.setattr("builtins.input", _fake_input(["q", "q", "", "x"]))

    run_cli.run_session(
        structurer,
        iter(events),
        SessionContext(session_id="test_session_01"),
        generate=_fake_generate,
    )

    out = capsys.readouterr().out
    assert "[probe]" in out  # 가짜 generate가 항상 PROBE 반환
    assert events[0].text in out  # q 두 번 후에도 첫 턴이 정상 진행됨
    assert events[1].text not in out  # 두 번째 턴까진 진행 안 함


def test_exhausted_transcript_allows_query_and_exit(monkeypatch, capsys):
    events = _make_events()
    structurer = Structurer(
        _make_schema(), SessionContext(session_id="test_session_01"), synthesize=_no_updates
    )
    monkeypatch.setattr("builtins.input", _fake_input(["", "", "", "", "q", "x"]))

    run_cli.run_session(
        structurer,
        iter(events),
        SessionContext(session_id="test_session_01"),
        generate=_fake_generate,
    )

    out = capsys.readouterr().out
    assert "전사 끝" in out
    assert "추천 질문" in out
    assert "세션 종료" in out


def test_unknown_command_neither_crashes_nor_advances(monkeypatch, capsys):
    events = _make_events()
    structurer = Structurer(
        _make_schema(), SessionContext(session_id="test_session_01"), synthesize=_no_updates
    )
    monkeypatch.setattr("builtins.input", _fake_input(["asdf", "", "x"]))

    run_cli.run_session(
        structurer,
        iter(events),
        SessionContext(session_id="test_session_01"),
        generate=_fake_generate,
    )

    out = capsys.readouterr().out
    assert "알 수 없는 명령" in out
    assert events[0].text in out
