"""TextInputSource 계약 확인 테스트 (dev-plan 3단계)."""

from __future__ import annotations

from interview_assistant.sources.text_input import TextInputSource


def test_texts_multiple_events_in_order(capsys):
    texts = ["로스팅은 언제 멈추세요", "소리보다 향이 먼저예요", "타이머는 참고만 해요"]
    events = list(TextInputSource(texts).stream())

    assert [e.text for e in events] == texts
    assert [e.turn_id for e in events] == ["t001", "t002", "t003"]
    assert all(e.speaker is None for e in events)
    timestamps = [e.timestamp for e in events]
    assert all(isinstance(t, float) for t in timestamps)
    assert timestamps == sorted(timestamps)

    out = capsys.readouterr().out
    for e in events:
        assert e.turn_id in out
        assert e.text in out


def test_texts_skips_blank_entries_without_stopping():
    texts = ["소리를 들어요", "", "  ", "향이 올라와요"]
    events = list(TextInputSource(texts).stream())

    assert [e.text for e in events] == ["소리를 들어요", "향이 올라와요"]
    assert [e.turn_id for e in events] == ["t001", "t002"]


def test_texts_empty_iterable_yields_nothing():
    assert list(TextInputSource([]).stream()) == []


def test_stdin_eof_ends_stream(monkeypatch):
    responses = iter(["안녕하세요"])

    def fake_input(prompt=""):
        try:
            return next(responses)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    events = list(TextInputSource().stream())

    assert [e.text for e in events] == ["안녕하세요"]
    assert events[0].turn_id == "t001"


def test_stdin_blank_line_ends_stream(monkeypatch):
    responses = iter(["안녕하세요", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    events = list(TextInputSource().stream())

    assert [e.text for e in events] == ["안녕하세요"]
