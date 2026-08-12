"""OpenAIWhisperFileSTTSource 계약 확인 테스트 (dev-plan 15단계, Milestone 1)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from interview_assistant.sources.stt.openai_whisper_source import OpenAIWhisperFileSTTSource


def _fake_client(segments):
    def create(**kwargs):
        return SimpleNamespace(segments=segments)

    return SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))


def test_converts_segments_to_transcript_events(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio-bytes")
    segments = [
        SimpleNamespace(text="첫 번째 발화", start=0.0),
        SimpleNamespace(text="두 번째 발화", start=4.1),
    ]
    client = _fake_client(segments)

    events = list(OpenAIWhisperFileSTTSource(audio, client=client).stream())

    assert [e.text for e in events] == ["첫 번째 발화", "두 번째 발화"]
    assert [e.timestamp for e in events] == [0.0, 4.1]
    assert [e.turn_id for e in events] == ["t001", "t002"]


def test_no_segments_yields_empty_stream(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio-bytes")
    client = _fake_client(None)

    events = list(OpenAIWhisperFileSTTSource(audio, client=client).stream())

    assert events == []


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="실제 API 키 필요")
def test_live_transcribes_real_korean_audio_fixture():
    from pathlib import Path

    fixture = Path(__file__).parent.parent / "fixtures" / "stt_sample_ko.wav"
    events = list(OpenAIWhisperFileSTTSource(fixture).stream())

    full_text = " ".join(e.text for e in events)
    assert "로스팅" in full_text or "크랙" in full_text
