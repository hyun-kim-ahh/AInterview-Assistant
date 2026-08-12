"""AssemblyAIFileSTTSource 계약 확인 테스트 (dev-plan 15단계, Milestone 1)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from interview_assistant.sources.stt.assemblyai_source import AssemblyAIFileSTTSource


def _fake_transcribe(utterances):
    def transcribe(path, config):
        return SimpleNamespace(utterances=utterances)

    return transcribe


def test_converts_utterances_to_transcript_events(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio-bytes")
    utterances = [
        SimpleNamespace(text="첫 번째 발화", start=500),   # ms
        SimpleNamespace(text="두 번째 발화", start=3200),  # ms
    ]
    transcribe = _fake_transcribe(utterances)

    events = list(AssemblyAIFileSTTSource(audio, transcribe=transcribe).stream())

    assert [e.text for e in events] == ["첫 번째 발화", "두 번째 발화"]
    assert [e.timestamp for e in events] == [0.5, 3.2]
    assert [e.turn_id for e in events] == ["t001", "t002"]


def test_no_utterances_yields_empty_stream(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio-bytes")
    transcribe = _fake_transcribe(None)

    events = list(AssemblyAIFileSTTSource(audio, transcribe=transcribe).stream())

    assert events == []


@pytest.mark.skipif(not os.environ.get("ASSEMBLYAI_API_KEY"), reason="실제 API 키 필요")
def test_live_transcribes_real_korean_audio_fixture():
    from pathlib import Path

    fixture = Path(__file__).parent.parent / "fixtures" / "stt_sample_ko.wav"
    events = list(AssemblyAIFileSTTSource(fixture).stream())

    full_text = " ".join(e.text for e in events)
    assert "로스팅" in full_text or "크랙" in full_text
