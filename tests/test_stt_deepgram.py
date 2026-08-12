"""DeepgramFileSTTSource 계약 확인 테스트 (dev-plan 15단계, Milestone 1)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from interview_assistant.sources.stt.deepgram_source import DeepgramFileSTTSource

FIXTURE_AUDIO = None  # 유닛 테스트는 가짜 client라 실제 오디오 파일이 필요 없음


def _fake_client(utterances):
    def transcribe_file(**kwargs):
        return SimpleNamespace(results=SimpleNamespace(utterances=utterances))

    return SimpleNamespace(
        listen=SimpleNamespace(
            v1=SimpleNamespace(media=SimpleNamespace(transcribe_file=transcribe_file))
        )
    )


def test_converts_utterances_to_transcript_events(tmp_path):
    # timestamp는 start가 아니라 end를 쓴다(2026-08-11) — 발화가 실제로 끝난
    # 시점을 기준으로 페이싱해야 발화 길이가 delay_seconds보다 길어도 화면에
    # 전사가 뜨는 시점이 말이 끝나기 전이 되는 일이 없다.
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio-bytes")
    utterances = [
        SimpleNamespace(transcript="첫 번째 발화", start=0.5, end=1.8),
        SimpleNamespace(transcript="두 번째 발화", start=3.2, end=5.0),
    ]
    client = _fake_client(utterances)

    events = list(DeepgramFileSTTSource(audio, client=client).stream())

    assert [e.text for e in events] == ["첫 번째 발화", "두 번째 발화"]
    assert [e.timestamp for e in events] == [1.8, 5.0]
    assert [e.turn_id for e in events] == ["t001", "t002"]


def test_no_utterances_yields_empty_stream(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio-bytes")
    client = _fake_client(None)

    events = list(DeepgramFileSTTSource(audio, client=client).stream())

    assert events == []


@pytest.mark.skipif(not os.environ.get("DEEPGRAM_API_KEY"), reason="실제 API 키 필요")
def test_live_transcribes_real_korean_audio_fixture():
    from pathlib import Path

    fixture = Path(__file__).parent.parent / "fixtures" / "stt_sample_ko.wav"
    events = list(DeepgramFileSTTSource(fixture).stream())

    full_text = " ".join(e.text for e in events)
    assert "로스팅" in full_text or "크랙" in full_text
