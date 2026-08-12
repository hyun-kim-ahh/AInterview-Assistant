"""DeepgramLiveMicSTTSource 계약 확인 테스트 (dev-plan 15단계, Milestone 3).

실제 마이크 캡처(sounddevice) 자체는 여전히 자동화 테스트 대상이 아니다 — 사람이 직접
말해보며 수동 검증해야 한다. 하지만 오디오 출처(audio_feeder)를 주입 가능하게 분리해둔
덕분에, 마이크 대신 오디오 파일을 실시간처럼 흘려보내는 make_file_audio_feeder로
실제 WebSocket 연결·메시지 파싱·큐 소비까지 스트리밍 코드 경로 전체를 라이브로 검증할
수 있다 — 이 파일의 라이브 테스트가 그것이다(순수 함수 _extract_transcript만 보던 것보다
훨씬 실질적).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from interview_assistant.sources.stt.deepgram_live_source import (
    DeepgramLiveMicSTTSource,
    _extract_transcript,
    make_file_audio_feeder,
)


def _make_message(*, is_final: bool, transcript: str):
    return SimpleNamespace(
        is_final=is_final,
        channel=SimpleNamespace(alternatives=[SimpleNamespace(transcript=transcript)]),
    )


def test_extracts_transcript_from_final_message():
    message = _make_message(is_final=True, transcript="로스팅은 언제 멈추세요")

    assert _extract_transcript(message) == "로스팅은 언제 멈추세요"


def test_ignores_interim_message():
    message = _make_message(is_final=False, transcript="로스팅은")

    assert _extract_transcript(message) is None


def test_ignores_final_message_with_blank_transcript():
    message = _make_message(is_final=True, transcript="   ")

    assert _extract_transcript(message) is None


@pytest.mark.skipif(not os.environ.get("DEEPGRAM_API_KEY"), reason="실제 API 키 필요")
def test_live_streams_audio_file_as_if_realtime_and_transcribes():
    """진짜 마이크 대신 stt_sample_ko.wav를 흘려보내 스트리밍 경로 전체를 검증.

    WebSocket 연결·오디오 전송·is_final 메시지 파싱·큐 소비까지 실제로 다 돈다 — 남는
    건 sounddevice의 실제 마이크 캡처뿐이고, 그건 여전히 사람이 브라우저에서 확인해야
    한다. fixture가 약 2분짜리라 실시간 그대로 재생하면 테스트가 너무 느려지므로,
    sleep_seconds를 짧게 줘서 "빨리 감기"로 흘려보낸다(같은 코드 경로, 시간만 단축).
    """
    from pathlib import Path

    fixture = Path(__file__).parent.parent / "fixtures" / "stt_sample_ko.wav"
    feeder = make_file_audio_feeder(fixture, chunk_seconds=0.5, sleep_seconds=0.01)
    source = DeepgramLiveMicSTTSource(audio_feeder=feeder)

    events = list(source.stream())

    assert len(events) >= 1
    full_text = " ".join(e.text for e in events)
    assert "로스팅" in full_text or "크랙" in full_text
