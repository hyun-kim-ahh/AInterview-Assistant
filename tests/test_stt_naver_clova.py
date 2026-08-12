"""NaverClovaFileSTTSource 계약 확인 테스트 (dev-plan 15단계, Milestone 1).

단문인식 API는 발화 단위 타임스탬프 없이 전체 텍스트 하나만 돌려주므로, 결과는
항상 TranscriptEvent 0개 또는 1개(t001, timestamp=0.0)다.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from interview_assistant.sources.stt.naver_clova_source import (
    MAX_DURATION_SECONDS,
    NaverClovaFileSTTSource,
)

_FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "stt_sample_ko.wav"


def _fixture_duration_seconds() -> float:
    with wave.open(str(_FIXTURE_PATH), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _fake_post(text):
    def post_fn(url, params=None, headers=None, data=None):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"text": text, "quota": 2},
        )

    return post_fn


def test_converts_text_response_to_single_transcript_event(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio-bytes")
    post_fn = _fake_post("기본적으로 2차 크랙 소리를 듣죠")

    events = list(NaverClovaFileSTTSource(audio, secret_key="fake", post_fn=post_fn).stream())

    assert [e.text for e in events] == ["기본적으로 2차 크랙 소리를 듣죠"]
    assert [e.timestamp for e in events] == [0.0]
    assert [e.turn_id for e in events] == ["t001"]


def test_empty_text_yields_empty_stream(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio-bytes")
    post_fn = _fake_post("")

    events = list(NaverClovaFileSTTSource(audio, secret_key="fake", post_fn=post_fn).stream())

    assert events == []


@pytest.mark.skipif(not os.environ.get("NAVER_CLOVA_SECRET_KEY"), reason="실제 API 키 필요")
@pytest.mark.skipif(
    _fixture_duration_seconds() > MAX_DURATION_SECONDS,
    reason=(
        f"공용 오디오 fixture가 {MAX_DURATION_SECONDS}초 제한을 넘음 — "
        "단문인식 API 자체 한계라 스킵(Milestone 2에서 이미 제외된 벤더)"
    ),
)
def test_live_transcribes_real_korean_audio_fixture():
    """단문인식 API 배선(인증+응답 파싱)만 확인한다 — 도메인 용어 정확도는 검증하지 않는다.

    실측 결과 이 티어의 전사 품질이 Deepgram/AssemblyAI보다 눈에 띄게 떨어짐을 확인함
    (예: "2차 크랙"이 "이 책을 직접"으로 나오는 등) — Milestone 2 벤더 비교에서 이미
    반영된 사실이라 여기서 정확도까지 강하게 요구하진 않는다.
    """
    events = list(NaverClovaFileSTTSource(_FIXTURE_PATH).stream())

    assert len(events) == 1
    assert events[0].text.strip() != ""
