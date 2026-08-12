"""
NaverClovaFileSTTSource — Naver CLOVA Speech 단문인식(short-sentence) API로 오디오 파일을
텍스트로 바꿔 TranscriptEvent 스트림으로 흘려보낸다 (dev-plan 15단계, Milestone 1 — 4벤더
비교 대상).

장문인식(long-sentence) API와 달리 도메인별 Invoke URL이 필요 없다 — 고정 공유
엔드포인트(`clovaspeech-gw.ncloud.com/recog/v1/stt`)에 Secret Key만으로 호출한다.
대신 제약이 있다: 최대 60초 오디오까지만 가능하고, 응답이 발화 단위 타임스탬프 없이
전체 텍스트 하나로만 온다(`{"text": ..., "quota": ...}`) — 그래서 TranscriptEvent를
턴 하나로만(t001, timestamp=0.0) 만든다. 더 긴 오디오·발화별 타임스탬프가 필요해지면
장문인식 도메인(Invoke URL 발급 필요)으로 바꿔야 한다.

공식 Python SDK가 없어 raw HTTP(requests)로 호출한다.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import requests

from interview_assistant.contracts import TranscriptEvent
from interview_assistant.sources.base import TranscriptSource
from interview_assistant.sources.pacing import replay_with_pacing

ENDPOINT = "https://clovaspeech-gw.ncloud.com/recog/v1/stt"
LANGUAGE = "Kor"
MAX_DURATION_SECONDS = 60

PostFn = Callable[..., Any]


class NaverClovaFileSTTSource(TranscriptSource):
    def __init__(
        self,
        path: str | Path,
        *,
        secret_key: str | None = None,
        post_fn: PostFn = requests.post,
        realtime: bool = False,
    ) -> None:
        secret_key = secret_key or os.environ["NAVER_CLOVA_SECRET_KEY"]
        self._realtime = realtime

        with open(path, "rb") as f:
            response = post_fn(
                ENDPOINT,
                params={"lang": LANGUAGE},
                headers={
                    "X-CLOVASPEECH-API-KEY": secret_key,
                    "Content-Type": "application/octet-stream",
                },
                data=f.read(),
            )
        response.raise_for_status()
        data = response.json()
        text = data.get("text", "")
        self._events = [TranscriptEvent(text=text, timestamp=0.0, turn_id="t001")] if text else []

    def stream(self) -> Iterator[TranscriptEvent]:
        yield from replay_with_pacing(self._events, self._realtime)
