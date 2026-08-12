"""
OpenAIWhisperFileSTTSource — OpenAI whisper-1 배치 전사 API로 오디오 파일을 텍스트로
바꿔 TranscriptEvent 스트림으로 흘려보낸다 (dev-plan 15단계, Milestone 1 — 4벤더 비교 대상).

whisper-1을 쓴다(gpt-4o-transcribe 계열은 verbose_json/timestamp_granularities를
지원하지 않아 발화 단위 타임스탬프를 못 얻음 — 조사로 확인됨).

이 프로젝트의 기존 OpenAI 클라이언트(llm_client.get_client())는 OpenRouter로
base_url을 오버라이드한 것이라 오디오 전사에는 못 쓴다(OpenRouter는 전사 엔드포인트가
없음, 확인됨) — 여기서는 진짜 OpenAI 엔드포인트를 쓰는 별도 클라이언트를 만든다.
OPENAI_API_KEY는 OPENROUTER_API_KEY와 완전히 별개의 환경변수다.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from openai import OpenAI

from interview_assistant.contracts import TranscriptEvent
from interview_assistant.sources.base import TranscriptSource
from interview_assistant.sources.pacing import replay_with_pacing

MODEL = "whisper-1"
LANGUAGE = "ko"


class OpenAIWhisperFileSTTSource(TranscriptSource):
    def __init__(
        self,
        path: str | Path,
        *,
        client: OpenAI | None = None,
        realtime: bool = False,
    ) -> None:
        client = client or OpenAI()  # OPENAI_API_KEY 환경변수 사용, base_url 오버라이드 없음
        self._realtime = realtime
        with open(path, "rb") as f:
            response = client.audio.transcriptions.create(
                model=MODEL,
                file=f,
                language=LANGUAGE,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        segments = response.segments or []
        self._events = [
            TranscriptEvent(text=s.text, timestamp=s.start, turn_id=f"t{i:03d}")
            for i, s in enumerate(segments, start=1)
        ]

    def stream(self) -> Iterator[TranscriptEvent]:
        yield from replay_with_pacing(self._events, self._realtime)
