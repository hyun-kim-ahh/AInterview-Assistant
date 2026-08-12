"""
AssemblyAIFileSTTSource — AssemblyAI 배치 전사 API로 오디오 파일을 텍스트로 바꿔
TranscriptEvent 스트림으로 흘려보낸다 (dev-plan 15단계, Milestone 1 — 4벤더 비교 대상).

주의: AssemblyAI의 실시간(스트리밍) 인식은 한국어를 지원하지 않는다(영/스/불/독만,
2026-07 확인) — 이 소스는 배치(파일) 전용이며, 실시간 마이크 후보에서는 제외된다.

DeepgramFileSTTSource와 동일한 패턴: 생성자에서 API를 한 번만 호출해 결과를 캐싱한다.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import assemblyai as aai

from interview_assistant.contracts import TranscriptEvent
from interview_assistant.sources.base import TranscriptSource
from interview_assistant.sources.pacing import replay_with_pacing

LANGUAGE_CODE = "ko"

TranscribeFn = Callable[[str, "aai.TranscriptionConfig"], Any]


class AssemblyAIFileSTTSource(TranscriptSource):
    def __init__(
        self,
        path: str | Path,
        *,
        transcribe: TranscribeFn | None = None,
        realtime: bool = False,
    ) -> None:
        transcribe = transcribe or aai.Transcriber().transcribe
        self._realtime = realtime
        config = aai.TranscriptionConfig(language_code=LANGUAGE_CODE, speaker_labels=True)
        transcript = transcribe(str(path), config)
        utterances = transcript.utterances or []
        self._events = [
            TranscriptEvent(text=u.text, timestamp=u.start / 1000.0, turn_id=f"t{i:03d}")
            for i, u in enumerate(utterances, start=1)
        ]

    def stream(self) -> Iterator[TranscriptEvent]:
        yield from replay_with_pacing(self._events, self._realtime)
