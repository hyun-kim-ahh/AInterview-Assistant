"""
TextInputSource — 문자열/표준입력을 TranscriptEvent로 변환 (dev-plan 3단계, "얇은 관통").

계약(TranscriptSource) 확인용 최소 구현. 화자분리 없음 → speaker는 항상 None.
타임스탬프는 stream() 호출(세션 시작) 시점부터 time.monotonic() 기준 경과 초.
각 이벤트는 방출과 동시에 표준출력에 찍어 "돌아가는 상태"를 바로 눈으로 확인한다.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

from interview_assistant.contracts import TranscriptEvent
from interview_assistant.sources.base import TranscriptSource


class TextInputSource(TranscriptSource):
    """texts를 주면 그 목록을, 안 주면 표준입력을 이벤트로 변환한다.

    texts 모드: 각 원소를 strip 후 비어 있으면 조용히 건너뛰고 계속 소비한다.
    표준입력 모드: 빈 줄(Enter만) 또는 EOF(Ctrl-D / 파이프 종료) 둘 다 스트림을 끝낸다.
    """

    def __init__(self, texts: Iterable[str] | None = None) -> None:
        self._texts = texts

    def stream(self) -> Iterator[TranscriptEvent]:
        is_programmatic = self._texts is not None
        start = time.monotonic()
        n = 0
        for raw in (self._texts if is_programmatic else self._read_stdin()):
            text = raw.strip()
            if not text:
                if is_programmatic:
                    continue  # texts 모드: 빈 항목은 건너뛰고 계속
                return  # 표준입력 모드: 빈 줄 = 세션 종료 신호
            n += 1
            event = TranscriptEvent(
                text=text,
                timestamp=time.monotonic() - start,
                turn_id=f"t{n:03d}",
            )
            print(f"[{event.turn_id} @ {event.timestamp:.1f}s] {event.text}")
            yield event

    @staticmethod
    def _read_stdin() -> Iterator[str]:
        while True:
            try:
                yield input("> ")
            except EOFError:
                return
