"""
FileReplaySource — fake_transcript.jsonl을 한 줄씩(원하면 타임스탬프 간격대로) 재생 (dev-plan 4단계).

TextInputSource와 달리 turn_id/timestamp/speaker를 새로 만들지 않는다: 기록된 세션을
"재현"하는 것이 목적이므로 파일에 적힌 값을 그대로 방출한다. realtime=True일 때만
기록된 타임스탬프 간격만큼 실제로 sleep하고, 그때만 이벤트를 표준출력에 찍는다
(사람이 실시간 재생을 지켜볼 때만 의미 있는 피드백; 기본값(False)은 하류 개발/테스트가
반복 소비할 조용하고 빠른 모드).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from interview_assistant.contracts import Speaker, TranscriptEvent
from interview_assistant.sources.base import TranscriptSource
from interview_assistant.sources.pacing import replay_with_pacing


class FileReplaySource(TranscriptSource):
    """jsonl 전사 파일을 순서대로 재생한다. 매 stream() 호출마다 파일을 처음부터 다시 연다."""

    def __init__(self, path: str | Path, realtime: bool = False) -> None:
        self._path = Path(path)
        self._realtime = realtime

    def _read_events(self) -> Iterator[TranscriptEvent]:
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield TranscriptEvent(
                    text=obj["text"],
                    timestamp=float(obj["timestamp"]),
                    turn_id=obj["turn_id"],
                    speaker=(
                        Speaker(obj["speaker"])
                        if obj.get("speaker") is not None
                        else None
                    ),
                )

    def stream(self) -> Iterator[TranscriptEvent]:
        yield from replay_with_pacing(self._read_events(), self._realtime)
