"""
DeepgramFileSTTSource — Deepgram 배치(사전녹음) 전사 API로 오디오 파일을 텍스트로 바꿔
TranscriptEvent 스트림으로 흘려보낸다 (dev-plan 15단계, Milestone 1 — 4벤더 비교 대상).

생성자에서 API를 한 번만 호출해 결과를 캐싱한다 — stream()을 여러 번 불러도 유료 API가
반복 청구되지 않게 하기 위함(FileReplaySource의 "매번 다시 여는" 방식과 다른 이유:
로컬 파일 재열기는 공짜지만 STT API 호출은 아니다). 실제 재생(선택적 페이싱)은
sources/pacing.py의 공용 로직에 위임한다.

(2026-08-11 추가) TranscriptEvent.timestamp에 발화의 start가 아니라 end를 쓴다 —
utterances=True로 이미 요청하고 있어 Deepgram 응답에 각 발화의 end도 항상 같이
온다(SDK 타입 `ListenV1ResponseResultsUtterancesItem`에 start/end 둘 다 필드로
존재, 지금까지 end만 버려지고 있었음). sources/pacing.py의 delay_seconds(현재
1초, config.AUDIO_FILE_TRANSCRIPT_DELAY_SECONDS)가 "발화 시작+1초"였을 때는
발화 길이가 1초보다 길면 실제로 아직 말하는 중에 전사가 화면에 뜨는 부자연스러움이
있었음 — end 기준으로 바꾸면 "말이 끝난 뒤 1초"가 되어 이 문제가 해결된다.
페이싱 로직(replay_with_pacing) 자체는 무변경 — 델타 계산이 그냥 시작 시각 간격이
아니라 종료 시각 간격을 쓰게 될 뿐이다.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from deepgram import DeepgramClient

from interview_assistant.config import AUDIO_FILE_TRANSCRIPT_DELAY_SECONDS
from interview_assistant.contracts import TranscriptEvent
from interview_assistant.sources.base import TranscriptSource
from interview_assistant.sources.pacing import replay_with_pacing

MODEL = "nova-3"
LANGUAGE = "ko"


class DeepgramFileSTTSource(TranscriptSource):
    def __init__(
        self,
        path: str | Path,
        *,
        client: DeepgramClient | None = None,
        realtime: bool = False,
    ) -> None:
        client = client or DeepgramClient()
        self._realtime = realtime
        with open(path, "rb") as f:
            response = client.listen.v1.media.transcribe_file(
                request=f.read(),
                model=MODEL,
                language=LANGUAGE,
                smart_format=True,
                utterances=True,
            )
        utterances = response.results.utterances or []
        # timestamp에 u.start가 아니라 u.end를 쓴다(2026-08-11) — 발화가 실제로
        # 끝난 시점을 기준으로 페이싱해야, 발화 길이가 delay_seconds(1초)보다 긴
        # 경우에도 화면에 전사가 뜨는 시점이 그 사람이 말을 다 끝내기 전이 되는
        # 일이 없다(예전 u.start 기준으로는 4초짜리 발화가 시작+1초에 떠서 실제로
        # 아직 말하는 중에 전사가 나타나는 부자연스러움이 있었음).
        self._events = [
            TranscriptEvent(text=u.transcript, timestamp=u.end, turn_id=f"t{i:03d}")
            for i, u in enumerate(utterances, start=1)
        ]

    def stream(self) -> Iterator[TranscriptEvent]:
        yield from replay_with_pacing(
            self._events, self._realtime, delay_seconds=AUDIO_FILE_TRANSCRIPT_DELAY_SECONDS
        )
