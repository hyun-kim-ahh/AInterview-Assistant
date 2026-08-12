"""
TranscriptSource — 전사 입력 추상화 (STT 디커플링의 핵심).

LiveSTTSource / TextInputSource / FileReplaySource 모두 이 인터페이스를 구현하고,
동일하게 TranscriptEvent를 방출한다. 하류 모듈은 입력 출처를 구분하지 않는다.
따라서 STT가 없어도 Text/File 소스로 파이프라인 전체를 개발·테스트할 수 있다.

※ 시드다. 구현체는 dev-plan 3(TextInput) → 4(FileReplay) → 15(LiveSTT) 순으로 채운다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from interview_assistant.contracts import TranscriptEvent


class TranscriptSource(ABC):
    """전사 이벤트 스트림의 공통 인터페이스."""

    @abstractmethod
    def stream(self) -> Iterator[TranscriptEvent]:
        """TranscriptEvent를 순차적으로 방출한다.

        구현별 동작:
          - TextInputSource : 표준입력/함수인자로 받은 텍스트를 이벤트로 변환
          - FileReplaySource: jsonl 전사를 (원하면 타임스탬프 간격대로) 재생
          - LiveSTTSource   : 실시간 STT 결과를 이벤트로 변환
        """
        raise NotImplementedError
