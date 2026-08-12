"""
DeepgramLiveMicSTTSource — 실시간 마이크 입력을 Deepgram 스트리밍 API로 전사해
TranscriptEvent로 흘려보낸다 (dev-plan 15단계, Milestone 3).

마이크 캡처와 WebSocket 수신(Deepgram SDK의 콜백 기반 API)은 서로 다른 스레드에서
동시에 돈다 — 생산자(오디오 feeder가 conn.send_media로 보냄) / 소비자(stream()이 큐에서
확정 transcript를 꺼내 yield) 구조. 확정 결과(is_final=True)만 큐에 넣는다 — 중간 결과
(interim)는 화면에 보여줄 데가 없어(전사 페인은 확정 턴만 표시) 버린다.

오디오를 실제로 어디서 가져오는지(`audio_feeder`)는 주입 가능하게 분리했다 — 기본값은
진짜 마이크(`_feed_from_microphone`, sounddevice)지만, 테스트/검증에서는
`make_file_audio_feeder(path)`로 오디오 파일을 실시간처럼 시간 맞춰 흘려보내 마이크
없이도 WebSocket 연결·메시지 파싱·큐 소비까지 실제 스트리밍 코드 경로 전체를 검증할 수
있다(순수 함수 `_extract_transcript`만 테스트하던 것보다 훨씬 실질적인 검증).
"""

from __future__ import annotations

import os
import queue
import threading
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import certifi

# websockets(Deepgram 실시간 스트리밍이 쓰는 라이브러리)는 requests/httpx와 달리 certifi를
# 자동으로 안 쓰고 ssl.create_default_context()의 시스템 기본 인증서 경로를 본다 — python.org
# macOS 설치본은 이 경로가 비어있어 SSLCertVerificationError가 난다("Install Certificates.command"를
# 안 돌린 경우). 이 값이 이미 설정돼 있으면 존중하고, 없을 때만 certifi 번들로 채운다.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import sounddevice as sd
from deepgram import DeepgramClient
from deepgram.core.events import EventType

from interview_assistant.contracts import TranscriptEvent
from interview_assistant.sources.base import TranscriptSource

MODEL = "nova-3"
LANGUAGE = "ko"
SAMPLE_RATE = 16000
BLOCK_SIZE = 1600  # 0.1초 분량 (16000 * 0.1)

AudioFeeder = Callable[[Any, threading.Event], None]


def _extract_transcript(message: Any) -> str | None:
    """확정된(is_final) 메시지에서 발화 텍스트를 뽑는다. 확정 아니거나 빈 텍스트면 None."""
    if not getattr(message, "is_final", False):
        return None
    transcript = message.channel.alternatives[0].transcript
    return transcript if transcript.strip() else None


def _feed_from_microphone(conn: Any, stop_event: threading.Event) -> None:
    def audio_callback(indata, frames, time_info, status) -> None:
        conn.send_media(indata.tobytes())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=BLOCK_SIZE, callback=audio_callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


def make_file_audio_feeder(
    path: str | Path,
    *,
    chunk_seconds: float = 0.1,
    sleep_seconds: float | None = None,
    grace_period_seconds: float = 3.0,
) -> AudioFeeder:
    """마이크 대신 오디오 파일을 흘려보내는 feeder(테스트/검증용).

    chunk_seconds는 청크 하나가 담는 오디오 길이, sleep_seconds는 청크 사이 실제
    대기 시간이다 — 기본은 둘이 같아서(생략 시 sleep_seconds=chunk_seconds) 진짜
    실시간처럼 재생된다. 자동화 테스트에서 긴 파일을 오래 기다리지 않고도 같은
    스트리밍 코드 경로(WebSocket 연결·send_media·is_final 파싱)를 검증하려면
    sleep_seconds를 chunk_seconds보다 짧게 줘서 "빨리 감기"로 흘려보내면 된다.

    파일을 다 보내고 나면 grace_period_seconds만큼 기다려(Deepgram이 마지막 발화를
    확정 지을 시간을 줌) stop_event를 스스로 세팅한다 — 별도 종료 트리거 없이
    list(source.stream())만으로 끝까지 돌 수 있다.
    """
    if sleep_seconds is None:
        sleep_seconds = chunk_seconds

    def feeder(conn: Any, stop_event: threading.Event) -> None:
        with wave.open(str(path), "rb") as wf:
            chunk_frames = int(wf.getframerate() * chunk_seconds)
            while not stop_event.is_set():
                data = wf.readframes(chunk_frames)
                if not data:
                    break
                conn.send_media(data)
                time.sleep(sleep_seconds)
        time.sleep(grace_period_seconds)
        stop_event.set()

    return feeder


class DeepgramLiveMicSTTSource(TranscriptSource):
    def __init__(
        self,
        *,
        client: DeepgramClient | None = None,
        audio_feeder: AudioFeeder | None = None,
    ) -> None:
        self._client = client or DeepgramClient()
        self._audio_feeder = audio_feeder or _feed_from_microphone
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def stream(self) -> Iterator[TranscriptEvent]:
        start_time = time.monotonic()
        turn_counter = 0

        with self._client.listen.v1.connect(
            model=MODEL, language=LANGUAGE, encoding="linear16", sample_rate=SAMPLE_RATE,
            smart_format=True,
        ) as conn:
            def on_message(message: Any) -> None:
                transcript = _extract_transcript(message)
                if transcript is not None:
                    self._queue.put(transcript)

            conn.on(EventType.MESSAGE, on_message)
            # start_listening()은 논블로킹이 아니라 수신 루프를 호출 스레드에서 그대로
            # 블로킹 실행한다(SDK 문서 인상과 달리 실측으로 확인됨) — 그래서 별도
            # 스레드로 돌리지 않으면 feeder_thread.start() 줄에 영영 도달하지 못해
            # 오디오를 한 바이트도 못 보내고 Deepgram의 무응답 타임아웃(~12초)으로
            # 연결이 끊긴다.
            threading.Thread(target=conn.start_listening, daemon=True).start()

            feeder_thread = threading.Thread(
                target=self._audio_feeder, args=(conn, self._stop_event), daemon=True
            )
            feeder_thread.start()

            while not self._stop_event.is_set():
                try:
                    transcript = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                turn_counter += 1
                yield TranscriptEvent(
                    text=transcript,
                    timestamp=time.monotonic() - start_time,
                    turn_id=f"t{turn_counter:03d}",
                )
