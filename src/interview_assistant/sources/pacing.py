"""
replay_with_pacing — 이미 만들어진 TranscriptEvent 목록을 타임스탬프 간격대로(원하면)
재생하는 공용 로직 (dev-plan 15단계에서 추출).

원래 FileReplaySource(4단계)에만 있던 "realtime=True면 델타만큼 sleep+print" 로직을
여기로 뽑았다 — 15단계에서 오디오 파일 STT 소스 4개가 전부 "한 번에 배치로 전사받은
결과를 다시 흘려보낸다"는 동일한 필요를 갖게 되면서(2번째 이상 소비자가 생긴 시점에
공용 모듈로 추출하는 이 프로젝트의 기존 원칙 — llm_client.py/schema_loader.py와 동일),
FileReplaySource도 이 함수를 쓰도록 리팩터했다. 동작은 그대로다.

(2026-08-05 추가) 첫 이벤트는 "이전 타임스탬프가 없다"는 이유로 자기 자신의
timestamp를 무시하고 항상 지연 없이 즉시 나가던 버그가 있었다 — 오디오 파일
모드에서 세션 화면에 오디오 재생을 연동한 뒤, 녹음 시작부터 첫 발화 전까지 무음
구간이 있으면 전사가 그만큼 실제 오디오 위치보다 앞서 나가는 어긋남으로 드러났다.
prev_ts를 None이 아니라 0.0으로 시작해 첫 이벤트도 자기 timestamp만큼 정상적으로
지연되게 고쳤다.

(2026-08-06 추가) delay_seconds — 전사가 실제 오디오 재생 위치보다 일정 시간
늦게 나타나게 하는 선택적 지연(오디오 파일 테스트 모드 전용, 사용자 요청 — 실제
STT는 처리 지연이 있어서 말하자마자 바로 뜨는 것보다 약간 늦게 뜨는 게 자연스럽다).
이벤트의 timestamp 자체(데이터)는 그대로 두고, 맨 처음에 한 번만 이만큼 더
sleep해서 이후 모든 이벤트가 동일하게 그만큼씩 늦게 나가게 한다 — 이벤트 사이의
상대적 간격(delta)은 그대로 보존된다. 기본값 0이라 FileReplaySource 등 다른
호출자는 그대로 무영향.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

from interview_assistant.contracts import TranscriptEvent


def replay_with_pacing(
    events: Iterable[TranscriptEvent], realtime: bool, *, delay_seconds: float = 0.0
) -> Iterator[TranscriptEvent]:
    if realtime and delay_seconds > 0:
        time.sleep(delay_seconds)
    prev_ts = 0.0
    for event in events:
        if realtime:
            delta = event.timestamp - prev_ts
            if delta > 0:
                time.sleep(delta)
            print(f"[{event.turn_id} @ {event.timestamp:.1f}s] {event.text}")
        prev_ts = event.timestamp
        yield event
