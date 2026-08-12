"""
SessionLogWriter — 들어오는 TranscriptEvent를 세션 로그 파일에 append (dev-plan 5단계, F2).

읽기 기능은 없다 — 저장 포맷이 fixtures/fake_transcript.jsonl과 동일 JSONL 스키마이므로,
기록된 로그는 그대로 FileReplaySource(4단계)로 재생 가능하다(별도 리더 불필요).

"중단·복구 안전": append()마다 열고-쓰고-flush하고-닫아서, 인스턴스가 파일 핸들을
계속 들고 있지 않는다 — 호출 사이에 유실될 수 있는 인메모리 버퍼가 없다. flush는
이 파이썬 프로세스가 죽는 상황까지 방어한다; 전원 손실/OS 크래시 방어(os.fsync)는
이 개인용 로컬 도구의 요구 범위 밖이라 의도적으로 생략한다.

F2의 "세션 메타데이터" 저장은 이번 단계 범위 밖 — 세션 생명주기(SessionContext 시작/
종료)가 실제로 배선되는 이후 단계에서 다룬다.

`export_corrected_snapshot`(신규, 전사 재검토 기능)은 append 로그와 저장 패턴이
다르다 — transcript.jsonl(이 클래스의 원래 목적)은 STT가 실시간으로 받아적은 그대로의
append-only 원본 감사로그로 절대 안 건드리고, 교정이 적용된 최신 스냅샷은 별도
transcript_corrected.json에 매번 통째로 덮어쓴다(KnowledgeStateExporter가 이미 쓰는
"전체 스냅샷 덮어쓰기" 패턴과 동일).
"""

from __future__ import annotations

import json
from pathlib import Path

from interview_assistant.contracts import TranscriptEvent


class SessionLogWriter:
    """전사 이벤트를 jsonl 세션 로그 파일에 순서대로 append한다."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _to_record(self, event: TranscriptEvent) -> dict:
        return {
            "speaker": event.speaker.value if event.speaker is not None else None,
            "text": event.text,
            "timestamp": event.timestamp,
            "turn_id": event.turn_id,
            "corrected": event.corrected,
        }

    def append(self, event: TranscriptEvent) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._to_record(event), ensure_ascii=False) + "\n")
            f.flush()

    def export_corrected_snapshot(self, events: list[TranscriptEvent]) -> None:
        """교정이 반영된 전사 스냅샷을 transcript_corrected.json에 통째로 덮어쓴다.
        원본 append-only 로그(transcript.jsonl)는 그대로 둔다."""
        snapshot_path = self._path.with_name("transcript_corrected.json")
        snapshot_path.write_text(
            json.dumps([self._to_record(e) for e in events], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
