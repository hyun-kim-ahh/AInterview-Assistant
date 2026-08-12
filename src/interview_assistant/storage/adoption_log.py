"""
AdoptionLogWriter — 질문 카드 채택/무시/편집을 로그 파일에 append (dev-plan 14단계, F11).

session_log.py(SessionLogWriter, 5단계)와 정확히 같은 패턴이다: append()마다 열고-쓰고-
flush하고-닫아서 인스턴스가 파일 핸들을 계속 들고 있지 않는다. os.fsync는 이 개인용
로컬 도구의 요구 범위 밖이라 의도적으로 생략한다.

`read_adoption_log`는 세션 목록/조회 화면(F0 한켠)이 지난 세션의 채택 기록을 읽기
전용으로 보여줘야 해서 추가됨 — 애초엔 "학습 신호용 기록이지 재생 대상이 아니다"였지만,
읽어서 보여주는 것과 재생(replay)은 다른 요구라 리더를 추가해도 원래 취지와 어긋나지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

from interview_assistant.contracts import AdoptionAction, AdoptionEvent, QuestionType


class AdoptionLogWriter:
    """채택/무시/편집 이벤트를 jsonl 로그 파일에 순서대로 append한다."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: AdoptionEvent) -> None:
        record = {
            "generated_at": event.generated_at,
            "question_type": event.question_type.value,
            "question_text": event.question_text,
            "action": event.action.value,
            "edited_text": event.edited_text,
            "target_item": event.target_item,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def read_adoption_log(path: str | Path) -> list[AdoptionEvent]:
    p = Path(path)
    if not p.exists():
        return []
    events = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        events.append(
            AdoptionEvent(
                generated_at=data["generated_at"],
                question_type=QuestionType(data["question_type"]),
                question_text=data["question_text"],
                action=AdoptionAction(data["action"]),
                edited_text=data.get("edited_text", ""),
                target_item=data.get("target_item"),
            )
        )
    return events
