"""
세션 메타데이터 — F0 화면에서 지난 세션 목록을 조회하기 위한 최소 정보.

세션 생성 시 1회만 쓰고 이후 갱신하지 않는다 — 진행 상황(턴 수, 최종 구조화 상태 등)은
transcript.jsonl/structured_state.json에서 언제든 파생할 수 있으므로 meta.json에
중복해서 담지 않는다.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SessionMeta:
    session_id: str
    domain: str
    has_schema: bool
    input_mode: str
    interview_goal: str
    started_at: str  # isoformat


def write_session_meta(directory: str | Path, meta: SessionMeta) -> None:
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)
    (p / "meta.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_session_meta(directory: str | Path) -> SessionMeta | None:
    path = Path(directory) / "meta.json"
    if not path.exists():
        return None
    try:
        return SessionMeta(**json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return None


def list_sessions(sessions_dir: str | Path) -> list[SessionMeta]:
    base = Path(sessions_dir)
    if not base.exists():
        return []
    metas = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        meta = read_session_meta(d)
        if meta is not None:
            metas.append(meta)
    return sorted(metas, key=lambda m: m.started_at, reverse=True)


def delete_session(sessions_dir: str | Path, session_id: str) -> bool:
    """세션 디렉터리를 통째로 삭제한다. sessions_dir 밖을 가리키거나 존재하지
    않으면 조용히 False(F0 화면은 삭제 실패를 별도 안내하지 않고 그냥 목록을
    그대로 다시 보여준다 — 이 개인용 로컬 도구의 UX 수준에 맞춤)."""
    base = Path(sessions_dir).resolve()
    target = (base / session_id).resolve()
    if target == base or base not in target.parents or not target.is_dir():
        return False
    shutil.rmtree(target)
    return True
