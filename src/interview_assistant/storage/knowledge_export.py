"""
KnowledgeState를 세션 디렉터리에 JSON+MD로 내보낸다(F6, design.md §4.3 스펙 그대로).

SessionLogWriter/AdoptionLogWriter와는 저장 패턴이 다르다 — 그 둘은 append-only 로그지만
KnowledgeState는 누적이 아니라 "현재 이해 상태"이므로, 매 구조화 틱마다 두 파일을
통째로 덮어쓴다(schema_loader.save_schema와 같은 전체 덮어쓰기 패턴). 원자적 쓰기
(tmp+rename)는 save_schema도 안 하는 것과 마찬가지로 이 개인용 로컬 도구 스코프 밖이라
의도적으로 생략한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from interview_assistant.contracts import KnowledgeState


def knowledge_state_to_dict(state: KnowledgeState) -> dict:
    """design.md §4.3의 지식 상태 JSON 스펙 그대로 직렬화한다."""
    return {
        "session_id": state.session_id,
        "overview": state.overview,
        "schema_items": [
            {
                "item_id": item.item_id,
                "label": item.label,
                "status": item.status.value,
                "summary": item.summary,
                "source_refs": list(item.source_refs),
                "contradictions": [
                    {"with_ref": c.with_ref, "note": c.note} for c in item.contradictions
                ],
            }
            for item in state.schema_items
        ],
    }


def _to_markdown(state: KnowledgeState, domain: str) -> str:
    lines = [f"# {domain} — 구조화 결과 ({state.session_id})", ""]
    if state.overview:
        lines.append("## 종합 정리")
        lines.append(state.overview)
        lines.append("")
    for item in state.schema_items:
        lines.append(f"## {item.label} ({item.status.value})")
        if item.summary:
            lines.append(item.summary)
            lines.append("")
        if item.source_refs:
            lines.append(f"(출처: {', '.join(item.source_refs)})")
        for contra in item.contradictions:
            lines.append(f"- ⚠ 모순: {contra.with_ref}와 상충 — {contra.note}")
        lines.append("")
    return "\n".join(lines) + "\n"


class KnowledgeStateExporter:
    """세션 디렉터리에 structured_state.json + structured_state.md를 덮어쓴다."""

    def __init__(self, directory: str | Path, domain: str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._domain = domain

    def export(self, state: KnowledgeState) -> None:
        (self._dir / "structured_state.json").write_text(
            json.dumps(knowledge_state_to_dict(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self._dir / "structured_state.md").write_text(
            _to_markdown(state, self._domain), encoding="utf-8"
        )


def load_exported_state(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
