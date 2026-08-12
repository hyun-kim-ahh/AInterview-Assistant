"""스키마 파일 로딩 — run_cli.py와 웹 앱(app/web.py)이 공유하는 인프라 유틸리티."""

from __future__ import annotations

import json
from pathlib import Path

from interview_assistant.contracts import InterviewSchema, SchemaItemDef


def load_schema(path: str | Path) -> InterviewSchema:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return InterviewSchema(
        domain=data["domain"], items=[SchemaItemDef(**item) for item in data["items"]]
    )


def save_schema(path: str | Path, schema: InterviewSchema) -> None:
    """load_schema()가 기대하는 것과 정확히 같은 JSON 모양으로 저장한다(왕복 안전)."""
    data = {
        "domain": schema.domain,
        "items": [
            {
                "item_id": item.item_id,
                "label": item.label,
                "criteria": item.criteria,
            }
            for item in schema.items
        ],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
