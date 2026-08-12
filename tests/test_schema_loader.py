"""schema_loader.load_schema 계약 확인 테스트 (dev-plan 10단계 — 2번째 소비자 추출)."""

from __future__ import annotations

from pathlib import Path

import pytest

from interview_assistant.contracts import InterviewSchema, SchemaItemDef
from interview_assistant.schema_loader import load_schema, save_schema

FIXTURE_PATH = Path(__file__).parent.parent / "schemas" / "example_schema.json"


def test_load_schema_from_real_fixture():
    schema = load_schema(FIXTURE_PATH)

    assert isinstance(schema, InterviewSchema)
    assert schema.domain == "숙련 커피 로스터 (예시 · 도메인 무관 데모용)"
    assert len(schema.items) == 4
    assert schema.items[0].item_id == "item_01"


def test_load_schema_missing_path_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_schema(Path("존재하지_않는_스키마.json"))


def test_load_schema_accepts_string_path():
    schema = load_schema(str(FIXTURE_PATH))

    assert schema.domain == "숙련 커피 로스터 (예시 · 도메인 무관 데모용)"


def test_save_schema_round_trips_with_load_schema(tmp_path):
    path = tmp_path / "saved_schema.json"
    schema = InterviewSchema(
        domain="한식 명인 (테스트)",
        items=[
            SchemaItemDef(item_id="item_01", label="장 담그기 시점", criteria="계절·온도 기준"),
            SchemaItemDef(item_id="item_02", label="발효 실패 진단", criteria="냄새·색 변화"),
        ],
    )

    save_schema(path, schema)
    loaded = load_schema(path)

    assert loaded == schema


def test_save_schema_creates_parent_directory_if_missing(tmp_path):
    path = tmp_path / "nested" / "deeper" / "schema.json"
    schema = InterviewSchema(domain="테스트 도메인", items=[])

    save_schema(path, schema)

    assert path.exists()
    assert load_schema(path).domain == "테스트 도메인"
