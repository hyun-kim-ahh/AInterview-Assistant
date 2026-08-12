"""KnowledgeStateExporter 계약 확인 테스트 — design.md §4.3 JSON 스펙 준수 + 덮어쓰기 확인."""

from __future__ import annotations

import json

from interview_assistant.contracts import (
    Contradiction,
    CoverageStatus,
    KnowledgeState,
    SchemaItem,
)
from interview_assistant.storage.knowledge_export import (
    KnowledgeStateExporter,
    knowledge_state_to_dict,
    load_exported_state,
)


def _make_state() -> KnowledgeState:
    return KnowledgeState(
        session_id="web_test",
        schema_items=[
            SchemaItem(
                item_id="item_01",
                label="로스팅 종료 시점 판단",
                status=CoverageStatus.PARTIAL,
                summary="향으로 판단",
                source_refs=["t002"],
                contradictions=[Contradiction(with_ref="t004", note="상충 메모")],
            )
        ],
    )


def test_knowledge_state_to_dict_matches_design_spec_shape():
    data = knowledge_state_to_dict(_make_state())

    assert data == {
        "session_id": "web_test",
        "overview": "",
        "schema_items": [
            {
                "item_id": "item_01",
                "label": "로스팅 종료 시점 판단",
                "status": "partial",
                "summary": "향으로 판단",
                "source_refs": ["t002"],
                "contradictions": [{"with_ref": "t004", "note": "상충 메모"}],
            }
        ],
    }


def _make_state_with_final_document() -> KnowledgeState:
    state = _make_state()
    state.overview = "## 종합 정리\n\n인터뷰 전체를 종합한 개요"
    state.schema_items[0].summary = "### 이 섹션 정리\n\n| 기준 | 예외 |\n|---|---|\n| 향 | 시간 |"
    return state


def test_knowledge_state_to_dict_includes_final_document_fields_when_populated():
    data = knowledge_state_to_dict(_make_state_with_final_document())

    assert data["overview"] == "## 종합 정리\n\n인터뷰 전체를 종합한 개요"
    assert "| 기준 | 예외 |" in data["schema_items"][0]["summary"]


def test_export_markdown_includes_final_document_fields_when_populated(tmp_path):
    exporter = KnowledgeStateExporter(tmp_path, domain="테스트 도메인")

    exporter.export(_make_state_with_final_document())

    md_text = (tmp_path / "structured_state.md").read_text(encoding="utf-8")
    # 마크다운 원문이 가공 없이 그대로 삽입돼야 한다(헤딩·표 문법 보존)
    assert "## 종합 정리\n\n인터뷰 전체를 종합한 개요" in md_text
    assert "| 기준 | 예외 |" in md_text


def test_export_writes_json_and_markdown_files(tmp_path):
    exporter = KnowledgeStateExporter(tmp_path, domain="테스트 도메인")

    exporter.export(_make_state())

    json_path = tmp_path / "structured_state.json"
    md_path = tmp_path / "structured_state.md"
    assert json_path.exists()
    assert md_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8")) == knowledge_state_to_dict(_make_state())
    md_text = md_path.read_text(encoding="utf-8")
    assert "테스트 도메인" in md_text
    assert "로스팅 종료 시점 판단" in md_text
    assert "향으로 판단" in md_text
    assert "상충 메모" in md_text


def test_export_overwrites_not_appends(tmp_path):
    exporter = KnowledgeStateExporter(tmp_path, domain="테스트 도메인")
    exporter.export(_make_state())

    second_state = KnowledgeState(session_id="web_test", schema_items=[])
    exporter.export(second_state)

    data = json.loads((tmp_path / "structured_state.json").read_text(encoding="utf-8"))
    assert data == knowledge_state_to_dict(second_state)


def test_export_creates_parent_directory_if_missing(tmp_path):
    directory = tmp_path / "nested" / "deeper"
    exporter = KnowledgeStateExporter(directory, domain="도메인")

    exporter.export(KnowledgeState(session_id="s1"))

    assert (directory / "structured_state.json").exists()


def test_load_exported_state_returns_none_for_missing_file(tmp_path):
    assert load_exported_state(tmp_path / "없음.json") is None


def test_load_exported_state_round_trips(tmp_path):
    exporter = KnowledgeStateExporter(tmp_path, domain="테스트 도메인")
    exporter.export(_make_state())

    loaded = load_exported_state(tmp_path / "structured_state.json")

    assert loaded == knowledge_state_to_dict(_make_state())
