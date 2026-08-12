"""SessionMeta 저장/조회 계약 확인 테스트 — F0 세션 목록 화면이 의존하는 최소 정보."""

from __future__ import annotations

from interview_assistant.storage.session_meta import (
    SessionMeta,
    delete_session,
    list_sessions,
    read_session_meta,
    write_session_meta,
)


def _make_meta(**overrides) -> SessionMeta:
    fields = {
        "session_id": "web_20260101_000000",
        "domain": "테스트 도메인",
        "has_schema": True,
        "input_mode": "fixture",
        "interview_goal": "목적",
        "started_at": "2026-01-01T00:00:00",
    }
    fields.update(overrides)
    return SessionMeta(**fields)


def test_write_and_read_session_meta_round_trips(tmp_path):
    directory = tmp_path / "web_20260101_000000"
    meta = _make_meta()

    write_session_meta(directory, meta)

    assert read_session_meta(directory) == meta


def test_read_session_meta_returns_none_when_missing(tmp_path):
    assert read_session_meta(tmp_path / "없음") is None


def test_read_session_meta_returns_none_for_corrupt_file(tmp_path):
    directory = tmp_path / "corrupt"
    directory.mkdir()
    (directory / "meta.json").write_text("이건 JSON이 아님", encoding="utf-8")

    assert read_session_meta(directory) is None


def test_write_session_meta_creates_parent_directory_if_missing(tmp_path):
    directory = tmp_path / "nested" / "deeper"

    write_session_meta(directory, _make_meta())

    assert (directory / "meta.json").exists()


def test_list_sessions_sorts_by_started_at_descending(tmp_path):
    write_session_meta(
        tmp_path / "s1", _make_meta(session_id="s1", started_at="2026-01-01T00:00:00")
    )
    write_session_meta(
        tmp_path / "s2", _make_meta(session_id="s2", started_at="2026-01-03T00:00:00")
    )
    write_session_meta(
        tmp_path / "s3", _make_meta(session_id="s3", started_at="2026-01-02T00:00:00")
    )

    result = list_sessions(tmp_path)

    assert [m.session_id for m in result] == ["s2", "s3", "s1"]


def test_list_sessions_skips_subdirectories_without_meta_json(tmp_path):
    write_session_meta(tmp_path / "has_meta", _make_meta(session_id="has_meta"))
    (tmp_path / "no_meta").mkdir()

    result = list_sessions(tmp_path)

    assert [m.session_id for m in result] == ["has_meta"]


def test_list_sessions_skips_corrupt_meta_json(tmp_path):
    write_session_meta(tmp_path / "good", _make_meta(session_id="good"))
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "meta.json").write_text("이건 JSON이 아님", encoding="utf-8")

    result = list_sessions(tmp_path)

    assert [m.session_id for m in result] == ["good"]


def test_list_sessions_returns_empty_list_for_nonexistent_directory(tmp_path):
    assert list_sessions(tmp_path / "존재하지_않음") == []


def test_delete_session_removes_directory_and_returns_true(tmp_path):
    write_session_meta(tmp_path / "web_20260101_000000", _make_meta())

    result = delete_session(tmp_path, "web_20260101_000000")

    assert result is True
    assert not (tmp_path / "web_20260101_000000").exists()


def test_delete_session_returns_false_for_nonexistent_session_id(tmp_path):
    assert delete_session(tmp_path, "없는_세션") is False


def test_delete_session_refuses_path_traversal_outside_sessions_dir(tmp_path):
    sessions_dir = tmp_path / "sessions"
    outside_dir = tmp_path / "outside"
    write_session_meta(outside_dir, _make_meta(session_id="outside"))
    sessions_dir.mkdir()

    result = delete_session(sessions_dir, "../outside")

    assert result is False
    assert outside_dir.exists()  # sessions_dir 밖은 절대 안 건드려야 함


def test_delete_session_refuses_sessions_dir_itself(tmp_path):
    write_session_meta(tmp_path / "s1", _make_meta(session_id="s1"))

    result = delete_session(tmp_path, ".")

    assert result is False
    assert (tmp_path / "s1").exists()
