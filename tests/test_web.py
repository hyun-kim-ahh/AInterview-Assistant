"""웹 UI(dev-plan 10단계) 라우팅·배선 계약 확인 테스트.

flask test_client()만 사용 — 실제 서버 프로세스·네트워크 없음.

Structurer는 반드시 web.Structurer 이름 자체를 가짜 클래스로 통째로 치환한다.
Structurer.__init__의 기본값 classify=classify_turns는 정의 시점에 한 번 바인딩되므로
(late-binding), classify_turns 정의부를 패치해도 이미 바인딩된 기본값엔 영향이 없다 —
그래서 클래스 자체를 바꿔치기해 __init__이 아예 다르게 동작하도록 한다. 패치는 반드시
POST /session 이전에 걸어야 한다(그 안에서 인스턴스가 한 번만 생성되기 때문).
generate_questions는 라우트가 매 요청마다 모듈 전역에서 새로 조회해 호출하므로
web.generate_questions를 패치하면 그대로 먹힌다 — 같은 함정이 없다.
load_schema/FileReplaySource는 순수 파일 I/O라 패치하지 않고 실제 fixture로 검증한다.
"""

from __future__ import annotations

import threading
import time

import pytest

from interview_assistant.app import web
from interview_assistant.contracts import (
    AdoptionAction,
    CoverageStatus,
    KnowledgeState,
    QuestionCandidate,
    QuestionCandidates,
    QuestionType,
    SchemaItem,
    TranscriptEvent,
)


class _FakeAdoptionLogWriter:
    def __init__(self, path):
        self.path = path
        self.events = []

    def append(self, event):
        self.events.append(event)


class _FakeSessionLogWriter:
    def __init__(self, path):
        self.path = path
        self.events = []
        self.corrected_snapshots = []

    def append(self, event):
        self.events.append(event)

    def export_corrected_snapshot(self, events):
        self.corrected_snapshots.append(list(events))


class _FakeKnowledgeStateExporter:
    def __init__(self, directory, domain):
        self.directory = directory
        self.domain = domain
        self.exported = []

    def export(self, state):
        self.exported.append(state)


class _FakeStructurer:
    def __init__(self, schema, session_context):
        self.schema = schema
        self.session_context = session_context
        self.ingested = []
        self._state = KnowledgeState(
            session_id=session_context.session_id,
            schema_items=[
                SchemaItem(item_id=d.item_id, label=d.label) for d in schema.items
            ],
        )

    @property
    def state(self):
        return self._state

    def ingest_batch(self, events):
        self.ingested.extend(events)

    def seed_outline(self, *, propose=None):
        pass  # create_session()이 항상 _start_outline_seeding_worker를 띄우므로 필수 스텁


@pytest.fixture
def client(monkeypatch, tmp_path):
    # create_session()이 백그라운드 데몬 스레드(구조화, 목차 시딩)를 띄운다 —
    # 기본적으로 no-op 처리해 대부분의 테스트가 실 스레드/시간 대기 없이 돈다.
    # 스레딩 자체를 검증하는 테스트만 이 패치를 명시적으로 되돌린다.
    monkeypatch.setattr(web, "_start_structuring_worker", lambda app_state: None)
    monkeypatch.setattr(web, "_start_outline_seeding_worker", lambda app_state: None)
    monkeypatch.setattr(web, "_start_question_worker", lambda app_state: None)
    monkeypatch.setattr(web, "_start_summary_worker", lambda app_state: None)
    monkeypatch.setattr(web, "_start_final_question_worker", lambda app_state: None)
    monkeypatch.setattr(web, "_start_finalize_worker", lambda app_state: None)
    # 전사 재검토(review_transcript)도 기본은 no-op — 안 하면 transcript_log에 턴 2개
    # 이상이 있는 상태에서 구조화 관련 테스트를 돌릴 때 진짜 OpenRouter 호출을 시도한다
    # (review_transcript는 Structurer를 거치지 않는 독립 함수라 _FakeStructurer로는 못 막음).
    monkeypatch.setattr(web, "review_transcript", lambda *a, **k: [])
    # create_session()이 실제 AdoptionLogWriter/SessionLogWriter/KnowledgeStateExporter를
    # 생성하고 write_session_meta를 호출해 리포 sessions/에 파일을 쓴다 — 테스트가 실
    # 파일을 남기지 않도록 기본적으로 가짜로 치환한다. SESSIONS_DIR 자체도 tmp_path로
    # 돌려서 index()의 list_sessions() 호출이 실제 리포의 sessions/를 스캔하지 않게 한다.
    monkeypatch.setattr(web, "AdoptionLogWriter", _FakeAdoptionLogWriter)
    monkeypatch.setattr(web, "SessionLogWriter", _FakeSessionLogWriter)
    monkeypatch.setattr(web, "KnowledgeStateExporter", _FakeKnowledgeStateExporter)
    monkeypatch.setattr(web, "write_session_meta", lambda *a, **k: None)
    monkeypatch.setattr(web, "SESSIONS_DIR", tmp_path)
    web._state = None
    with web.app.test_client() as c:
        yield c
    web._state = None


def _create_session(client, **overrides):
    form = {
        "schema_path": str(web.DEFAULT_SCHEMA_PATH),
        "interview_goal": "목적",
        "expert_profile": "프로필",
        "focus_notes": "포커스",
    }
    form.update(overrides)
    return client.post("/session", data=form, follow_redirects=True)


def test_index_shows_setup_form_when_no_session(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"schema_path" in response.data
    assert "다음 턴".encode() not in response.data


def test_create_session_with_valid_schema_renders_session_view(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)

    response = _create_session(client)

    assert response.status_code == 200
    assert "숙련 커피 로스터".encode() in response.data
    assert web._state is not None


def test_create_session_with_bad_schema_path_shows_error_and_creates_no_session(client):
    response = client.post(
        "/session",
        data={
            "schema_path": "존재하지_않는_스키마.json",
            "interview_goal": "",
            "expert_profile": "",
            "focus_notes": "",
        },
    )

    assert response.status_code == 400
    assert "스키마를 불러올 수 없습니다".encode() in response.data
    assert web._state is None


def test_next_turn_advances_and_appends_transcript(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)

    response = client.post("/next", follow_redirects=True)

    assert "로스팅을 언제 멈춰야".encode() in response.data
    # /next는 이제 원문만 append한다 — 구조화(_run_structuring_pass)는 별도 주기 루프 몫.
    assert web._state.structurer.ingested == []
    web._run_structuring_pass(web._state)
    assert len(web._state.structurer.ingested) == 1


def test_next_turn_exhaustion_marks_ended_without_crashing(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)

    for _ in range(11):  # fixture엔 턴 10개뿐 — 한 번 더 눌러 소진 처리 확인
        response = client.post("/next", follow_redirects=True)

    assert response.status_code == 200
    assert "전사 끝".encode() in response.data
    assert web._state.exhausted is True


def test_ask_generates_questions_synchronously_and_displays_them(client, monkeypatch):
    # 2026-07-30부터: /ask는 버튼을 누른 시점에 그 자리에서 generate_questions를
    # 호출한다(토큰 효율 위해 상시 배경 생성을 되돌림) — 더 이상 캐시 표출 전용이 아님.
    # (2026-08-11부터) /ask는 redirect가 아니라 /status와 같은 모양의 JSON을
    # 반환한다(전체 페이지 리로드가 audio_file 모드의 <audio> 태그를 매번
    # 재생성해 재생을 끊기게 하던 문제 때문 — session.html이 fetch로 받아 카드만
    # innerHTML 교체).
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    fresh_result = QuestionCandidates(
        generated_at="t001",
        candidates=[QuestionCandidate(type=QuestionType.PROBE, text="방금 생성된 질문")],
    )
    monkeypatch.setattr(web, "generate_questions", lambda *a, **k: fresh_result)
    _create_session(client)

    response = client.post("/ask")

    assert web._state.latest_candidates == fresh_result
    assert response.status_code == 200
    data = response.get_json()
    assert data["candidates"]["candidates"] == [
        {"type": "probe", "text": "방금 생성된 질문", "target_item": ""}
    ]


def test_ask_regenerates_every_time_it_is_pressed(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    calls = []

    def _spy_generate(*a, **k):
        calls.append(1)
        return QuestionCandidates(generated_at=f"t{len(calls):03d}", candidates=[])

    monkeypatch.setattr(web, "generate_questions", _spy_generate)
    _create_session(client)

    client.post("/ask")
    client.post("/ask")

    assert len(calls) == 2  # 누를 때마다 다시 생성됨 — "새 턴 있을 때만" 조건 없음


def test_refresh_candidates_serializes_concurrent_calls(client, monkeypatch):
    # (2026-08-11) question_generation_lock 회귀 테스트 — 배경 트리거(_question_loop)의
    # generate_questions 호출이 아직 진행 중일 때 /end의 마지막 질문 생성처럼 두 번째
    # 호출이 들어오면, 첫 호출이 완전히 끝날 때까지 기다린 뒤에 실행돼야 한다(겹쳐서
    # 돌지 않음) — 그래야 나중에 시작한(=더 "최종"에 가까운) 호출의 결과가 덮어써지지
    # 않고 항상 마지막에 반영된다.
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    calls = []

    def _slow_generate(*a, **k):
        calls.append("start")
        time.sleep(0.1)
        calls.append("finish")
        return QuestionCandidates(generated_at=f"t{len(calls):03d}", candidates=[])

    monkeypatch.setattr(web, "generate_questions", _slow_generate)
    _create_session(client)
    app_state = web._state

    first_call = threading.Thread(target=web._refresh_candidates, args=(app_state,))
    first_call.start()
    time.sleep(0.03)  # 첫 호출이 락을 잡고 generate_questions의 sleep에 들어갈 시간을 준다

    web._refresh_candidates(app_state)  # 락이 풀릴 때까지 여기서 블록
    first_call.join()

    # 겹쳤다면 ["start", "start", "finish", "finish"]류로 나왔을 것 — 직렬화됐다면
    # 한 호출이 완전히 끝나야만 다음 호출이 시작된다.
    assert calls == ["start", "finish", "start", "finish"]


def test_new_session_resets_state_and_allows_second_session(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    assert web._state is not None

    client.post("/new", follow_redirects=True)
    assert web._state is None

    response = _create_session(client)
    assert response.status_code == 200
    assert web._state is not None


def test_next_and_ask_before_session_started_redirect_safely(client):
    next_response = client.post("/next", follow_redirects=True)
    ask_response = client.post("/ask", follow_redirects=True)

    assert next_response.status_code == 200
    assert ask_response.status_code == 200
    assert b"schema_path" in next_response.data
    assert b"schema_path" in ask_response.data
    assert web._state is None


def test_session_view_shows_coverage_panel_with_all_items_uncovered(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)

    response = _create_session(client)

    assert "로스팅 종료 시점 판단".encode() in response.data
    assert response.data.count(b"status-uncovered") == 4  # fixture 스키마 항목 4개


def test_coverage_panel_reflects_item_status(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.structurer.state.schema_items[0].status = CoverageStatus.COVERED

    response = client.get("/")

    assert b"status-covered" in response.data
    assert response.data.count(b"status-uncovered") == 3
    assert "3건 대기".encode() in response.data


def test_coverage_panel_shows_contradiction_badge(client, monkeypatch):
    from interview_assistant.contracts import Contradiction

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.structurer.state.schema_items[0].contradictions.append(
        Contradiction(with_ref="t004", note="테스트 모순")
    )

    response = client.get("/")

    assert b"item-contradiction-badge" in response.data
    assert "⚠ 모순 1건".encode() in response.data


def test_regenerate_once_calls_generate_questions_with_full_transcript_log(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    captured = {}

    def _spy_generate(state, recent_events, session_context, **kwargs):
        captured["recent_events"] = recent_events
        return QuestionCandidates(generated_at="", candidates=[])

    monkeypatch.setattr(web, "generate_questions", _spy_generate)
    _create_session(client)
    client.post("/next", follow_redirects=True)
    client.post("/next", follow_redirects=True)

    web._regenerate_once(web._state)

    assert [e.turn_id for e in captured["recent_events"]] == ["t001", "t002"]


def test_create_session_starts_outline_seeding_worker(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    captured = {}

    def _spy_start_worker(app_state):
        captured["app_state"] = app_state

    monkeypatch.setattr(web, "_start_outline_seeding_worker", _spy_start_worker)

    _create_session(client)

    assert captured["app_state"] is web._state


def test_adopt_question_logs_adopted_action_when_text_unchanged(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)

    client.post(
        "/adopt",
        data={
            "question_type": "probe",
            "original_text": "원래 질문",
            "text": "원래 질문",
            "target_item": "item_01",
        },
        follow_redirects=True,
    )

    logged = web._state.adoption_log.events
    assert len(logged) == 1
    assert logged[0].action == AdoptionAction.ADOPTED
    assert logged[0].question_text == "원래 질문"
    assert logged[0].edited_text == ""
    assert logged[0].target_item == "item_01"


def test_adopt_question_logs_edited_action_when_text_changed(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)

    client.post(
        "/adopt",
        data={
            "question_type": "gap",
            "original_text": "원래 질문",
            "text": "수정된 질문",
            "target_item": "",
        },
        follow_redirects=True,
    )

    logged = web._state.adoption_log.events
    assert len(logged) == 1
    assert logged[0].action == AdoptionAction.EDITED
    assert logged[0].question_text == "원래 질문"
    assert logged[0].edited_text == "수정된 질문"
    assert logged[0].target_item is None


def test_adopt_question_adds_to_kept_questions_list(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)

    client.post(
        "/adopt",
        data={
            "question_type": "probe",
            "original_text": "원래 질문",
            "text": "원래 질문",
            "target_item": "item_01",
        },
        follow_redirects=True,
    )
    client.post(
        "/adopt",
        data={
            "question_type": "gap",
            "original_text": "두 번째 질문",
            "text": "두 번째 질문",
        },
        follow_redirects=True,
    )

    kept = web._state.kept_questions
    assert len(kept) == 2
    assert kept[0].question_text == "원래 질문"
    assert kept[1].question_text == "두 번째 질문"
    # adoption_log(파일 기록)에도 똑같이 남아야 함
    assert len(web._state.adoption_log.events) == 2


def test_adopt_question_returns_status_json_instead_of_redirect(client, monkeypatch):
    # (2026-08-11) /adopt도 /ask와 같은 이유로 redirect가 아니라 JSON을 반환한다 —
    # 전체 페이지 리로드가 audio_file 모드의 <audio> 태그를 재생성해 재생을
    # 끊기게 하던 문제 때문(session.html이 fetch로 받아 킵한 질문 패널만 갱신).
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)

    response = client.post(
        "/adopt",
        data={
            "question_type": "probe",
            "original_text": "원래 질문",
            "text": "원래 질문",
            "target_item": "item_01",
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["kept_questions"] == [{"question_type": "probe", "text": "원래 질문"}]


def test_session_view_shows_kept_question_count_and_toggle(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)

    client.post(
        "/adopt",
        data={
            "question_type": "probe",
            "original_text": "킵할 질문",
            "text": "킵할 질문",
        },
        follow_redirects=True,
    )

    response = client.get("/")

    assert "킵한 질문 (1)".encode() in response.data
    assert "킵할 질문".encode() in response.data
    assert b'id="toggle-kept-btn"' in response.data
    assert b'id="kept-questions-panel"' in response.data


class _FakeDeepgramFileSTTSource:
    events = [TranscriptEvent(text="파일에서 나온 첫 턴", timestamp=0.0, turn_id="t001")]

    def __init__(self, path, *, realtime=True):
        self.path = path
        self.realtime = realtime

    def stream(self):
        return iter(self.events)


class _FailingDeepgramFileSTTSource:
    def __init__(self, path, *, realtime=True):
        raise OSError(f"파일을 열 수 없습니다: {path}")


class _FakeDeepgramLiveMicSTTSource:
    def __init__(self):
        self.stopped = False

    def stream(self):
        return iter(())

    def stop(self):
        self.stopped = True


def test_create_session_default_fixture_mode_has_auto_ingest_false(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)

    _create_session(client)

    assert web._state.auto_ingest is False
    assert web._state.stt_source is None


def test_create_session_with_audio_file_mode_sets_auto_ingest(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "DeepgramFileSTTSource", _FakeDeepgramFileSTTSource)
    monkeypatch.setattr(web, "_start_ingestion_worker", lambda app_state: None)

    response = _create_session(client, input_mode="audio_file", audio_path="/fake/audio.wav")

    assert response.status_code == 200
    assert web._state.auto_ingest is True
    assert web._state.stt_source is None


def test_create_session_with_audio_file_mode_bad_path_shows_error(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "DeepgramFileSTTSource", _FailingDeepgramFileSTTSource)

    response = client.post(
        "/session",
        data={
            "schema_path": str(web.DEFAULT_SCHEMA_PATH),
            "interview_goal": "",
            "expert_profile": "",
            "focus_notes": "",
            "input_mode": "audio_file",
            "audio_path": "/no/such/file.wav",
        },
    )

    assert response.status_code == 400
    assert "오디오 파일을 전사할 수 없습니다".encode() in response.data
    assert web._state is None


def test_create_session_with_audio_file_mode_stores_input_mode_and_audio_path(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "DeepgramFileSTTSource", _FakeDeepgramFileSTTSource)
    monkeypatch.setattr(web, "_start_ingestion_worker", lambda app_state: None)

    _create_session(client, input_mode="audio_file", audio_path="/fake/audio.wav")

    assert web._state.input_mode == "audio_file"
    assert web._state.audio_path == "/fake/audio.wav"


def test_create_session_with_audio_file_mode_resolves_relative_audio_path(client, monkeypatch):
    # send_file은 상대 경로를 프로세스 CWD가 아니라 Flask 앱 root_path 기준으로 찾는다 —
    # STT가 여는 경로와 /audio가 서빙하는 경로가 어긋나지 않도록 절대 경로로 정규화해야 함.
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "DeepgramFileSTTSource", _FakeDeepgramFileSTTSource)
    monkeypatch.setattr(web, "_start_ingestion_worker", lambda app_state: None)

    _create_session(client, input_mode="audio_file", audio_path="schemas/example_schema.json")

    from pathlib import Path

    assert Path(web._state.audio_path).is_absolute()


def test_session_audio_returns_404_when_no_session(client):
    response = client.get("/audio")

    assert response.status_code == 404


def test_session_audio_returns_404_for_fixture_mode_session(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)

    response = client.get("/audio")

    assert response.status_code == 404


def test_session_audio_serves_file_for_audio_file_mode_session(client, monkeypatch, tmp_path):
    audio_bytes = b"RIFF....WAVEfmt fake audio bytes"
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(audio_bytes)

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "DeepgramFileSTTSource", _FakeDeepgramFileSTTSource)
    monkeypatch.setattr(web, "_start_ingestion_worker", lambda app_state: None)
    _create_session(client, input_mode="audio_file", audio_path=str(audio_file))

    response = client.get("/audio")

    assert response.status_code == 200
    assert response.data == audio_bytes


def test_create_session_with_live_mic_mode_stores_stt_source(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "DeepgramLiveMicSTTSource", _FakeDeepgramLiveMicSTTSource)
    monkeypatch.setattr(web, "_start_ingestion_worker", lambda app_state: None)

    _create_session(client, input_mode="live_mic")

    assert web._state.auto_ingest is True
    assert isinstance(web._state.stt_source, _FakeDeepgramLiveMicSTTSource)


def test_next_turn_noop_when_auto_ingest_true(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "DeepgramLiveMicSTTSource", _FakeDeepgramLiveMicSTTSource)
    monkeypatch.setattr(web, "_start_ingestion_worker", lambda app_state: None)
    _create_session(client, input_mode="live_mic")

    client.post("/next", follow_redirects=True)

    assert web._state.structurer.ingested == []
    assert web._state.transcript_log == []


def test_new_session_stops_live_mic_source(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "DeepgramLiveMicSTTSource", _FakeDeepgramLiveMicSTTSource)
    monkeypatch.setattr(web, "_start_ingestion_worker", lambda app_state: None)
    _create_session(client, input_mode="live_mic")
    mic_source = web._state.stt_source

    client.post("/new", follow_redirects=True)

    assert mic_source.stopped is True
    assert web._state is None


def test_end_interview_sets_ended_and_generates_final_candidates_in_background(client, monkeypatch):
    def _really_start_finalize_worker(app_state):
        threading.Thread(target=web._finalize_interview, args=(app_state,), daemon=True).start()

    def _really_start_final_question_worker(app_state):
        threading.Thread(target=web._refresh_candidates, args=(app_state,), daemon=True).start()

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    final_result = QuestionCandidates(
        generated_at="final",
        candidates=[QuestionCandidate(type=QuestionType.PROBE, text="마무리 질문")],
    )
    monkeypatch.setattr(web, "generate_questions", lambda *a, **k: final_result)
    monkeypatch.setattr(web, "_start_finalize_worker", _really_start_finalize_worker)
    monkeypatch.setattr(web, "_start_final_question_worker", _really_start_final_question_worker)
    _create_session(client)

    client.post("/end", follow_redirects=True)
    time.sleep(0.05)  # 백그라운드 스레드가 돌 시간을 준다

    assert web._state is not None  # /new와 달리 세션이 그대로 유지돼야 함
    assert web._state.ended is True
    assert web._state.latest_candidates == final_result


def test_end_interview_keeps_session_screen_and_features_working(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "generate_questions", lambda *a, **k: QuestionCandidates(generated_at="", candidates=[]))
    _create_session(client)

    client.post("/end", follow_redirects=True)
    response = client.get("/")

    # 화면이 setup.html(F0)로 안 넘어가고 session.html 그대로인지 확인
    assert "인터뷰 종료".encode() in response.data
    assert b"schema_path" not in response.data
    # 나머지 기능(추천 질문 재생성)은 계속 동작해야 함
    ask_response = client.post("/ask", follow_redirects=True)
    assert ask_response.status_code == 200


def test_end_interview_is_idempotent(client, monkeypatch):
    def _really_start_finalize_worker(app_state):
        threading.Thread(target=web._finalize_interview, args=(app_state,), daemon=True).start()

    def _really_start_final_question_worker(app_state):
        threading.Thread(target=web._refresh_candidates, args=(app_state,), daemon=True).start()

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    calls = []

    def _spy_generate(*a, **k):
        calls.append(1)
        return QuestionCandidates(generated_at=f"t{len(calls):03d}", candidates=[])

    monkeypatch.setattr(web, "generate_questions", _spy_generate)
    monkeypatch.setattr(web, "_start_finalize_worker", _really_start_finalize_worker)
    monkeypatch.setattr(web, "_start_final_question_worker", _really_start_final_question_worker)
    _create_session(client)

    client.post("/end", follow_redirects=True)
    client.post("/end", follow_redirects=True)
    time.sleep(0.05)  # 백그라운드 스레드가 돌 시간을 준다

    assert len(calls) == 1  # 두 번째 /end는 아무 일도 안 함
    assert web._state.ended is True


def test_end_interview_with_no_active_session_redirects_without_error(client):
    response = client.post("/end", follow_redirects=True)

    assert response.status_code == 200
    assert web._state is None


def test_end_interview_stops_live_mic_source(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "DeepgramLiveMicSTTSource", _FakeDeepgramLiveMicSTTSource)
    monkeypatch.setattr(web, "_start_ingestion_worker", lambda app_state: None)
    monkeypatch.setattr(web, "generate_questions", lambda *a, **k: QuestionCandidates(generated_at="", candidates=[]))
    _create_session(client, input_mode="live_mic")
    mic_source = web._state.stt_source

    client.post("/end", follow_redirects=True)

    assert mic_source.stopped is True
    assert web._state is not None  # live_mic여도 세션은 유지됨


def test_next_turn_ignored_after_interview_ended(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "generate_questions", lambda *a, **k: QuestionCandidates(generated_at="", candidates=[]))
    _create_session(client)
    client.post("/end", follow_redirects=True)

    client.post("/next", follow_redirects=True)

    assert web._state.transcript_log == []


def test_end_interview_writes_final_document_in_background(client, monkeypatch):
    from interview_assistant.contracts import FinalDocument

    def _really_start_finalize_worker(app_state):
        threading.Thread(target=web._finalize_interview, args=(app_state,), daemon=True).start()

    def _really_start_summary_worker(app_state):
        threading.Thread(target=web._write_final_document, args=(app_state,), daemon=True).start()

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "generate_questions", lambda *a, **k: QuestionCandidates(generated_at="", candidates=[]))
    monkeypatch.setattr(web, "_start_finalize_worker", _really_start_finalize_worker)
    monkeypatch.setattr(web, "_start_summary_worker", _really_start_summary_worker)
    monkeypatch.setattr(
        web,
        "write_final_document",
        lambda *a, **k: FinalDocument(
            overview="종합 개요",
            section_summaries={"item_01": "섹션 요약"},
        ),
    )
    _create_session(client)

    client.post("/end", follow_redirects=True)
    time.sleep(0.05)  # 백그라운드 스레드가 돌 시간을 준다

    state = web._state.structurer.state
    assert state.overview == "종합 개요"
    assert state.schema_items[0].summary == "섹션 요약"
    assert web._state.knowledge_exporter.exported[-1] is state
    assert web._state.summarizing is False  # 백그라운드 작성이 끝나면 표시가 꺼짐


def test_end_interview_sets_summarizing_true_before_background_completes(client, monkeypatch):
    # client 픽스처의 기본 no-op _start_finalize_worker를 그대로 둬서(백그라운드가
    # 절대 안 도는 상태), summarizing=True가 end_interview()에서 그 워커를 띄우기도
    # 전에 세워지는지(=/end 직후에도 True로 남아있는지) 확인한다.
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "generate_questions", lambda *a, **k: QuestionCandidates(generated_at="", candidates=[]))
    _create_session(client)

    client.post("/end", follow_redirects=True)

    assert web._state.summarizing is True


def test_write_final_document_resets_summarizing_even_when_it_raises(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "generate_questions", lambda *a, **k: QuestionCandidates(generated_at="", candidates=[]))
    monkeypatch.setattr(web, "write_final_document", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API 실패")))
    _create_session(client)
    web._state.summarizing = True

    with pytest.raises(RuntimeError):
        web._write_final_document(web._state)

    assert web._state.summarizing is False  # 실패해도 표시는 꺼져야 함(finally)


def test_status_endpoint_includes_summarizing_field(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.summarizing = True

    data = client.get("/status").get_json()

    assert data["summarizing"] is True


def test_status_endpoint_candidates_null_before_any_generation(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)

    data = client.get("/status").get_json()

    assert data["candidates"] is None


def test_status_endpoint_serializes_latest_candidates(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.latest_candidates = QuestionCandidates(
        generated_at="t003",
        candidates=[
            QuestionCandidate(
                type=QuestionType.CONTRADICTION, text="모순 확인 질문", target_item="item_01"
            )
        ],
    )

    data = client.get("/status").get_json()

    assert data["candidates"]["generated_at"] == "t003"
    assert data["candidates"]["candidates"] == [
        {"type": "contradiction", "text": "모순 확인 질문", "target_item": "item_01"}
    ]


def test_end_interview_summary_worker_does_nothing_when_no_content(client, monkeypatch):
    from interview_assistant.contracts import FinalDocument

    def _really_start_finalize_worker(app_state):
        threading.Thread(target=web._finalize_interview, args=(app_state,), daemon=True).start()

    def _really_start_summary_worker(app_state):
        threading.Thread(target=web._write_final_document, args=(app_state,), daemon=True).start()

    calls = []

    def _spy_write_final_document(*a, **k):
        calls.append(1)
        return FinalDocument()

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "generate_questions", lambda *a, **k: QuestionCandidates(generated_at="", candidates=[]))
    monkeypatch.setattr(web, "_start_finalize_worker", _really_start_finalize_worker)
    monkeypatch.setattr(web, "_start_summary_worker", _really_start_summary_worker)
    monkeypatch.setattr(web, "write_final_document", _spy_write_final_document)
    _create_session(client)  # 캡처도 미분류도 전혀 없는 상태

    client.post("/end", follow_redirects=True)
    time.sleep(0.05)

    assert len(calls) == 1  # 호출은 되지만(스킵 가드는 final_document.py 내부 몫)
    state = web._state.structurer.state
    assert state.overview == ""
    assert state.schema_items[0].summary == ""


def test_ingestion_loop_processes_events_and_marks_exhausted(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    app_state = web._state
    app_state.events = iter(
        [
            TranscriptEvent(text="첫 턴", timestamp=0.0, turn_id="t001"),
            TranscriptEvent(text="둘째 턴", timestamp=1.0, turn_id="t002"),
        ]
    )

    web._ingestion_loop(app_state)

    assert [e.text for e in app_state.transcript_log] == ["첫 턴", "둘째 턴"]
    assert app_state.exhausted is True
    # _ingestion_loop은 이제 원문만 담는다 — 구조화는 _structuring_loop가 별도로 담당.
    assert app_state.structurer.ingested == []


def test_adopt_question_before_session_started_redirects_safely(client):
    response = client.post("/adopt", data={"action": "adopted"}, follow_redirects=True)

    assert response.status_code == 200
    assert b"schema_path" in response.data
    assert web._state is None


def test_session_view_renders_card_form_with_prefilled_text(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.latest_candidates = QuestionCandidates(
        generated_at="t001",
        candidates=[QuestionCandidate(type=QuestionType.PROBE, text="편집 가능한 질문")],
    )

    response = client.get("/")

    assert "편집 가능한 질문".encode() in response.data
    assert b'name="original_text"' in response.data
    assert b'name="question_type"' in response.data
    assert b'name="target_item"' in response.data


def test_status_endpoint_returns_active_false_when_no_session(client):
    response = client.get("/status")

    assert response.status_code == 200
    assert response.get_json() == {"active": False}


def test_status_endpoint_reflects_transcript_and_coverage_after_next_turn(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    client.post("/next", follow_redirects=True)
    web._state.structurer.state.schema_items[0].status = CoverageStatus.COVERED

    data = client.get("/status").get_json()

    assert data["active"] is True
    assert data["exhausted"] is False
    assert [t["turn_id"] for t in data["transcript"]] == ["t001"]
    assert data["transcript"][0]["text"] == "로스팅을 언제 멈춰야 할지는 어떻게 판단하세요?"
    assert data["gap_count"] == 3
    assert data["structured_state"]["schema_items"][0]["status"] == "covered"


def test_index_does_not_auto_select_schema_or_audio(client, tmp_path, monkeypatch):
    """파일이 있어도 F0을 처음 열 땐 아무것도 미리 골라놓지 않는다(2026-08-07,
    "세션 이름은 비워두라고 했다"는 피드백으로 자동 로드를 되돌림) — 도메인은
    빈 문자열, 스키마/오디오 드롭다운 둘 다 "선택 안 함"이 selected여야 한다."""
    import re

    from interview_assistant.contracts import InterviewSchema, SchemaItemDef
    from interview_assistant.schema_loader import save_schema

    schemas_dir = tmp_path / "schemas"
    save_schema(
        schemas_dir / "a_first.json",
        InterviewSchema(domain="첫 번째 스키마", items=[SchemaItemDef(item_id="item_01", label="첫 항목")]),
    )
    monkeypatch.setattr(web, "SCHEMAS_DIR", schemas_dir)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_file = audio_dir / "a_first.wav"
    audio_file.write_bytes(b"fake wav")
    monkeypatch.setattr(web, "AUDIO_DIR", audio_dir)

    response = client.get("/")
    html = response.data.decode()

    assert "첫 항목".encode() not in response.data
    assert b'name="domain" value="">' in response.data

    schema_option = re.search(rf'<option value="{re.escape(str(schemas_dir / "a_first.json"))}"[^>]*>', html)
    audio_option = re.search(rf'<option value="{re.escape(str(audio_file))}"[^>]*>', html)
    assert "selected" not in schema_option.group()
    assert "selected" not in audio_option.group()
    assert len(re.findall(r'<option value=""\s*selected>선택 안 함</option>', html)) == 2


def test_index_lists_multiple_schema_files_in_dropdown(client, tmp_path, monkeypatch):
    from interview_assistant.contracts import InterviewSchema
    from interview_assistant.schema_loader import save_schema

    schemas_dir = tmp_path / "schemas"
    save_schema(schemas_dir / "a_first.json", InterviewSchema(domain="A", items=[]))
    save_schema(schemas_dir / "z_second.json", InterviewSchema(domain="Z", items=[]))
    monkeypatch.setattr(web, "SCHEMAS_DIR", schemas_dir)

    response = client.get("/")

    assert b"a_first.json" in response.data
    assert b"z_second.json" in response.data


def test_index_shows_blank_when_schemas_directory_has_no_files(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web, "SCHEMAS_DIR", tmp_path / "empty_schemas")

    response = client.get("/")

    assert b'name="domain" value="">' in response.data
    assert "스키마 파일이 없습니다".encode() in response.data


def test_index_lists_multiple_audio_files_in_dropdown(client, tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a_first.wav").write_bytes(b"fake wav")
    (audio_dir / "z_second.mp3").write_bytes(b"fake mp3")
    monkeypatch.setattr(web, "AUDIO_DIR", audio_dir)

    response = client.get("/")

    assert b"a_first.wav" in response.data
    assert b"z_second.mp3" in response.data


def test_index_shows_placeholder_when_audio_directory_has_no_files(client, tmp_path, monkeypatch):
    monkeypatch.setattr(web, "AUDIO_DIR", tmp_path / "empty_audio")

    response = client.get("/")

    assert "오디오 파일이 없습니다".encode() in response.data


def test_load_schema_route_populates_grid_from_existing_file(client, tmp_path):
    from interview_assistant.contracts import InterviewSchema, SchemaItemDef
    from interview_assistant.schema_loader import save_schema

    path = tmp_path / "my_schema.json"
    save_schema(
        path,
        InterviewSchema(
            domain="테스트 도메인",
            items=[SchemaItemDef(item_id="item_01", label="첫 항목", criteria="기준1")],
        ),
    )

    response = client.post("/schema/load", data={"schema_path": str(path)})

    assert "테스트 도메인".encode() in response.data
    assert "첫 항목".encode() in response.data
    assert "기준1".encode() in response.data


def test_load_schema_route_with_missing_file_shows_info_and_blank_grid(client):
    response = client.post("/schema/load", data={"schema_path": "존재하지_않는_스키마.json"})

    assert response.status_code == 200
    assert "경로에 파일이 없습니다".encode() in response.data


def test_save_schema_route_writes_file_and_shows_confirmation(client, tmp_path):
    from interview_assistant.schema_loader import load_schema

    path = tmp_path / "new_schema.json"

    response = client.post(
        "/schema/save",
        data={
            "schema_path": str(path),
            "domain": "새 도메인",
            "item_label[]": ["항목 A", "항목 B"],
            "item_criteria[]": ["기준 A", "기준 B"],
        },
    )

    assert response.status_code == 200
    assert "저장했습니다".encode() in response.data
    saved = load_schema(path)
    assert saved.domain == "새 도메인"
    assert [i.label for i in saved.items] == ["항목 A", "항목 B"]
    assert [i.item_id for i in saved.items] == ["item_01", "item_02"]


def test_save_schema_route_requires_domain_and_path(client, tmp_path):
    response = client.post(
        "/schema/save",
        data={"schema_path": "", "domain": "", "item_label[]": [], "item_criteria[]": []},
    )

    assert response.status_code == 400
    assert "경로와 도메인이 모두 필요합니다".encode() in response.data


def test_save_schema_route_drops_blank_item_rows(client, tmp_path):
    from interview_assistant.schema_loader import load_schema

    path = tmp_path / "sparse_schema.json"

    client.post(
        "/schema/save",
        data={
            "schema_path": str(path),
            "domain": "도메인",
            "item_label[]": ["실제 항목", "   ", ""],
            "item_criteria[]": ["기준", "", ""],
        },
    )

    saved = load_schema(path)
    assert len(saved.items) == 1
    assert saved.items[0].label == "실제 항목"


def _fail_if_called(*args, **kwargs):
    raise AssertionError("use_schema=no 세션에서는 load_schema가 호출되면 안 된다")


def test_create_session_with_use_schema_no_builds_empty_item_schema(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "load_schema", _fail_if_called)

    _create_session(client, use_schema="no", domain="자유 인터뷰")

    assert web._state.schema.items == []
    assert web._state.schema.domain == "자유 인터뷰"


def test_create_session_with_use_schema_no_and_blank_domain_uses_default_label(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "load_schema", _fail_if_called)

    _create_session(client, use_schema="no", domain="")

    assert web._state.schema.domain == "자유 형식 인터뷰"


def test_create_session_default_use_schema_preserves_existing_load_schema_behavior(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)

    _create_session(client)

    assert len(web._state.schema.items) == 4  # DEFAULT_SCHEMA_PATH 실제 파일 그대로 로드됨


def test_session_view_shows_empty_state_when_no_schema_items_yet(client, monkeypatch):
    # 스키마가 없어도(혹은 아직 동적 섹션이 하나도 안 생겼어도) "정리" 패널 자체는
    # 3-컬럼 레이아웃의 고정 영역이라 계속 보인다 — "미분류" 폴백이 없어진 뒤로는
    # 빈 상태 문구가 그 자리를 대신한다(2026-08-10).
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client, use_schema="no", domain="자유 인터뷰")

    response = client.get("/")

    assert b'id="structure-panel"' in response.data
    assert "스키마 없음".encode() in response.data
    assert "아직 정리된 내용이 없습니다".encode() in response.data


def test_session_view_shows_structure_panel_with_items_when_schema_present(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)

    response = client.get("/")

    assert b'id="structure-panel"' in response.data
    assert "로스팅 종료 시점 판단".encode() in response.data  # 실제 스키마 항목 라벨(fixture)
    assert "스키마 없음".encode() not in response.data


def test_session_view_initial_render_shows_dynamically_created_sections_for_no_schema_session(
    client, monkeypatch
):
    # 회귀 테스트: session.html이 최초 렌더링 시 (불변인) state.schema.items가 아니라
    # 살아있는 state.structurer.state.schema_items를 기준으로 항목 목록을 보여줘야 한다.
    # 스키마 없는 세션은 state.schema.items가 영원히 []라서, 이 가드가 잘못돼 있으면
    # 동적으로 생긴 섹션이 풀 페이지 로드 시 전혀 안 보인다(폴링 전까지는).
    from interview_assistant.contracts import SchemaItem

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client, use_schema="no", domain="자유 인터뷰")
    web._state.structurer.state.schema_items.append(
        SchemaItem(item_id="dyn_001", label="동적으로 생긴 섹션")
    )

    response = client.get("/")

    assert "동적으로 생긴 섹션".encode() in response.data


def test_run_structuring_pass_calls_ingest_batch_once_with_all_pending(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    app_state = web._state
    app_state.transcript_log.extend(
        [
            TranscriptEvent(text="첫 턴", timestamp=0.0, turn_id="t001"),
            TranscriptEvent(text="둘째 턴", timestamp=1.0, turn_id="t002"),
        ]
    )

    result = web._run_structuring_pass(app_state)

    assert result is True
    assert app_state.structured_count == 2
    assert len(app_state.structurer.ingested) == 2


def test_run_structuring_pass_serializes_concurrent_calls(client, monkeypatch):
    # (2026-08-11) structuring_lock 회귀 테스트 — 배경 루프의 호출이 아직 진행 중일
    # 때(ingest_batch가 느리게 응답하는 상황을 흉내냄) /end처럼 두 번째 호출이 같은
    # app_state에 대해 들어오면, 락이 풀릴 때까지 기다렸다가 그 시점엔 이미 pending이
    # 비어 있어 ingest_batch를 또 호출하지 않아야 한다(중복 LLM 호출 방지).
    class _SlowFakeStructurer(_FakeStructurer):
        def ingest_batch(self, events):
            time.sleep(0.1)
            super().ingest_batch(events)

    monkeypatch.setattr(web, "Structurer", _SlowFakeStructurer)
    _create_session(client)
    app_state = web._state
    app_state.transcript_log.append(TranscriptEvent(text="첫 턴", timestamp=0.0, turn_id="t001"))

    results = []
    first_call = threading.Thread(
        target=lambda: results.append(web._run_structuring_pass(app_state))
    )
    first_call.start()
    time.sleep(0.03)  # 첫 호출이 락을 잡고 ingest_batch의 sleep에 들어갈 시간을 준다

    second_result = web._run_structuring_pass(app_state)  # 락이 풀릴 때까지 여기서 블록
    first_call.join()

    assert results == [True]  # 첫 호출만 실제로 처리함
    assert second_result is False  # 두 번째는 그 사이 pending이 사라져 즉시 스킵
    assert len(app_state.structurer.ingested) == 1  # ingest_batch가 딱 한 번만 불림


def test_run_structuring_pass_returns_false_when_nothing_pending(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)

    assert web._run_structuring_pass(web._state) is False


def test_run_structuring_pass_exports_knowledge_state(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.transcript_log.append(TranscriptEvent(text="발화", timestamp=0.0, turn_id="t001"))

    web._run_structuring_pass(web._state)

    assert len(web._state.knowledge_exporter.exported) == 1


def test_apply_transcript_corrections_updates_matching_turn_text(client, monkeypatch):
    from interview_assistant.contracts import TranscriptCorrection

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.transcript_log.append(TranscriptEvent(text="저 자에 로스팅", timestamp=0.0, turn_id="t001"))
    monkeypatch.setattr(
        web,
        "review_transcript",
        lambda *a, **k: [TranscriptCorrection(turn_id="t001", corrected_text="저는 로스팅", confidence=0.95)],
    )

    web._apply_transcript_corrections(web._state)

    assert web._state.transcript_log[0].text == "저는 로스팅"
    assert web._state.transcript_log[0].corrected is True
    assert web._state.corrected_turn_ids == {"t001"}
    assert len(web._state.transcript_writer.corrected_snapshots) == 1


def test_apply_transcript_corrections_ignores_below_confidence_threshold(client, monkeypatch):
    from interview_assistant.contracts import TranscriptCorrection

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.transcript_log.append(TranscriptEvent(text="원문 그대로", timestamp=0.0, turn_id="t001"))
    monkeypatch.setattr(
        web,
        "review_transcript",
        lambda *a, **k: [TranscriptCorrection(turn_id="t001", corrected_text="바뀐 문장", confidence=0.5)],
    )

    web._apply_transcript_corrections(web._state)

    assert web._state.transcript_log[0].text == "원문 그대로"
    assert web._state.transcript_log[0].corrected is False


def test_apply_transcript_corrections_ignores_unknown_turn_id(client, monkeypatch):
    from interview_assistant.contracts import TranscriptCorrection

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.transcript_log.append(TranscriptEvent(text="원문", timestamp=0.0, turn_id="t001"))
    monkeypatch.setattr(
        web,
        "review_transcript",
        lambda *a, **k: [TranscriptCorrection(turn_id="t999_no_such_turn", corrected_text="아무거나", confidence=0.95)],
    )

    web._apply_transcript_corrections(web._state)  # 예외 없이 조용히 무시돼야 함

    assert web._state.transcript_log[0].text == "원문"


def test_apply_transcript_corrections_does_not_reapply_to_already_corrected_turn(client, monkeypatch):
    from interview_assistant.contracts import TranscriptCorrection

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.transcript_log.append(TranscriptEvent(text="원문", timestamp=0.0, turn_id="t001"))
    web._state.corrected_turn_ids.add("t001")  # 이미 한 번 교정된 걸로 표시
    monkeypatch.setattr(
        web,
        "review_transcript",
        lambda *a, **k: [TranscriptCorrection(turn_id="t001", corrected_text="또 바뀐 문장", confidence=0.99)],
    )

    web._apply_transcript_corrections(web._state)

    assert web._state.transcript_log[0].text == "원문"  # 재교정 안 됨(플립플롭 방지)


def test_structuring_loop_processes_pending_turns_on_configured_interval(client, monkeypatch):
    def _really_start_structuring_worker(app_state):
        threading.Thread(target=web._structuring_loop, args=(app_state,), daemon=True).start()

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    # client 픽스처의 기본 no-op을 되돌려 실제 데몬 스레드가 뜨게 한다.
    monkeypatch.setattr(web, "_start_structuring_worker", _really_start_structuring_worker)

    _create_session(client, structuring_interval_seconds="0.01")
    web._state.transcript_log.append(TranscriptEvent(text="발화", timestamp=0.0, turn_id="t001"))

    time.sleep(0.05)  # 백그라운드 스레드가 한 바퀴 돌 시간을 준다

    assert len(web._state.structurer.ingested) == 1


def test_question_loop_regenerates_when_judge_triggers_and_passes_hint(client, monkeypatch):
    from interview_assistant.contracts import TriggerDecision

    def _really_start_question_worker(app_state):
        threading.Thread(target=web._question_loop, args=(app_state,), daemon=True).start()

    generate_calls = []

    def _spy_generate(*a, **k):
        generate_calls.append(k)
        return QuestionCandidates(generated_at=f"t{len(generate_calls):03d}", candidates=[])

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "generate_questions", _spy_generate)
    monkeypatch.setattr(
        web, "judge_trigger", lambda *a, **k: TriggerDecision(should_trigger=True, reason="테스트 이유")
    )
    monkeypatch.setattr(web, "QUESTION_GENERATION_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(web, "_start_question_worker", _really_start_question_worker)

    _create_session(client)
    web._state.transcript_log.append(TranscriptEvent(text="발화", timestamp=0.0, turn_id="t001"))

    time.sleep(0.05)  # 백그라운드 스레드가 한 바퀴 돌 시간을 준다

    assert len(generate_calls) == 1
    assert generate_calls[0]["trigger_hint"] == "테스트 이유"
    assert web._state.last_question_turn_count == 1


def test_question_loop_skips_regeneration_when_no_new_turns(client, monkeypatch):
    def _really_start_question_worker(app_state):
        threading.Thread(target=web._question_loop, args=(app_state,), daemon=True).start()

    calls = []

    def _spy_generate(*a, **k):
        calls.append(1)
        return QuestionCandidates(generated_at=f"t{len(calls):03d}", candidates=[])

    judge_calls = []

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "generate_questions", _spy_generate)
    monkeypatch.setattr(web, "judge_trigger", lambda *a, **k: judge_calls.append(1))
    monkeypatch.setattr(web, "QUESTION_GENERATION_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(web, "_start_question_worker", _really_start_question_worker)

    _create_session(client)  # transcript_log가 빈 채로 시작 — 새 턴이 전혀 없음

    time.sleep(0.05)  # 최소 한 바퀴는 돌지만 새 턴이 없으니 스킵돼야 함

    assert calls == []
    assert judge_calls == []  # 새 턴 자체가 없으면 판단 호출까지 갈 필요도 없음(1차 게이트)


def test_question_loop_skips_generation_when_judge_says_not_yet(client, monkeypatch):
    from interview_assistant.contracts import TriggerDecision

    def _really_start_question_worker(app_state):
        threading.Thread(target=web._question_loop, args=(app_state,), daemon=True).start()

    calls = []

    def _spy_generate(*a, **k):
        calls.append(1)
        return QuestionCandidates(generated_at=f"t{len(calls):03d}", candidates=[])

    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    monkeypatch.setattr(web, "generate_questions", _spy_generate)
    monkeypatch.setattr(
        web, "judge_trigger", lambda *a, **k: TriggerDecision(should_trigger=False, reason="")
    )
    monkeypatch.setattr(web, "QUESTION_GENERATION_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(web, "_start_question_worker", _really_start_question_worker)

    _create_session(client)
    web._state.transcript_log.append(TranscriptEvent(text="발화", timestamp=0.0, turn_id="t001"))

    time.sleep(0.05)  # 판단 호출은 되지만 새 턴이 있어도 생성은 스킵돼야 함

    assert calls == []
    assert web._state.last_question_turn_count == 0  # 생성 안 됐으니 커서도 안 움직임


def test_create_session_reads_structuring_interval_from_form(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)

    _create_session(client, structuring_interval_seconds="10")

    assert web._state.structuring_interval_seconds == 10.0


def test_create_session_defaults_structuring_interval_to_five(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)

    _create_session(client)

    assert web._state.structuring_interval_seconds == 5.0


def test_create_session_starts_structuring_worker_unconditionally(client, monkeypatch):
    # fixture 모드(auto_ingest=False)에서도 구조화 워커는 항상 시작돼야 한다 —
    # "/next"가 더 이상 구조화를 안 하므로 이 루프가 유일한 구조화 경로다.
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    captured = {}

    def _spy_start_structuring_worker(app_state):
        captured["app_state"] = app_state

    monkeypatch.setattr(web, "_start_structuring_worker", _spy_start_structuring_worker)

    _create_session(client)

    assert captured["app_state"] is web._state


def test_status_endpoint_includes_structured_state_with_summary_and_source_refs(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.structurer.state.schema_items[0].summary = "향으로 판단"
    web._state.structurer.state.schema_items[0].source_refs.append("t002")

    data = client.get("/status").get_json()

    item = data["structured_state"]["schema_items"][0]
    assert item["summary"] == "향으로 판단"
    assert item["source_refs"] == ["t002"]


def test_render_markdown_returns_empty_string_for_empty_input():
    assert web._render_markdown("") == ""


def test_render_markdown_converts_headings_and_lists_to_html():
    html = web._render_markdown("# 제목\n\n- 첫째\n- 둘째")

    assert "<h1>" in html
    assert "<li>첫째</li>" in html
    assert "<li>둘째</li>" in html


def test_render_markdown_converts_table_syntax_to_html_table():
    html = web._render_markdown("| 기준 | 예외 |\n|---|---|\n| 향 | 시간 |")

    assert "<table>" in html
    assert "<th>기준</th>" in html
    assert "<td>향</td>" in html


def test_status_endpoint_includes_prerendered_html_for_final_document_fields(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    state = web._state.structurer.state
    state.overview = "# 개요"
    state.schema_items[0].summary = "## 섹션 요약"

    data = client.get("/status").get_json()
    s = data["structured_state"]

    assert "<h1>개요</h1>" in s["overview_html"]
    assert "<h2>섹션 요약</h2>" in s["schema_items"][0]["summary_html"]


def test_status_endpoint_transcript_items_include_corrected_field(client, monkeypatch):
    monkeypatch.setattr(web, "Structurer", _FakeStructurer)
    _create_session(client)
    web._state.transcript_log.append(TranscriptEvent(text="원문", timestamp=0.0, turn_id="t001"))
    web._state.transcript_log[0].corrected = True

    data = client.get("/status").get_json()

    assert data["transcript"][0]["corrected"] is True


def test_session_detail_route_renders_saved_session(client):
    # 이 라우트의 목적 자체가 실제 파일 포맷 왕복 확인이라, client 픽스처의 가짜 writer가
    # 아니라 진짜 storage 클래스로 세션 디렉터리를 채운다(web.SESSIONS_DIR는 픽스처가
    # 이미 tmp_path로 돌려둠).
    from interview_assistant.contracts import (
        AdoptionAction,
        AdoptionEvent,
        KnowledgeState,
        QuestionType,
        SchemaItem,
    )
    from interview_assistant.storage.adoption_log import AdoptionLogWriter as RealAdoptionLogWriter
    from interview_assistant.storage.knowledge_export import KnowledgeStateExporter as RealExporter
    from interview_assistant.storage.session_log import SessionLogWriter as RealSessionLogWriter
    from interview_assistant.storage.session_meta import SessionMeta
    from interview_assistant.storage.session_meta import write_session_meta as real_write_session_meta

    session_dir = web.SESSIONS_DIR / "web_20260101_000000"
    real_write_session_meta(
        session_dir,
        SessionMeta(
            session_id="web_20260101_000000",
            domain="테스트 도메인",
            has_schema=True,
            input_mode="fixture",
            interview_goal="목적",
            started_at="2026-01-01T00:00:00",
        ),
    )
    RealSessionLogWriter(session_dir / "transcript.jsonl").append(
        TranscriptEvent(text="테스트 발화", timestamp=0.0, turn_id="t001")
    )
    RealExporter(session_dir, "테스트 도메인").export(
        KnowledgeState(
            session_id="web_20260101_000000",
            schema_items=[SchemaItem(item_id="item_01", label="항목")],
        )
    )
    RealAdoptionLogWriter(session_dir / "adoptions.jsonl").append(
        AdoptionEvent(
            generated_at="t001",
            question_type=QuestionType.PROBE,
            question_text="채택된 질문",
            action=AdoptionAction.ADOPTED,
        )
    )

    response = client.get("/sessions/web_20260101_000000")

    assert response.status_code == 200
    assert "테스트 도메인".encode() in response.data
    assert "테스트 발화".encode() in response.data
    assert "항목".encode() in response.data
    assert "채택된 질문".encode() in response.data


def test_session_detail_route_prefers_corrected_transcript_snapshot(client):
    # transcript_corrected.json이 있으면 원본 transcript.jsonl 대신 그걸 읽어야 한다.
    from interview_assistant.storage.session_log import SessionLogWriter as RealSessionLogWriter
    from interview_assistant.storage.session_meta import SessionMeta
    from interview_assistant.storage.session_meta import write_session_meta as real_write_session_meta

    session_dir = web.SESSIONS_DIR / "web_20260101_000000"
    real_write_session_meta(
        session_dir,
        SessionMeta(
            session_id="web_20260101_000000",
            domain="테스트 도메인",
            has_schema=True,
            input_mode="fixture",
            interview_goal="목적",
            started_at="2026-01-01T00:00:00",
        ),
    )
    writer = RealSessionLogWriter(session_dir / "transcript.jsonl")
    writer.append(TranscriptEvent(text="원본 오인식 텍스트", timestamp=0.0, turn_id="t001"))
    writer.export_corrected_snapshot(
        [TranscriptEvent(text="교정된 텍스트", timestamp=0.0, turn_id="t001", corrected=True)]
    )

    response = client.get("/sessions/web_20260101_000000")

    assert "교정된 텍스트".encode() in response.data
    assert "원본 오인식 텍스트".encode() not in response.data
    assert "corrected-badge".encode() in response.data  # corrected 플래그가 스냅샷에서 round-trip돼야 함


def test_session_detail_404_for_unknown_session_id(client):
    response = client.get("/sessions/does-not-exist")

    assert response.status_code == 404


def test_index_shows_session_list_with_links_to_detail(client):
    from interview_assistant.storage.session_meta import SessionMeta
    from interview_assistant.storage.session_meta import write_session_meta as real_write_session_meta

    session_dir = web.SESSIONS_DIR / "web_20260101_000000"
    real_write_session_meta(
        session_dir,
        SessionMeta(
            session_id="web_20260101_000000",
            domain="지난 세션 도메인",
            has_schema=True,
            input_mode="fixture",
            interview_goal="",
            started_at="2026-01-01T00:00:00",
        ),
    )

    response = client.get("/")

    assert "지난 세션 도메인".encode() in response.data
    assert b"/sessions/web_20260101_000000" in response.data


def test_index_shows_empty_session_list_message_when_none_exist(client):
    response = client.get("/")

    assert "아직 지난 세션이 없습니다".encode() in response.data


def test_delete_session_route_removes_directory_and_redirects(client):
    from interview_assistant.storage.session_meta import SessionMeta
    from interview_assistant.storage.session_meta import write_session_meta as real_write_session_meta

    session_dir = web.SESSIONS_DIR / "web_20260101_000000"
    real_write_session_meta(
        session_dir,
        SessionMeta(
            session_id="web_20260101_000000",
            domain="지울 세션",
            has_schema=True,
            input_mode="fixture",
            interview_goal="",
            started_at="2026-01-01T00:00:00",
        ),
    )

    response = client.post("/sessions/web_20260101_000000/delete")

    assert response.status_code == 302
    assert not session_dir.exists()
    assert "지울 세션".encode() not in client.get("/").data


def test_delete_session_route_for_unknown_id_still_redirects(client):
    response = client.post("/sessions/없는_세션/delete")

    assert response.status_code == 302
