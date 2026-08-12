"""LLM 호출 관련 설정 한 곳에 모음 — 프롬프트를 만드는 여러 모듈(structure_synthesizer.py,
outline_seeder.py, question_engine.py 등)에 흩어져 있던 MODEL 상수(전부 동일 값 중복)와
튜닝 파라미터를 정리한다.

모델명만 .env로 오버라이드 가능하게 한다(API 키를 .env로 관리하는 기존 방식과
일관) — 모델을 실험적으로 바꿔볼 때 코드를 안 고쳐도 되게. os.environ.get은 호출
시점에 평가해야 한다(모듈 import 시점에 한 번만 읽어 상수로 굳혀버리면, 이미 이
프로젝트가 겪은 "Python 기본 인자 late-binding" 함정과 같은 종류의 문제 — 테스트에서
monkeypatch.setenv해도 이미 import된 모듈엔 반영 안 됨). 그래서 함수로 둔다.
나머지 값들은 굳이 환경변수로 뺄 필요 없는 개발 중 튜닝값이라 평범한 상수로 둔다.

(2026-08-11 추가) 모델을 용도별로 분리했다 — "지식 구조(KnowledgeState)를 만들거나
통째로 다시 쓰는 일"(실시간 구조화·최종 정리·초기 섹션 제안)은 get_structuring_model()로
고성능 모델을, "빠르게 판단만 하거나 짧게 고쳐 쓰는 일"(트리거 판단·질문 생성·전사교정)은
기존 get_model()의 경량 모델을 그대로 쓴다.
"""

from __future__ import annotations

import os

_DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
_STRUCTURING_MODEL = "anthropic/claude-sonnet-5"


def get_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", _DEFAULT_MODEL)


# structurer/outline_seeder.py, structurer/structure_synthesizer.py,
# structurer/final_document.py 전용(2026-08-11) — 지식 구조를 실제로 만들거나
# 통째로 다시 쓰는 호출이라, 판단만 하는 나머지 호출(get_model()의 경량 모델)보다
# 고성능 모델을 쓴다. get_model()과 마찬가지로 함수형 — 이유도 동일(호출 시점 평가).
def get_structuring_model() -> str:
    return os.environ.get("OPENROUTER_STRUCTURING_MODEL", _STRUCTURING_MODEL)


# question_engine.py — 질문 생성에 넣을 최근 전사 시간창(초).
RECENT_WINDOW_SECONDS = 60

# question_engine.py — 질문 생성 호출의 max_tokens.
QUESTION_MAX_TOKENS = 2048

# question_engine.py — 한 번에 반환하는 질문 후보 상한.
MAX_QUESTION_CANDIDATES = 4

# structurer/structurer.py — 세션당 동적으로 생성 가능한 섹션 개수 상한(폭주 방지).
MAX_DYNAMIC_ITEMS = 40

# structurer/outline_seeder.py — 세션 시작 시 미리 제안하는 초기 섹션 개수 상한.
INITIAL_OUTLINE_MAX_SECTIONS = 5

# structurer/structurer.py — 섹션 하나당 제목을 다시 지을 수 있는 최대 횟수(폭주/
# 플립플롭 방지). MAX_DYNAMIC_ITEMS와 같은 취지.
MAX_RENAMES_PER_ITEM = 2

# structurer/transcript_reviewer.py — 전사 오인식 재검토에 넣을 최근 전사 시간창(초).
# 질문 생성용 RECENT_WINDOW_SECONDS(60)보다 길게 잡는다 — "뒤에 나온 내용으로 앞을
# 고친다"는 특성상 교정 대상 턴과 그 근거가 되는 턴이 같은 창 안에 있어야 한다.
TRANSCRIPT_REVIEW_WINDOW_SECONDS = 180

# structurer/transcript_reviewer.py — 이 확신도 미만인 교정 제안은 코드에서 버린다
# (프롬프트 지시와 별개로 수치로 강제 — 잘못된 교정이 조용히 적용되는 걸 막는 마지막 방어선).
TRANSCRIPT_CORRECTION_MIN_CONFIDENCE = 0.85

# app/web.py — 추천 질문 백그라운드 보완 생성 주기(초). 버튼(즉시 생성)과는 별개로,
# 새 턴이 쌓였을 때만 이 주기로 한 번씩 미리 생성해둔다("알아서 추천"도 필요하다는
# 요청 + 고정 주기 상시 생성은 대화 흐름과 무관해 낭비라는 절충안 — 2026-08-04).
QUESTION_GENERATION_INTERVAL_SECONDS = 10

# structurer/final_document.py — 인터뷰 종료 시 1회, 개요+섹션별 요약+미분류 요약을
# 한 번의 호출로 전부 받아야 하는 데다 이제 마크다운 표/헤딩까지 포함할 수 있어
# 응답이 길어질 수 있으므로 QUESTION_MAX_TOKENS보다 넉넉히 잡는다.
FINAL_DOCUMENT_MAX_TOKENS = 4096

# question_engine/trigger_judge.py — "지금 추천 질문을 생성할 때인가" 판단에 보여줄
# 최근 전사 시간창(초). 질문 생성 자체의 RECENT_WINDOW_SECONDS(60)보다 길게 잡는다 —
# "방금 흐려진 곁가지"를 알아채려면 조금 더 넓게 봐야 한다.
TRIGGER_JUDGE_WINDOW_SECONDS = 120

# question_engine/trigger_judge.py — 판단 호출의 max_tokens. 출력이 bool+짧은 이유뿐이라
# QUESTION_MAX_TOKENS보다 훨씬 작게 잡는다.
TRIGGER_JUDGE_MAX_TOKENS = 512

# sources/stt/deepgram_source.py — 오디오 파일 테스트 모드에서 전사가 화면에 나타나는
# 시점을 실제 오디오 재생 위치보다 이만큼 늦춘다(2026-08-06, 사용자 요청). 배치로
# 미리 다 받아온 결과를 실시간처럼 재생하다 보니 사람이 말하자마자 바로 전사가 뜨는
# 게 오히려 부자연스럽다는 판단 — 실제 STT는 처리 지연이 있어서 약간 늦게 뜨는 게
# 자연스럽다.
AUDIO_FILE_TRANSCRIPT_DELAY_SECONDS = 1.0

# app/web.py — 구조화 배치를 만들 때 "이번 틱에 새로 쌓인 턴"뿐 아니라 그
# 앞의 이만큼도 다시 포함시킨다(2026-08-10) — 이전 틱 끝에서 시작된 발화가
# 이번 틱에야 의미가 분명해지는 경계 케이스 대응.
STRUCTURING_LOOKBACK_TURNS = 3
