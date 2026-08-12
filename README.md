# Ainterview Assistant

전문가 인터뷰를 실시간으로 **전사 → 구조화 → 추천 질문**까지 이어주는 개인용 로컬 도구입니다. 인터뷰가 진행되는 동안 배경에서 계속 지식 구조(`KnowledgeState`)를 다시 정리해나가고, 인터뷰어가 원할 때 그 시점 맥락 기반의 후속 질문 후보를 즉석에서 생성해줍니다.

**도메인 무관** — 대상 분야는 코드가 아니라 세션별로 주입하는 인터뷰 스키마(JSON)가 결정합니다. 금융이든 요리든 의료든, 스키마만 갈아끼우면 됩니다(스키마 없이 자유 형식으로도 진행 가능).

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# .env에 OPENROUTER_API_KEY를 채워야 LLM 호출(구조화·질문생성)이 동작합니다.
# 실제 오디오(오디오 파일 테스트/실시간 마이크)를 쓰려면 DEEPGRAM_API_KEY도 필요합니다.
# 텍스트만으로 파이프라인을 확인하려면(input_mode=fixture) 둘 다 없어도 대부분 동작합니다.

python scripts/run_app.py
# http://127.0.0.1:5001 접속
```

테스트는 `fixtures/`의 예시 스키마·가짜 전사로 돕니다(실제 인터뷰나 API 키 없이도 대부분 실행 가능):

```bash
pytest
```

## 폴더 구조

```
src/interview_assistant/
├── contracts.py       # 모든 모듈이 공유하는 데이터 계약 — 모듈끼리 서로의 내부를 몰라도 이걸로 통신
├── config.py           # 모델 선택 · 시간창 · 임계값 등 튜닝 상수
├── sources/             # 전사 입력 (TranscriptSource: fixture 재생 / 오디오 파일 배치 STT / 실시간 마이크 STT)
├── structurer/          # 전사 → 지식 구조 (실시간 재종합, 초기 목차 제안, 전사 오인식 교정, 최종 정리)
├── question_engine/     # 지식 구조 → 추천 질문 (생성 + "지금이 물어볼 때인가" 배경 판단)
├── storage/             # 세션 로그 · 채택 로그 · 구조화 결과를 로컬 파일(JSON/JSONL)로 저장
└── app/                 # Flask 웹 UI — 위 모듈들을 배선하는 곳
scripts/                 # 웹 서버 진입점, STT 벤더 비교 도구
schemas/                 # 사용자가 관리하는 인터뷰 스키마 JSON
audio/                   # 오디오 파일 테스트용 파일 — 로컬 전용, git에는 안 올라감
sessions/                # 세션마다 자동 생성되는 결과(전사·구조화·채택 로그) — 개인 데이터라 git에는 안 올라감
```

## 핵심 설계 원칙

- **모듈 독립성** — 모든 통신은 `contracts.py`의 데이터 타입으로만. 어떤 모듈도 다른 모듈의 내부를 직접 import하지 않습니다.
- **STT 디커플링** — 전사 입력은 `TranscriptSource` 인터페이스 뒤에 있어, STT 없이도 fixture 재생만으로 하류 전체를 개발·테스트할 수 있습니다.
- **풀(pull) 방식** — 추천 질문은 AI가 대화에 끼어드는 게 아니라, 인터뷰어가 트리거(버튼 또는 배경 판단 통과)했을 때만 표출됩니다.
- **개인용·로컬 전용** — 단일 사용자, 맥북 로컬 실행을 전제로 합니다. 서버/멀티유저/인증은 범위 밖입니다.
