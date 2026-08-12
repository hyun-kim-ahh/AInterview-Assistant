"""
4개 STT 벤더(Deepgram/Naver Clova Speech/OpenAI/AssemblyAI) 배치 전사 비교 (dev-plan
15단계, Milestone 1). fixtures/stt_sample_ko.wav(macOS 내장 TTS로 합성한 한국어 오디오)를
각 벤더에 흘려서 전사 텍스트·턴별 타임스탬프·실측 응답 시간·대략 비용을 나란히 출력한다.

개발 시점 평가 도구다 — 여기서 나온 결과를 보고 실시간 마이크용 벤더를 하나 확정한다
(Milestone 2). 환경변수(.env)에 키가 없는 벤더는 죽지 않고 건너뛴다.

비용은 조사 시점(2026-07-28) 공식 요금 페이지 기준 근사치다 — 실제 청구 금액은 각
콘솔에서 재확인할 것.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from interview_assistant.sources.base import TranscriptSource  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

AUDIO_PATH = REPO_ROOT / "fixtures" / "stt_sample_ko.wav"
REFERENCE_TEXT_PATH = REPO_ROOT / "fixtures" / "stt_sample_ko.txt"

# 분당 요금(달러, 근사치) — 조사 시점 공식 페이지 기준. 오디오 길이는 초 단위로 환산해 곱함.
COST_PER_MINUTE_USD = {
    "Deepgram": 0.0077,       # Nova-3 배치, monolingual
    "Naver Clova Speech": None,  # 콘솔에서 직접 확인 필요 — 요금 미확인
    "OpenAI (whisper-1)": 0.006,
    "AssemblyAI": 0.0025,     # Universal-2, $0.15/hour
}


def _vendor_sources() -> list[tuple[str, str, type[TranscriptSource]]]:
    """(표시 이름, 필요 환경변수 중 하나라도 없으면 건너뜀 판단용 키 목록, 클래스)."""
    from interview_assistant.sources.stt.assemblyai_source import AssemblyAIFileSTTSource
    from interview_assistant.sources.stt.deepgram_source import DeepgramFileSTTSource
    from interview_assistant.sources.stt.naver_clova_source import NaverClovaFileSTTSource
    from interview_assistant.sources.stt.openai_whisper_source import (
        OpenAIWhisperFileSTTSource,
    )

    return [
        ("Deepgram", ["DEEPGRAM_API_KEY"], DeepgramFileSTTSource),
        ("Naver Clova Speech", ["NAVER_CLOVA_SECRET_KEY"], NaverClovaFileSTTSource),
        ("OpenAI (whisper-1)", ["OPENAI_API_KEY"], OpenAIWhisperFileSTTSource),
        ("AssemblyAI", ["ASSEMBLYAI_API_KEY"], AssemblyAIFileSTTSource),
    ]


def main() -> None:
    if not AUDIO_PATH.exists():
        print(f"오디오 fixture가 없습니다: {AUDIO_PATH}")
        return

    print(f"[정답 스크립트]\n{REFERENCE_TEXT_PATH.read_text(encoding='utf-8').strip()}\n")

    for name, required_env, source_cls in _vendor_sources():
        missing = [k for k in required_env if not os.environ.get(k)]
        print(f"{'─' * 60}\n{name}\n{'─' * 60}")
        if missing:
            print(f"  환경변수 없음({', '.join(missing)}) — 건너뜀\n")
            continue

        start = time.monotonic()
        try:
            events = list(source_cls(AUDIO_PATH).stream())
        except Exception as exc:  # noqa: BLE001 — 비교 도구라 한 벤더 실패가 전체를 죽이면 안 됨
            print(f"  호출 실패: {exc}\n")
            continue
        elapsed = time.monotonic() - start

        full_text = " ".join(e.text for e in events)
        print(f"  응답 시간: {elapsed:.2f}s · 턴 수: {len(events)}")
        for e in events:
            print(f"    [{e.turn_id} @ {e.timestamp:.1f}s] {e.text}")
        print(f"  [전체 텍스트] {full_text}")

        cost_per_min = COST_PER_MINUTE_USD.get(name)
        if cost_per_min is not None and events:
            duration_min = events[-1].timestamp / 60.0
            print(f"  대략 비용: ${cost_per_min * duration_min:.4f} (참고치, 실제 요금은 콘솔에서 확인)")
        print()


if __name__ == "__main__":
    main()
