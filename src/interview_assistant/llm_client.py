"""OpenRouter 클라이언트 생성 — Structurer/Question Engine이 공통으로 쓰는 인프라 유틸리티."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"

# 여러 LLM 호출 모듈(structure_synthesizer/outline_seeder/question_engine/trigger_judge)이
# 전부 strict JSON Schema structured output을 쓰는데, response_format/extra_body
# 모양이 토씨 하나 안 틀리고 반복되던 걸 여기로 모았다.
STRUCTURED_EXTRA_BODY = {"provider": {"require_parameters": True}}


def get_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])


def structured_response_format(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
