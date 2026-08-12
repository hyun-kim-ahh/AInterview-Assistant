"""테스트 세션 시작 시 .env를 로드한다.

기존에는 llm_client.py가 임포트되는 김에 load_dotenv()가 간접적으로 불려서
OPENROUTER_API_KEY 기반 skipif가 우연히 맞물렸지만, STT 벤더 소스(15단계)는 그
임포트 체인을 안 타므로 명시적으로 한 번 불러야 한다.
"""

from dotenv import load_dotenv

load_dotenv()
