"""웹 UI 진입점 (dev-plan 10단계).

포트는 5000이 아니라 5001을 쓴다 — macOS의 AirPlay 수신 기능(제어센터)이 기본적으로
5000번 포트를 점유해서, 5000번으로 띄우면 브라우저에서 "접속 거부"가 나는 경우가 있다
(curl 등 일부 클라이언트는 우연히 통과되기도 해 헷갈리기 쉽다).
"""

from interview_assistant.app.web import app

if __name__ == "__main__":
    # threaded=True(2026-08-11) — 안 켜면 개발 서버가 요청을 한 번에 하나씩만
    # 처리한다. /end·/ask처럼 LLM 호출로 오래 걸리는 요청이 처리되는 동안은
    # 화면이 2초마다 보내는 /status 폴링(session.html의 pollStatus)도 전부 막혀서,
    # 서버 메모리엔 이미 세워진 상태(예: summarizing=True)를 화면이 한참 뒤에야
    # 반영하는 것처럼 보였다.
    #
    # use_reloader=False(2026-08-11 추가) — 실시간 마이크 모드가 작동하지 않는
    # 문제의 실제 원인이었음. debug=True의 기본 리로더(Werkzeug)는 코드를 감시하는
    # "감시자" 프로세스와 실제로 요청을 처리하는 "워커" 자식 프로세스, 이렇게 두
    # 개의 Python 프로세스를 동시에 띄운다 — 그런데 이 파일 맨 위의
    # `from interview_assistant.app.web import app`이 두 프로세스 모두에서 실행되며
    # sounddevice(PortAudio)를 각자 초기화한다. 같은 마이크 장치를 두 프로세스가
    # 동시에 "알고 있는" 상태가 되면서, 워커 프로세스가 실제로 마이크 스트림을 열
    # 때 macOS CoreAudio가 `Invalid Property Value`(PortAudio 에러코드 -9986) 에러를
    # 내고, 이게 백그라운드 스레드에서 조용히 죽어버려 화면엔 아무 표시 없이
    # 마이크가 그냥 안 들리는 것처럼 보였다(실 서버로 재현: 리로더 켜짐 2/2 실패,
    # 리로더 꺼짐 2/2 성공). Werkzeug의 인터랙티브 디버거(예외 발생 시 트레이스백
    # 페이지)는 debug=True 자체로 계속 켜져 있으니 잃는 기능 없음 — 코드 수정 후
    # 자동 재시작만 안 된다(이 프로젝트는 애초에 리로더를 안 믿고 수동으로
    # 재시작하는 걸 권장해왔다 — CLAUDE.md/메모리의 로컬 검증 절차 참고).
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False, threaded=True)
