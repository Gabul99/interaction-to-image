/**
 * API 및 WebSocket 기본 URL 설정
 *
 * 환경 변수를 통해 설정할 수 있습니다:
 * - VITE_API_BASE_URL: API 기본 URL (예: http://localhost:8000)
 * - VITE_WS_BASE_URL: WebSocket 기본 URL (예: ws://localhost:8000)
 *
 * SSH 터널링을 사용하는 경우:
 * - 로컬: http://localhost:8001 (SSH 포워딩된 포트)
 * - 원격 서버: http://localhost:8000
 */

const getApiBaseUrl = (): string => {
    // Vite 환경 변수 접근
    const apiUrl = import.meta.env.VITE_API_BASE_URL;
    if (apiUrl) {
      return apiUrl;
    }
  
    // 브라우저의 호스트명을 확인하여 원격 서버에서 직접 접근하는지 판단
    const hostname = window.location.hostname;
  
    // 원격 서버에서 직접 접근하는 경우 (예: gpu10 또는 서버 IP 주소)
    if (hostname !== "localhost" && hostname !== "127.0.0.1") {
      // 원격 서버의 호스트명을 사용하여 백엔드에 접근
      // 원격 서버 IP: 10.2.11.20, 10.10.1.20
      return `http://${hostname}:8000`;
    }
  
    // 로컬에서 접근하는 경우 SSH 터널링된 포트 사용
    // 또는 원격 서버 IP를 직접 사용 (SSH 터널링 대신)
    // 원격 서버 IP: 10.2.11.20 또는 10.10.1.20
    // 하지만 사설 IP이므로 SSH 터널링이 필요할 수 있음
    return "http://localhost:8001";
  };
  
  const getWebSocketBaseUrl = (): string => {
    // Vite 환경 변수 접근
    const wsUrl = import.meta.env.VITE_WS_BASE_URL;
    if (wsUrl) {
      return wsUrl;
    }
  
    // WebSocket은 별도 포트(8001)에서 실행됨
    // SSH 터널링을 사용하는 경우, 로컬에서는 HTTP 포트 8001을 사용하므로
    // WebSocket도 같은 포트를 사용하되 경로로 구분하거나, 다른 포트를 사용
    const apiUrl = getApiBaseUrl();
    const hostname = window.location.hostname;
  
    // 원격 서버에서 직접 접근하는 경우
    if (hostname !== "localhost" && hostname !== "127.0.0.1") {
      return `ws://${hostname}:8001`;
    }
  
    // 로컬에서 접근하는 경우 (SSH 터널링)
    // HTTP는 8001, WebSocket은 8003 사용 (SSH 터널링: 8003 -> 원격 8001)
    if (apiUrl.includes(":8001")) {
      return "ws://localhost:8003";
    }
  
    // 기본값
    return "ws://localhost:8001";
  };
  
  export const API_BASE_URL = getApiBaseUrl();
  export const WS_BASE_URL = getWebSocketBaseUrl();
  
  /**
   * Mock 모드 설정
   *
   * true로 설정하면 실제 서버 연결 없이 더미 데이터로 시뮬레이션합니다.
   * 로컬에서 서버를 돌릴 수 없는 상황에서 테스트할 때 사용합니다.
   *
   * 환경 변수로 설정: VITE_USE_MOCK_MODE=true
   * 또는 이 파일에서 직접 수정: export const USE_MOCK_MODE = true;
   */
  const getUseMockMode = (): boolean => {
    // Vite 환경 변수 접근 (import.meta.env 사용)
    // @ts-ignore - Vite 환경 변수는 타입 정의가 없을 수 있음
    const mockMode = import.meta.env.VITE_USE_MOCK_MODE;
  
    // 디버깅: 모든 환경 변수 확인
    console.log(`[Config] ========== 환경 변수 디버깅 ==========`);
    console.log(`[Config] import.meta.env:`, import.meta.env);
    console.log(
      `[Config] VITE_USE_MOCK_MODE 값:`,
      mockMode,
      `(타입: ${typeof mockMode})`
    );
    console.log(`[Config] import.meta.env.MODE:`, import.meta.env.MODE);
    console.log(`[Config] import.meta.env.DEV:`, import.meta.env.DEV);
    console.log(`[Config] import.meta.env.PROD:`, import.meta.env.PROD);
  
    if (mockMode !== undefined && mockMode !== null && mockMode !== "") {
      // 대소문자 구분 없이 체크 (TRUE, true, True 모두 허용)
      const mockModeStr = String(mockMode).toLowerCase().trim();
      const isEnabled =
        mockModeStr === "true" ||
        mockMode === true ||
        mockMode === 1 ||
        mockModeStr === "1";
      console.log(
        `[Config] Mock 모드 파싱 결과:`,
        isEnabled,
        `(원본: "${mockMode}", 변환: "${mockModeStr}")`
      );
      return isEnabled;
    }
  
    // 기본값: false (실제 서버 연결)
    console.log(
      `[Config] ⚠️ VITE_USE_MOCK_MODE 환경 변수가 설정되지 않음 (값: ${mockMode}), 기본값 false 사용`
    );
    console.log(`[Config] 💡 .env 파일 위치 확인: front/i2i-front/.env`);
    console.log(`[Config] 💡 .env 파일 내용: VITE_USE_MOCK_MODE=true`);
    console.log(`[Config] 💡 개발 서버를 재시작했는지 확인하세요!`);
    return false;
  };
  
  // 환경 변수에서 읽기 시도, 실패하면 코드에서 직접 설정
  let USE_MOCK_MODE_VALUE = getUseMockMode();
  
  // 환경 변수가 로드되지 않은 경우, 코드에서 직접 설정할 수 있음
  // 아래 주석을 해제하고 true로 설정하면 Mock 모드가 강제로 활성화됩니다
  // const USE_MOCK_MODE_VALUE = true; // 강제로 Mock 모드 활성화
  
  export const USE_MOCK_MODE = USE_MOCK_MODE_VALUE;
  
  console.log(
    `[Config] Mock 모드 최종 상태: ${USE_MOCK_MODE ? "활성화 ✅" : "비활성화 ❌"}`
  );
  