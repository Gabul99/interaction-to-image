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
  // Vite 환경 변수 접근 (타입 안전하지 않지만 런타임에서는 작동)
  const meta = import.meta as any;
  const apiUrl = meta.env?.VITE_API_BASE_URL;
  if (apiUrl) {
    return apiUrl;
  }
  
  // 브라우저의 호스트명을 확인하여 원격 서버에서 직접 접근하는지 판단
  const hostname = window.location.hostname;
  
  // 원격 서버에서 직접 접근하는 경우 (예: gpu10 또는 서버 IP 주소)
  if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
    // 원격 서버의 호스트명을 사용하여 백엔드에 접근
    // 원격 서버 IP: 10.2.11.20, 10.10.1.20
    return `http://${hostname}:8000`;
  }
  
  // 로컬에서 접근하는 경우 SSH 터널링된 포트 사용
  // 또는 원격 서버 IP를 직접 사용 (SSH 터널링 대신)
  // 원격 서버 IP: 10.2.11.20 또는 10.10.1.20
  // 하지만 사설 IP이므로 SSH 터널링이 필요할 수 있음
  return 'http://localhost:8001';
};

const getWebSocketBaseUrl = (): string => {
  // Vite 환경 변수 접근
  const meta = import.meta as any;
  const wsUrl = meta.env?.VITE_WS_BASE_URL;
  if (wsUrl) {
    return wsUrl;
  }
  
  // WebSocket은 별도 포트(8001)에서 실행됨
  // SSH 터널링을 사용하는 경우, 로컬에서는 HTTP 포트 8001을 사용하므로
  // WebSocket도 같은 포트를 사용하되 경로로 구분하거나, 다른 포트를 사용
  const apiUrl = getApiBaseUrl();
  const hostname = window.location.hostname;
  
  // 원격 서버에서 직접 접근하는 경우
  if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
    return `ws://${hostname}:8001`;
  }
  
  // 로컬에서 접근하는 경우 (SSH 터널링)
  // HTTP는 8001, WebSocket은 8003 사용 (SSH 터널링: 8003 -> 원격 8001)
  if (apiUrl.includes(':8001')) {
    return 'ws://localhost:8003';
  }
  
  // 기본값
  return 'ws://localhost:8001';
};

export const API_BASE_URL = getApiBaseUrl();
export const WS_BASE_URL = getWebSocketBaseUrl();

