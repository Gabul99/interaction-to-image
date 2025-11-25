import { type ImageStep } from "../stores/imageStore";

/**
 * WebSocket 연결을 통해 이미지 스트림을 받습니다.
 * 
 * @param sessionId 세션 ID
 * @param websocketUrl WebSocket URL (선택적, 없으면 기본 URL 사용)
 * @param onImageStep 이미지 스텝 수신 시 호출되는 콜백
 * @param onError 에러 발생 시 호출되는 콜백
 * @param onComplete 스트림 완료 시 호출되는 콜백
 * @returns WebSocket 연결 객체 (닫기 위해 사용)
 */
export function connectImageStream(
  sessionId: string,
  websocketUrl: string,
  onImageStep: (step: ImageStep) => void,
  onError: (error: Error) => void,
  onComplete: () => void
): WebSocket | null {
  console.log("[WebSocket] 이미지 스트림 연결 시도:", websocketUrl);

  try {
    // TODO: 실제 백엔드 연결 시 아래 주석을 해제하고 mockup 코드를 제거
    /*
    const ws = new WebSocket(websocketUrl);
    
    ws.onopen = () => {
      console.log("[WebSocket] 연결 성공");
      // 세션 ID 전송 (필요한 경우)
      ws.send(JSON.stringify({ type: 'subscribe', sessionId }));
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'image_step') {
          const imageStep: ImageStep = {
            id: data.stepId,
            url: data.imageUrl,
            step: data.step,
            timestamp: data.timestamp || Date.now(),
          };
          onImageStep(imageStep);
        } else if (data.type === 'complete') {
          onComplete();
          ws.close();
        } else if (data.type === 'error') {
          onError(new Error(data.message || 'Unknown error'));
        }
      } catch (error) {
        console.error("[WebSocket] 메시지 파싱 실패:", error);
        onError(error as Error);
      }
    };
    
    ws.onerror = (error) => {
      console.error("[WebSocket] 에러 발생:", error);
      onError(new Error('WebSocket connection error'));
    };
    
    ws.onclose = () => {
      console.log("[WebSocket] 연결 종료");
    };
    
    return ws;
    */

    // ===== MOCKUP (백엔드 연결 전까지 사용) =====
    // 실제 WebSocket 대신 시뮬레이션
    // 이 부분은 imageStore의 simulateImageStream에서 처리되므로
    // 여기서는 null을 반환하여 mockup 모드임을 표시
    console.log("[WebSocket] Mockup 모드: 실제 WebSocket 연결 없음");
    return null;
    // ===== MOCKUP END =====
  } catch (error) {
    console.error("[WebSocket] 연결 실패:", error);
    onError(error as Error);
    return null;
  }
}

/**
 * WebSocket 연결을 닫습니다.
 */
export function disconnectImageStream(ws: WebSocket | null): void {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
    console.log("[WebSocket] 연결 닫기");
  }
}

