import { type ImageStep } from "../stores/imageStore";

/**
 * ============================================================================
 * 백엔드 WebSocket 스펙: 이미지 스트림 연결
 * ============================================================================
 * 
 * WebSocket URL 형식:
 * - 메인 브랜치: ws://{host}/ws/image-stream/{sessionId}
 * - 브랜치: ws://{host}/ws/image-stream/{sessionId}/{branchId}
 * 
 * 서버 → 클라이언트 메시지 형식:
 * 
 * 1. 이미지 스텝 (image_step):
 * {
 *   "type": "image_step",
 *   "sessionId": "session_1234567890",
 *   "branchId": "branch_1234567890",  // 브랜치인 경우만
 *   "nodeId": "node_image_1234567890_1",
 *   "parentNodeId": "node_prompt_1234567890",
 *   "step": 5,
 *   "totalSteps": 20,
 *   "imageUrl": "https://example.com/images/step_5.png",
 *   "imageData": "base64_encoded_image_data", // 또는 URL만 사용
 *   "timestamp": 1234567890
 * }
 * 
 * 2. 생성 완료 (generation_complete):
 * {
 *   "type": "generation_complete",
 *   "sessionId": "session_1234567890",
 *   "branchId": "branch_1234567890",  // 브랜치인 경우만
 *   "nodeId": "node_image_1234567890_20",
 *   "finalImageUrl": "https://example.com/images/final.png"
 * }
 * 
 * 3. 에러 (error):
 * {
 *   "type": "error",
 *   "code": "ERROR_CODE",
 *   "message": "에러 메시지"
 * }
 * 
 * 백엔드 구현 요구사항:
 * 1. 각 diffusion step마다 이미지를 생성하여 전송
 * 2. step은 현재 단계, totalSteps는 전체 단계 수
 * 3. nodeId는 각 이미지 노드의 고유 ID
 * 4. parentNodeId는 이전 노드의 ID (체인 구조)
 * 5. 이미지는 base64 인코딩 또는 URL로 전송 가능
 * 6. 여러 브랜치가 동시에 생성될 수 있으므로 병렬 처리 필요
 * 7. 각 브랜치는 독립적인 WebSocket 스트림을 가질 수 있음
 * 
 * @param sessionId 세션 ID
 * @param websocketUrl WebSocket URL
 * @param onImageStep 이미지 스텝 수신 시 호출되는 콜백
 * @param onError 에러 발생 시 호출되는 콜백
 * @param onComplete 스트림 완료 시 호출되는 콜백
 * @returns WebSocket 연결 객체 (닫기 위해 사용)
 * 
 * @see BACKEND_SPEC.md 섹션 3
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

