import { type ImageStep } from "../stores/imageStore";
import { API_BASE_URL } from "../config/api";

/**
 * WebSocket 연결을 통해 이미지 스트림을 받습니다.
 * 
 * @param sessionId 세션 ID
 * @param websocketUrl WebSocket URL (선택적, 없으면 기본 URL 사용)
 * @param onImageStep 이미지 스텝 수신 시 호출되는 콜백
 * @param onError 에러 발생 시 호출되는 콜백
 * @param onComplete 스트림 완료 시 호출되는 콜백
 * @param onFeedbackRequest 피드백 요청 수신 시 호출되는 콜백 (선택적)
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
    const ws = new WebSocket(websocketUrl);
    
    ws.onopen = () => {
      console.log("[WebSocket] 연결 성공:", websocketUrl);
    };
    
    ws.onmessage = (event) => {
      try {
        // 메시지가 너무 크면 일부만 로그
        const messagePreview = event.data.length > 200 ? event.data.substring(0, 200) + '...' : event.data;
        console.log("[WebSocket] 원본 메시지 수신 (길이:", event.data.length, "):", messagePreview);
        
        const data = JSON.parse(event.data);
        console.log("[WebSocket] 파싱된 메시지:", data.type, data.step ? `step=${data.step}` : "", data.nodeId ? `nodeId=${data.nodeId}` : "");
        
        if (data.type === 'connected') {
          // WebSocket 연결 확인 메시지
          console.log("[WebSocket] 연결 확인됨:", data.sessionId);
        } else if (data.type === 'image_step') {
          console.log("[WebSocket] ✅ 이미지 스텝 메시지 수신:", {
            step: data.step,
            nodeId: data.nodeId,
            parentNodeId: data.parentNodeId,
            imageUrlLength: data.imageUrl ? data.imageUrl.length : 0,
            hasImageUrl: !!data.imageUrl,
            imageUrlPreview: data.imageUrl ? data.imageUrl.substring(0, 50) + '...' : '없음'
          });
          
          if (!data.imageUrl) {
            console.error("[WebSocket] ❌ imageUrl이 없습니다!", data);
            return;
          }
          
          // 이미지 URL 처리 (상대 경로인 경우 API_BASE_URL 추가)
          let imageUrl = data.imageUrl || data.imageData || '';
          if (imageUrl && imageUrl.startsWith('/images/')) {
            // 상대 경로인 경우 API_BASE_URL 추가
            imageUrl = `${API_BASE_URL}${imageUrl}`;
          }
          
          const imageStep: ImageStep = {
            id: data.nodeId || data.stepId || `step_${data.step}`,
            url: imageUrl,
            step: data.step || 0,
            timestamp: data.timestamp || Date.now(),
            // 그래프 구조를 위한 추가 정보
            nodeId: data.nodeId,
            parentNodeId: data.parentNodeId,
            sessionId: data.sessionId,
            branchId: data.branchId,
          } as ImageStep & { nodeId?: string; parentNodeId?: string; sessionId?: string; branchId?: string };
          
          console.log("[WebSocket] 이미지 스텝 객체 생성 완료, 콜백 호출:", {
            step: imageStep.step,
            nodeId: imageStep.nodeId,
            parentNodeId: imageStep.parentNodeId,
            urlLength: imageStep.url.length
          });
          
          onImageStep(imageStep);
        } else if (data.type === 'generation_complete' || data.type === 'complete') {
          console.log("[WebSocket] 생성 완료");
          onComplete();
          ws.close();
        } else if (data.type === 'feedback_request') {
          // 피드백 요청은 무시 (Branching 버튼으로 처리)
          console.log("[WebSocket] 피드백 요청 수신 (무시됨, Branching 버튼 사용):", data.step, data.message);
        } else if (data.type === 'error') {
          console.error("[WebSocket] 에러 메시지 수신:", data.message);
          onError(new Error(data.message || 'Unknown error'));
        } else {
          console.log("[WebSocket] 알 수 없는 메시지 타입:", data.type);
        }
      } catch (error) {
        console.error("[WebSocket] 메시지 파싱 실패:", error, "원본 데이터:", event.data.substring(0, 100));
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

