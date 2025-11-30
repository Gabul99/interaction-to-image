import { type FeedbackRecord } from "../types";
import { API_BASE_URL } from "../config/api";

/**
 * ============================================================================
 * 백엔드 API 스펙: 브랜치 생성
 * ============================================================================
 * 
 * 엔드포인트: POST /api/branch/create
 * 
 * 기능:
 * - 사용자가 특정 이미지 노드에서 피드백을 주어 브랜치를 생성
 * - 피드백 정보를 받아 새로운 이미지 생성 파이프라인 시작
 * - 브랜치 ID와 WebSocket URL 반환
 * 
 * 요청 형식:
 * {
 *   "sessionId": "session_1234567890",
 *   "sourceNodeId": "node_image_1234567890_10",
 *   "feedback": [
 *     {
 *       "id": "feedback_1234567890_0",
 *       "area": "full",              // "full" | "bbox" | "point"
 *       "type": "text",              // "text" | "image"
 *       "text": "더 밝게 만들어주세요",
 *       "timestamp": 1234567890
 *     },
 *     {
 *       "id": "feedback_1234567890_1",
 *       "area": "bbox",
 *       "type": "text",
 *       "text": "이 부분을 더 선명하게",
 *       "bbox": {
 *         "x": 0.2,
 *         "y": 0.3,
 *         "width": 0.4,
 *         "height": 0.5
 *       },
 *       "timestamp": 1234567890
 *     },
 *     {
 *       "id": "feedback_1234567890_2",
 *       "area": "point",
 *       "type": "text",
 *       "text": "이 지점을 중심으로",
 *       "point": { "x": 0.5, "y": 0.6 },
 *       "timestamp": 1234567890
 *     },
 *     {
 *       "id": "feedback_1234567890_3",
 *       "area": "full",
 *       "type": "image",
 *       "imageUrl": "https://example.com/reference_image.png",
 *       "timestamp": 1234567890
 *     }
 *   ]
 * }
 * 
 * 응답 형식:
 * {
 *   "branchId": "branch_1234567890",
 *   "websocketUrl": "ws://localhost:8000/ws/image-stream/session_1234567890/branch_1234567890"
 * }
 * 
 * 백엔드 구현 요구사항:
 * 1. 브랜치 ID 생성 (형식: branch_{timestamp})
 * 2. 피드백 정보 저장 및 관리
 * 3. sourceNodeId에서 시작하는 새로운 이미지 생성 파이프라인 시작
 * 4. 피드백을 기반으로 이미지 생성 조건 설정
 * 5. WebSocket을 통해 브랜치의 이미지 스트림 전송 시작
 * 6. 여러 브랜치가 동시에 생성될 수 있으므로 병렬 처리 필요
 * 
 * 피드백 타입별 처리:
 * - "full" + "text": 전체 이미지에 대한 텍스트 피드백
 * - "full" + "image": 전체 이미지에 대한 참조 이미지 피드백
 * - "bbox" + "text": 특정 영역에 대한 텍스트 피드백
 * - "bbox" + "image": 특정 영역에 대한 참조 이미지 피드백
 * - "point" + "text": 특정 포인트에 대한 텍스트 피드백
 * 
 * @param sessionId 세션 ID
 * @param sourceNodeId 브랜치가 시작되는 노드 ID
 * @param feedback 피드백 리스트
 * @returns 브랜치 정보 (branchId, websocketUrl)
 * 
 * @see BACKEND_SPEC.md 섹션 4
 */
export async function createBranch(
  sessionId: string,
  sourceNodeId: string,
  feedback: FeedbackRecord[]
): Promise<{ branchId: string; websocketUrl?: string }> {
  console.log("[API] 브랜치 생성 요청:", {
    sessionId,
    sourceNodeId,
    feedbackCount: feedback.length,
  });

  try {
    const response = await fetch(`${API_BASE_URL}/api/branch/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId,
        sourceNodeId,
        feedback: feedback.map(f => ({
        id: f.id,
        area: f.area,
        type: f.type,
        timestamp: f.timestamp,
          text: f.text,
          imageUrl: f.imageUrl,
          point: f.point,
          bbox: f.bbox,
          bboxId: f.bboxId,
        })),
      }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("[API] 브랜치 생성 완료:", data);
    return {
      branchId: data.branchId,
      websocketUrl: data.websocketUrl,
    };
  } catch (error) {
    console.error("[API] 브랜치 생성 실패:", error);
    throw error;
  }
}

