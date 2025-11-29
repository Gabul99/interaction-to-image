import { type FeedbackRecord } from "../types";

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
    // TODO: 실제 백엔드 연결 시 아래 주석을 해제하고 mockup 코드를 제거
    /*
    const formData = new FormData();
    formData.append("sessionId", sessionId);
    formData.append("sourceNodeId", sourceNodeId);
    
    // 피드백 배열을 JSON으로 직렬화
    const feedbacksJson = feedback.map(f => {
      const feedbackObj: any = {
        id: f.id,
        area: f.area,
        type: f.type,
        timestamp: f.timestamp,
      };
      
      if (f.text) feedbackObj.text = f.text;
      if (f.imageUrl) feedbackObj.imageUrl = f.imageUrl;
      if (f.point) {
        feedbackObj.point = { x: f.point.x, y: f.point.y };
      }
      if (f.bbox) {
        feedbackObj.bbox = {
          x: f.bbox.x,
          y: f.bbox.y,
          width: f.bbox.width,
          height: f.bbox.height,
        };
      }
      if (f.bboxId) feedbackObj.bboxId = f.bboxId;
      
      return feedbackObj;
    });
    
    formData.append("feedback", JSON.stringify(feedbacksJson));
    
    // 참조 이미지 파일이 있는 경우 별도로 추가
    feedback.forEach((f, index) => {
      if (f.imageUrl && f.imageUrl.startsWith('data:')) {
        // base64 이미지를 파일로 변환하여 전송
        const blob = dataURLtoBlob(f.imageUrl);
        formData.append(`reference_image_${index}`, blob);
      }
    });
    
    const response = await fetch('/api/branch/create', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return {
      branchId: data.branchId,
      websocketUrl: data.websocketUrl,
    };
    */

    // ===== MOCKUP (백엔드 연결 전까지 사용) =====
    await new Promise((resolve) => setTimeout(resolve, 500));
    
    const branchId = `branch_${Date.now()}`;
    console.log("[API] 브랜치 생성 완료 (Mockup):", branchId);
    
    return {
      branchId,
      // websocketUrl: `ws://localhost:8000/ws/image-stream/${sessionId}/${branchId}`, // 실제 백엔드 연결 시 사용
    };
    // ===== MOCKUP END =====
  } catch (error) {
    console.error("[API] 브랜치 생성 실패:", error);
    throw error;
  }
}

