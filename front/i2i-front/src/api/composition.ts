import { type ObjectChip, type BoundingBox } from "../types";

/**
 * ============================================================================
 * 백엔드 API 스펙: 객체 리스트 생성
 * ============================================================================
 * 
 * 엔드포인트: POST /api/composition/objects
 * 
 * 기능:
 * - 프롬프트를 받아 LLM을 이용하여 이미지 생성에 필요한 객체 리스트를 생성
 * - 각 객체는 고유한 ID와 색상을 가져야 함
 * 
 * 요청 형식:
 * {
 *   "prompt": "a beautiful sunset over the ocean with a sailboat"
 * }
 * 
 * 응답 형식:
 * {
 *   "objects": [
 *     {
 *       "id": "obj_1234567890_0",
 *       "label": "Sunset",
 *       "color": "#6366f1"
 *     },
 *     ...
 *   ]
 * }
 * 
 * 백엔드 구현 요구사항:
 * 1. LLM (GPT-4, Claude 등)을 사용하여 프롬프트에서 객체 추출
 * 2. 각 객체에 고유한 ID 생성 (형식: obj_{timestamp}_{index})
 * 3. 각 객체에 색상 할당 (12가지 색상 중 선택)
 * 4. 객체는 이미지 생성에 필요한 주요 요소들을 포함해야 함
 * 
 * @param prompt 이미지 생성 프롬프트
 * @returns 객체 리스트 (LLM이 생성한 객체 목록)
 * 
 * @see BACKEND_SPEC.md 섹션 1
 */
export async function requestObjectList(prompt: string): Promise<ObjectChip[]> {
  console.log("[API] 객체 리스트 요청:", prompt);

  try {
    // TODO: 실제 백엔드 연결 시 아래 주석을 해제하고 mockup 코드를 제거
    /*
    const response = await fetch('/api/composition/objects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data.objects;
    */

    // ===== MOCKUP (백엔드 연결 전까지 사용) =====
    await new Promise((resolve) => setTimeout(resolve, 1000));

    // 프롬프트에서 키워드를 추출하여 객체 리스트 생성 (시뮬레이션)
    const keywords = prompt
      .toLowerCase()
      .split(/[,\s]+/)
      .filter((word) => word.length > 2)
      .slice(0, 5);

    const defaultObjects = [
      "person",
      "car",
      "tree",
      "building",
      "sky",
      "mountain",
      "water",
      "animal",
      "flower",
      "object",
    ];

    const objectLabels = keywords.length > 0 
      ? keywords 
      : defaultObjects.slice(0, 3);

    const colors = [
      "#6366f1", // indigo
      "#8b5cf6", // purple
      "#ec4899", // pink
      "#f43f5e", // rose
      "#ef4444", // red
      "#f59e0b", // amber
      "#eab308", // yellow
      "#84cc16", // lime
      "#22c55e", // green
      "#10b981", // emerald
    ];

    const objects: ObjectChip[] = objectLabels.map((label, index) => ({
      id: `obj_${Date.now()}_${index}`,
      label: label.charAt(0).toUpperCase() + label.slice(1),
      color: colors[index % colors.length],
    }));

    console.log("[API] 객체 리스트 수신:", objects);
    return objects;
    // ===== MOCKUP END =====
  } catch (error) {
    console.error("[API] 객체 리스트 요청 실패:", error);
    throw error;
  }
}

/**
 * ============================================================================
 * 백엔드 API 스펙: 이미지 생성 시작
 * ============================================================================
 * 
 * 엔드포인트: POST /api/composition/start
 * 
 * 기능:
 * - 사용자가 설정한 구도 정보를 받아 이미지 생성을 시작
 * - 구도 정보가 없을 수도 있음 (bboxes가 빈 배열이거나 없음)
 * - 세션 ID와 WebSocket URL을 반환
 * 
 * 요청 형식:
 * {
 *   "prompt": "a beautiful sunset over the ocean with a sailboat",
 *   "objects": [ { "id": "...", "label": "...", "color": "..." } ],
 *   "bboxes": [
 *     {
 *       "objectId": "obj_123",
 *       "x": 0.1,      // 상대 좌표 (0~1)
 *       "y": 0.2,      // 상대 좌표 (0~1)
 *       "width": 0.3,  // 상대 크기 (0~1)
 *       "height": 0.4  // 상대 크기 (0~1)
 *     }
 *   ]
 * }
 * 
 * 응답 형식:
 * {
 *   "sessionId": "session_1234567890",
 *   "rootNodeId": "node_prompt_1234567890",
 *   "websocketUrl": "ws://localhost:8000/ws/image-stream/session_1234567890"
 * }
 * 
 * 백엔드 구현 요구사항:
 * 1. 세션 ID 생성 및 관리
 * 2. 프롬프트 노드 ID 생성 (rootNodeId)
 * 3. WebSocket URL 생성 및 반환
 * 4. 구도 정보가 없는 경우에도 처리 가능해야 함
 * 5. 이미지 생성 파이프라인 초기화
 * 6. WebSocket 연결 준비 (이미지 스트림 시작)
 * 
 * @param prompt 이미지 생성 프롬프트
 * @param objects 객체 리스트 (선택적, 구도 설정이 있는 경우)
 * @param bboxes 바운딩 박스 리스트 (선택적, 구도 설정이 있는 경우)
 * @returns 세션 정보 (sessionId, rootNodeId, websocketUrl)
 * 
 * @see BACKEND_SPEC.md 섹션 2
 */
export async function startImageGeneration(
  prompt: string,
  objects?: ObjectChip[],
  bboxes?: Array<{ objectId: string; x: number; y: number; width: number; height: number }>
): Promise<{ sessionId: string; websocketUrl?: string }> {
  console.log("[API] 이미지 생성 시작 요청:", { prompt, objects, bboxes });

  try {
    // TODO: 실제 백엔드 연결 시 아래 주석을 해제하고 mockup 코드를 제거
    /*
    const response = await fetch('/api/composition/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        objects: objects || [],
        bboxes: bboxes || [],
      }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return {
      sessionId: data.sessionId,
      websocketUrl: data.websocketUrl, // 예: 'ws://localhost:8000/ws/image-stream/{sessionId}'
    };
    */

    // ===== MOCKUP (백엔드 연결 전까지 사용) =====
    await new Promise((resolve) => setTimeout(resolve, 500));
    
    const sessionId = `session_${Date.now()}`;
    console.log("[API] 이미지 생성 세션 시작:", sessionId);
    
    return {
      sessionId,
      // websocketUrl: `ws://localhost:8000/ws/image-stream/${sessionId}`, // 실제 백엔드 연결 시 사용
    };
    // ===== MOCKUP END =====
  } catch (error) {
    console.error("[API] 이미지 생성 시작 실패:", error);
    throw error;
  }
}

/**
 * @deprecated 이 함수는 startImageGeneration으로 통합되었습니다.
 * 하위 호환성을 위해 유지되지만, 새로운 코드에서는 startImageGeneration을 사용하세요.
 */
export async function sendComposition(
  prompt: string,
  bboxes: Array<{ objectId: string; x: number; y: number; width: number; height: number }>
): Promise<void> {
  console.warn("[API] sendComposition은 deprecated입니다. startImageGeneration을 사용하세요.");
  
  try {
    await startImageGeneration(prompt, undefined, bboxes);
  } catch (error) {
    console.error("[API] 구도 설정 전송 실패:", error);
    throw error;
  }
}
