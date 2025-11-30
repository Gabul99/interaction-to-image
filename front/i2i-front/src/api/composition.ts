import { type ObjectChip, type BoundingBox } from "../types";
import { API_BASE_URL, WS_BASE_URL } from "../config/api";

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
    const response = await fetch(`${API_BASE_URL}/api/composition/objects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("[API] 객체 리스트 수신:", data.objects);
    return data.objects;

    // ===== MOCKUP (백엔드 연결 전까지 사용) - 제거됨 =====
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
): Promise<{ sessionId: string; rootNodeId?: string; websocketUrl?: string }> {
  console.log("=".repeat(80));
  console.log("[API] ========== 이미지 생성 시작 요청 ==========");
  console.log("[API] 프롬프트:", prompt);
  console.log("[API] 객체 리스트:", objects);
  console.log("[API] 바운딩 박스:", bboxes);
  console.log("[API] API_BASE_URL:", API_BASE_URL);
  console.log("[API] 요청 URL:", `${API_BASE_URL}/api/composition/start`);
  console.log("=".repeat(80));

  try {
    const requestBody = {
      prompt,
      objects: objects || [],
      bboxes: bboxes || [],
      width: 512,
      height: 512,
      num_inference_steps: 50,
    };
    console.log("[API] 요청 본문:", JSON.stringify(requestBody, null, 2));
    
    const response = await fetch(`${API_BASE_URL}/api/composition/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });
    
    console.log("[API] 응답 상태:", response.status, response.statusText);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error("[API] 에러 응답 본문:", errorText);
      throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
    }
    
    const data = await response.json();
    console.log("=".repeat(80));
    console.log("[API] ========== 이미지 생성 세션 시작 성공 ==========");
    console.log("[API] 응답 데이터:", JSON.stringify(data, null, 2));
    console.log("[API] sessionId:", data.sessionId);
    console.log("[API] rootNodeId:", data.rootNodeId);
    console.log("[API] websocketUrl (원본):", data.websocketUrl);
    console.log("=".repeat(80));
    
    // WebSocket URL 변환 (SSH 터널링을 위해)
    // 백엔드가 ws://localhost:8001을 반환하지만, 로컬에서는 ws://localhost:8003을 사용해야 함
    let websocketUrl = data.websocketUrl;
    if (!websocketUrl) {
      console.error("[API] ⚠️⚠️⚠️ WebSocket URL이 응답에 없습니다! 백엔드 응답:", data);
      // 기본 URL 사용
      websocketUrl = `${WS_BASE_URL}/ws/image-stream/${data.sessionId}`;
    } else if (websocketUrl.includes('localhost:8001')) {
      // 원격 서버의 8001 포트를 로컬의 8003으로 변경 (SSH 터널링)
      websocketUrl = websocketUrl.replace('localhost:8001', 'localhost:8003');
      console.log("[API] WebSocket URL을 SSH 터널링 포트로 변경:", websocketUrl);
    } else if (websocketUrl.includes('localhost:8000')) {
      // 원격 서버의 8000 포트를 로컬의 8001로 변경 (SSH 터널링)
      websocketUrl = websocketUrl.replace('localhost:8000', 'localhost:8001');
      console.log("[API] WebSocket URL을 SSH 터널링 포트로 변경:", websocketUrl);
    } else {
      // 경로만 추출하여 config의 WS_BASE_URL 사용
      try {
        const urlObj = new URL(websocketUrl);
        const path = urlObj.pathname;
        websocketUrl = `${WS_BASE_URL}${path}`;
        console.log("[API] WebSocket URL을 config 기반으로 변경:", websocketUrl);
      } catch (e) {
        console.warn("[API] WebSocket URL 파싱 실패, 원본 사용:", websocketUrl);
      }
    }
    
    const result = {
      sessionId: data.sessionId,
      rootNodeId: data.rootNodeId,
      websocketUrl: websocketUrl,
    };
    
    console.log("[API] 반환할 결과:", result);
    return result;
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
