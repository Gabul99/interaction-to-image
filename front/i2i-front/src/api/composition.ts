import { type ObjectChip, type BoundingBox } from "../types";

/**
 * API Endpoint: POST /api/composition/objects
 * 
 * 프롬프트를 서버로 전송하고 객체 리스트를 받아옵니다.
 * 
 * @param prompt 이미지 생성 프롬프트
 * @returns 객체 리스트 (LLM이 생성한 객체 목록)
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
 * API Endpoint: POST /api/composition/start
 * 
 * 객체 리스트 및 구도 설정을 서버로 전송하고 이미지 생성 세션을 시작합니다.
 * 서버는 세션 ID를 반환하며, 이후 WebSocket을 통해 이미지 스트림을 받을 수 있습니다.
 * 
 * @param prompt 이미지 생성 프롬프트
 * @param objects 객체 리스트 (선택적, 구도 설정이 있는 경우)
 * @param bboxes 바운딩 박스 리스트 (선택적, 구도 설정이 있는 경우)
 * @returns 세션 정보 (sessionId, websocketUrl 등)
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
