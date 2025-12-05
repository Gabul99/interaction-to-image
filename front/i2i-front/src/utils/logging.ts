import { saveLog } from "../api/logging";
import { saveSession } from "../lib/api";
import { saveSession as saveSessionPrompt } from "../api/simplePixArt";
import type { LogEntry, GraphSession } from "../types";

/**
 * 통합 로깅 및 세션 저장 함수
 * 로그와 세션을 같은 logId/timestamp로 동기화하여 저장
 * 
 * @param action - 액션 타입 (예: "prompt_node_created", "generation_started")
 * @param actionData - 액션별 데이터
 * @param participant - 참가자 번호
 * @param mode - 모드 ("step" | "prompt")
 * @param graphSession - 현재 GraphSession
 * @param bookmarkedNodeIds - 북마크된 노드 ID 배열 (선택적)
 */
export async function logActionAndSaveSession(
  action: string,
  actionData: Record<string, any>,
  participant: number,
  mode: string,
  graphSession: GraphSession,
  bookmarkedNodeIds?: string[]
): Promise<void> {
  // UUID 생성 (브라우저의 crypto.randomUUID 사용)
  const logId = crypto.randomUUID();
  
  // 타임스탬프 생성 (밀리초 단위)
  const timestamp = Date.now();
  
  // LogEntry 생성
  const logEntry: LogEntry = {
    logId,
    timestamp,
    participant,
    mode,
    sessionId: graphSession.id,
    action,
    data: actionData,
  };
  
  try {
    // 로그 저장
    await saveLog(logEntry);
    console.log(`[Logging] Log saved: ${action} (logId: ${logId})`);
    
    // 세션 저장 (같은 logId/timestamp 사용)
    // mode에 따라 적절한 API 사용
    if (mode === "prompt") {
      await saveSessionPrompt(
        mode,
        participant,
        graphSession,
        bookmarkedNodeIds || [],
        logId,
        timestamp
      );
    } else {
      await saveSession(
        mode,
        participant,
        graphSession,
        bookmarkedNodeIds || [],
        logId,
        timestamp
      );
    }
    console.log(`[Logging] Session saved with logId: ${logId}`);
  } catch (error) {
    console.error(`[Logging] Failed to save log and session:`, error);
    throw error;
  }
}

