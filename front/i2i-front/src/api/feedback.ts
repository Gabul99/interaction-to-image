import { type FeedbackData } from "../types";

/**
 * ============================================================================
 * 백엔드 API 스펙: 피드백 제출 (Deprecated - 브랜치 생성으로 대체됨)
 * ============================================================================
 * 
 * 이 API는 브랜치 생성 API로 대체되었습니다.
 * 새로운 코드에서는 브랜치 생성 API를 사용하세요.
 * 
 * @deprecated 브랜치 생성 API 사용 권장
 * @see createBranch API
 */
export async function submitFeedbacks(
  sessionId: string,
  feedbacks: FeedbackData[]
): Promise<void> {
  console.log("[API] 피드백 제출 요청:", {
    sessionId,
    count: feedbacks.length,
  });

  try {
    // TODO: 실제 백엔드 연결 시 아래 주석을 해제하고 mockup 코드를 제거
    /*
    // FormData를 사용하여 파일과 함께 전송
    const formData = new FormData();
    formData.append("sessionId", sessionId);
    
    // 피드백 배열을 JSON으로 직렬화
    const feedbacksJson = feedbacks.map(feedback => {
      const feedbackObj: any = {
        area: feedback.area,
        type: feedback.type,
      };
      
      if (feedback.text) feedbackObj.text = feedback.text;
      if (feedback.point) {
        feedbackObj.point = { x: feedback.point.x, y: feedback.point.y };
      }
      if (feedback.bbox) {
        feedbackObj.bbox = {
          x: feedback.bbox.x,
          y: feedback.bbox.y,
          width: feedback.bbox.width,
          height: feedback.bbox.height,
        };
      }
      
      return feedbackObj;
    });
    
    formData.append("feedbacks", JSON.stringify(feedbacksJson));
    
    // 이미지 파일들은 별도로 추가
    feedbacks.forEach((feedback, index) => {
      if (feedback.image) {
        formData.append(`image_${index}`, feedback.image);
      }
    });
    
    const response = await fetch('/api/feedback', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    console.log("[API] 피드백 제출 완료");
    */

    // ===== MOCKUP (백엔드 연결 전까지 사용) =====
    await new Promise((resolve) => setTimeout(resolve, 500));
    console.log("[API] 피드백 제출 완료 (Mockup)");
    // ===== MOCKUP END =====
  } catch (error) {
    console.error("[API] 피드백 제출 실패:", error);
    throw error;
  }
}

/**
 * API Endpoint: POST /api/feedback/skip
 *
 * 피드백 요청을 건너뛰고 서버에 알립니다.
 *
 * @param sessionId 세션 ID
 */
export async function skipFeedback(sessionId: string): Promise<void> {
  console.log("[API] 피드백 건너뛰기 요청:", sessionId);

  try {
    // TODO: 실제 백엔드 연결 시 아래 주석을 해제하고 mockup 코드를 제거
    /*
    const response = await fetch('/api/feedback/skip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    console.log("[API] 피드백 건너뛰기 완료");
    */

    // ===== MOCKUP (백엔드 연결 전까지 사용) =====
    await new Promise((resolve) => setTimeout(resolve, 300));
    console.log("[API] 피드백 건너뛰기 완료 (Mockup)");
    // ===== MOCKUP END =====
  } catch (error) {
    console.error("[API] 피드백 건너뛰기 실패:", error);
    throw error;
  }
}

/**
 * @deprecated 이 함수는 submitFeedbacks로 통합되었습니다.
 * 하위 호환성을 위해 유지되지만, 새로운 코드에서는 submitFeedbacks를 사용하세요.
 */
export async function sendFeedbackToServer(
  feedback: FeedbackData
): Promise<void> {
  console.warn(
    "[API] sendFeedbackToServer는 deprecated입니다. submitFeedbacks를 사용하세요."
  );

  // 단일 피드백을 배열로 변환하여 submitFeedbacks 호출
  // sessionId는 임시로 빈 문자열 사용 (실제로는 세션 ID가 필요)
  await submitFeedbacks("", [feedback]);
}
