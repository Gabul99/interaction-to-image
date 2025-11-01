import { type FeedbackData } from "../types";

// 서버와 소켓 통신을 한다고 가정
// 실제로는 WebSocket을 통해 전송하거나, REST API를 사용할 수 있습니다

/**
 * 피드백을 서버로 전송합니다.
 * @param feedback 피드백 데이터
 */
export async function sendFeedbackToServer(
  feedback: FeedbackData
): Promise<void> {
  // 실제 서버 구현 시 이 부분을 수정해야 합니다
  console.log("피드백 서버로 전송:", feedback);

  // FormData를 사용하여 파일과 함께 전송
  const formData = new FormData();
  formData.append("area", feedback.area);
  formData.append("type", feedback.type);

  if (feedback.text) {
    formData.append("text", feedback.text);
  }

  if (feedback.image) {
    formData.append("image", feedback.image);
  }

  if (feedback.point) {
    formData.append("pointX", feedback.point.x.toString());
    formData.append("pointY", feedback.point.y.toString());
  }

  if (feedback.bbox) {
    formData.append("bboxX", feedback.bbox.x.toString());
    formData.append("bboxY", feedback.bbox.y.toString());
    formData.append("bboxWidth", feedback.bbox.width.toString());
    formData.append("bboxHeight", feedback.bbox.height.toString());
  }

  try {
    // 실제 서버 endpoint로 전송
    // 예시: await fetch('/api/feedback', { method: 'POST', body: formData });

    // 현재는 시뮬레이션으로만 처리
    await new Promise((resolve) => setTimeout(resolve, 500));
    console.log("피드백 전송 완료");
  } catch (error) {
    console.error("피드백 전송 실패:", error);
    throw error;
  }
}

/**
 * 피드백 요청을 건너뛰고 서버에 알립니다.
 */
export async function skipFeedback(): Promise<void> {
  console.log("피드백 건너뛰기");

  try {
    // 실제 서버 endpoint로 전송
    // 예시: await fetch('/api/feedback/skip', { method: 'POST' });

    // 현재는 시뮬레이션으로만 처리
    await new Promise((resolve) => setTimeout(resolve, 300));
    console.log("피드백 건너뛰기 완료");
  } catch (error) {
    console.error("피드백 건너뛰기 실패:", error);
    throw error;
  }
}
