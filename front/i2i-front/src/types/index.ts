export type ToolMode = "point" | "bbox" | "none";

export interface InteractionData {
  type: "point" | "bbox";
  x: number;
  y: number;
  width?: number;
  height?: number;
}

// 피드백 관련 타입
export type FeedbackArea = "full" | "point" | "bbox";
export type FeedbackType = "text" | "image";

export interface FeedbackData {
  area: FeedbackArea;
  type: FeedbackType;
  text?: string;
  image?: File;
  point?: { x: number; y: number }; // 포인팅의 경우 좌표
  bbox?: { x: number; y: number; width: number; height: number }; // BBOX의 경우 좌표
}
