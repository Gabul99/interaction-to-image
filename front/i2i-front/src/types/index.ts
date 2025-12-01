export type ToolMode = "select" | "bbox" | "sketch" | "eraser" | "point" | "none";

export interface InteractionData {
  type: "point" | "bbox" | "sketch";
  x: number;
  y: number;
  width?: number;
  height?: number;
  sketchData?: SketchLayer[]; // 스케치 데이터
}

// 스케치 레이어 데이터 구조
export interface SketchLayer {
  objectId: string;
  color: string;
  paths: Array<{ x: number; y: number }[]>; // 각 경로는 점들의 배열
}

// 피드백 관련 타입
export type FeedbackArea = "full" | "point" | "bbox" | "sketch";
export type FeedbackType = "text" | "image";

export interface FeedbackData {
  area: FeedbackArea;
  type: FeedbackType;
  text?: string;
  image?: File;
  point?: { x: number; y: number }; // 포인팅의 경우 좌표
  bbox?: { x: number; y: number; width: number; height: number }; // BBOX의 경우 좌표
}

export interface FeedbackRecord {
  id: string;
  area: FeedbackArea;
  type: FeedbackType;
  text?: string;
  imageUrl?: string; // 이미지 피드백의 경우 URL
  point?: { x: number; y: number };
  bbox?: { x: number; y: number; width: number; height: number };
  bboxId?: string; // BBOX의 경우 ID
  timestamp: number;
}

// 객체 및 구도 설정 관련 타입
export interface ObjectChip {
  id: string;
  label: string;
  color: string; // hex color
}

export interface BoundingBox {
  id: string;
  objectId: string; // 연결된 객체의 id
  x: number; // 상대 좌표 (0~1)
  y: number; // 상대 좌표 (0~1)
  width: number; // 상대 크기 (0~1)
  height: number; // 상대 크기 (0~1)
  color: string; // 객체의 색상
}

export interface CompositionState {
  objects: ObjectChip[];
  bboxes: BoundingBox[];
  selectedObjectId: string | null;
  isConfigured: boolean;
}

// 그래프 관련 타입
export interface GraphNode {
  id: string;
  type: 'prompt' | 'image';
  data: {
    prompt?: string;
    imageUrl?: string;
    step?: number;
    sessionId?: string;
    backendBranchId?: string; // Backend branch ID for this node
    mergedFrom?: [string, string]; // Source node IDs if this is a merged node
    // Composition 데이터 (루트 노드에만 저장)
    compositionData?: {
      bboxes: BoundingBox[];
      sketchLayers: SketchLayer[];
    };
  };
  position: { x: number; y: number };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type?: 'default' | 'branch';
  data?: {
    feedback?: FeedbackRecord[];
    branchId?: string;
    isMergeEdge?: boolean; // True if this edge is part of a merge
  };
}

export interface Branch {
  id: string;
  sourceNodeId: string; // 브랜치가 시작된 노드
  feedback: FeedbackRecord[];
  nodes: string[]; // 이 브랜치에 속한 노드 ID들
}

export interface GraphSession {
  id: string; // Session ID와 동일
  nodes: GraphNode[];
  edges: GraphEdge[];
  branches: Branch[];
}
