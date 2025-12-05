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
  guidanceScale?: number; // Guidance scale for text or style guidance
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
  type: 'prompt' | 'image' | 'placeholder' | 'loading';
  data: {
    prompt?: string;
    imageUrl?: string;
    step?: number;
    sessionId?: string;
    backendBranchId?: string; // Backend branch ID for this node (e.g., "B0", "B1")
    backendSessionId?: string; // Backend session ID (for prompt nodes with parallel sessions)
    uniqueBranchId?: string; // Unique branch ID combining session + branch (e.g., "session123_B0")
    mergedFrom?: [string, string]; // Source node IDs if this is a merged node
    // Composition 데이터 (루트 노드에만 저장)
    compositionData?: {
      bboxes: BoundingBox[];
      sketchLayers: SketchLayer[];
    };
    // Placeholder node 관련
    onClick?: () => void; // Placeholder node 클릭 핸들러
    // Row index for parallel prompt nodes
    rowIndex?: number;
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
    branchId?: string; // Unique branch ID (e.g., "session123_B0")
    backendBranchId?: string; // Backend branch ID (e.g., "B0")
    isMergeEdge?: boolean; // True if this edge is part of a merge
  };
}

export interface Branch {
  id: string; // Unique branch ID (e.g., "session123_B0")
  backendBranchId?: string; // Backend branch ID (e.g., "B0")
  backendSessionId?: string; // Backend session ID this branch belongs to
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

// 로깅 관련 타입
export interface LogEntry {
  logId: string; // UUID v4
  timestamp: number; // 밀리초
  participant: number;
  mode: string; // "step" | "prompt"
  sessionId: string; // GraphSession ID
  action: string; // 액션 타입
  data: Record<string, any>; // 액션별 데이터
}

// 각 액션별 로그 데이터 타입
export interface PromptNodeCreatedData {
  nodeId: string;
  prompt: string;
  compositionData?: {
    bboxes: BoundingBox[];
    sketchLayers: SketchLayer[];
  };
}

export interface CompositionConfiguredData {
  promptNodeId: string;
  objects: ObjectChip[];
  bboxes: BoundingBox[];
  sketchLayers?: SketchLayer[];
}

export interface GenerationStartedData {
  sourceNodeId: string;
  sourceNodeType: "prompt" | "image";
  sourceNodeStep?: number;
  branchId: string;
  prompt: string;
  isRegeneration: boolean;
  compositionData?: {
    bboxes: BoundingBox[];
    sketchLayers: SketchLayer[];
  };
}

export interface NextStepClickedData {
  sourceNodeId: string;
  sourceNodeStep: number;
  branchId: string;
  expectedNextStep: number;
}

export interface RunToEndStartedData {
  sourceNodeId: string;
  sourceNodeStep: number;
  branchId: string;
  currentStep: number;
  targetStep: number;
}

export interface RunToEndPausedData {
  branchId: string;
  pausedAtStep: number;
  totalStepsGenerated: number;
  duration: number; // 밀리초
}

export interface SimpleGenerateData {
  promptNodeId: string;
  prompt: string;
  imageUrl?: string;
  isRegeneration: boolean;
}

export interface ImageReceivedData {
  nodeId: string;
  branchId: string;
  step: number;
  imageUrl: string;
  generationDuration: number; // 밀리초
  sourceAction: "next_step" | "run_to_end" | "generation_started" | "simple_generate" | "branch" | "merge";
}

export interface BranchCreatedData {
  sourceNodeId: string;
  sourceNodeStep: number;
  sourceBranchId: string;
  newBranchId: string;
  feedback: {
    type: FeedbackType;
    area: FeedbackArea;
    text?: string;
    imageUrl?: string;
    point?: { x: number; y: number };
    bbox?: { x: number; y: number; width: number; height: number };
    guidanceScale?: number;
  };
  isAfterComplete: boolean;
  sourceBranchMaxStep: number;
}

export interface MergeCreatedData {
  sourceNode1Id: string;
  sourceNode1Step: number;
  sourceNode1BranchId: string;
  sourceNode2Id: string;
  sourceNode2Step: number;
  sourceNode2BranchId: string;
  newBranchId: string;
  mergeStartStep: number;
  mergeWeight: number;
}

export interface NodeDeletedData {
  nodeId: string;
  nodeType: "prompt" | "image";
  nodeStep?: number;
  branchId?: string;
  deletedNodeIds: string[];
}

export interface BacktrackData {
  targetNodeId: string;
  targetStep: number;
  branchId: string;
  backtrackToStep: number;
  removedNodeIds: string[];
}

export interface BookmarkToggledData {
  nodeId: string;
  nodeStep: number;
  branchId: string;
  isBookmarked: boolean;
  imageUrl: string;
}

export interface NodeSelectedData {
  nodeId: string | null;
  nodeType?: "prompt" | "image";
  nodeStep?: number;
  branchId?: string;
}
