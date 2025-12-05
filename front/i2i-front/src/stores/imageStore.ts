import { create } from "zustand";
import {
  type FeedbackArea,
  type ObjectChip,
  type BoundingBox,
  type CompositionState,
  type FeedbackRecord,
  type GraphSession,
  type GraphNode,
  type GraphEdge,
  type Branch,
  type SketchLayer,
} from "../types";
import { connectImageStream, disconnectImageStream } from "../api/websocket";
import { USE_MOCK_MODE } from "../config/api";
import { saveSession as saveSessionStep, loadSession as loadSessionStep } from "../lib/api";
import { saveSession as saveSessionPrompt, loadSession as loadSessionPrompt } from "../api/simplePixArt";

// Grid layout constants (must match GraphCanvas.tsx)
const GRID_CELL_WIDTH = 80;
const GRID_CELL_HEIGHT = 280;
const GRID_START_X = 100;
const GRID_START_Y = 50;

/**
 * Create a unique branch ID by combining backend session ID and backend branch ID.
 * This ensures branch IDs are unique across parallel sessions.
 * @param backendSessionId - Backend session ID
 * @param backendBranchId - Backend branch ID (e.g., "B0", "B1")
 * @returns Unique branch ID (e.g., "sess_abc123_B0")
 */
export const createUniqueBranchId = (
  backendSessionId: string,
  backendBranchId: string
): string => {
  // Use a short prefix of the session ID to keep IDs readable
  const shortSessionId = backendSessionId.slice(0, 8);
  return `sess_${shortSessionId}_${backendBranchId}`;
};

/**
 * Extract the backend branch ID from a unique branch ID.
 * @param uniqueBranchId - Unique branch ID (e.g., "sess_abc123_B0")
 * @returns Backend branch ID (e.g., "B0") or the original ID if not in unique format
 */
export const extractBackendBranchId = (uniqueBranchId: string): string => {
  const match = uniqueBranchId.match(/^sess_[a-zA-Z0-9]+_(B\d+)$/);
  return match ? match[1] : uniqueBranchId;
};

/**
 * Check if a branch ID is in the unique format (contains session prefix).
 * @param branchId - Branch ID to check
 * @returns True if the ID is in unique format
 */
export const isUniqueBranchId = (branchId: string): boolean => {
  return branchId.startsWith("sess_");
};

/**
 * Calculate branch row index within its own session
 * Each session has independent row space - branches from different sessions don't affect each other
 * 
 * For a session with base row R (from prompt node's rowIndex):
 * - Main branch (B0) is at row R
 * - Non-main branches (B1, B2, ...) are at rows R+1, R+2, ...
 * 
 * @param branchId - Branch ID (unique format like "sess_xxx_B0" or backend format like "B0")
 * @param branches - Array of all branches in the graph session
 * @param sessionBaseRow - Optional base row for the session (from prompt node's rowIndex)
 * @returns Row index
 */
export const getBranchRowIndex = (
  branchId: string,
  branches: Branch[],
  sessionBaseRow?: number
): number => {
  // Extract backend branch ID from unique format
  const backendBranchId = extractBackendBranchId(branchId);
  
  // Find the branch to get its session ID
  const branch = branches.find((b) => b.id === branchId);
  const backendSessionId = branch?.backendSessionId;
  
  // Determine the base row for this session
  // If sessionBaseRow is provided, use it; otherwise, calculate from branches
  let baseRow = sessionBaseRow ?? 0;
  
  if (sessionBaseRow === undefined && backendSessionId) {
    // Find the main branch (B0) for this session to get its base row
    const sessionMainBranch = branches.find(
      (b) => b.backendSessionId === backendSessionId && extractBackendBranchId(b.id) === "B0"
    );
    if (sessionMainBranch) {
      // Count how many sessions' main branches come before this one
      const mainBranches = branches.filter((b) => extractBackendBranchId(b.id) === "B0");
      const mainIndex = mainBranches.findIndex((b) => b.id === sessionMainBranch.id);
      baseRow = mainIndex >= 0 ? mainIndex : 0;
    }
  }
  
  // If this is the main branch (B0), return the base row
  if (backendBranchId === "B0") {
    return baseRow;
  }

  // For non-main branches, calculate offset within THIS session only
  // Get all non-main branches for this session
  const sessionNonMainBranches = branches
    .filter((b) => {
      if (extractBackendBranchId(b.id) === "B0") return false;
      // Only include branches from the same session
      return b.backendSessionId === backendSessionId;
    })
    .sort((a, b) => {
      // Sort by branch number (B1, B2, B3...)
      const getNum = (id: string): number => {
        const backendId = extractBackendBranchId(id);
        const match = backendId.match(/^B(\d+)$/);
        return match ? parseInt(match[1], 10) : 0;
      };
      return getNum(a.id) - getNum(b.id);
    });

  // Find the index of the target branch within this session's non-main branches
  const index = sessionNonMainBranches.findIndex((b) => b.id === branchId);
  
  // Return base row + 1 (for main branch) + index within session's non-main branches
  return baseRow + 1 + (index >= 0 ? index : 0);
};

/**
 * Helper to get the session base row from nodes (prompt node's rowIndex)
 * @param backendSessionId - The backend session ID
 * @param nodes - Array of all nodes in the graph session
 * @returns The base row for the session, or undefined if not found
 */
export const getSessionBaseRow = (
  backendSessionId: string | undefined,
  nodes: GraphNode[]
): number | undefined => {
  if (!backendSessionId) return undefined;
  
  const promptNode = nodes.find(
    (n) => n.type === "prompt" && n.data?.backendSessionId === backendSessionId
  );
  
  if (promptNode?.data?.rowIndex !== undefined) {
    return promptNode.data.rowIndex as number;
  }
  
  return undefined;
};

export interface ImageStep {
  id: string; // 서버에서 보내준 UUID
  url: string;
  step: number;
  timestamp: number;
  // 그래프 구조를 위한 추가 정보 (선택적)
  nodeId?: string;
  parentNodeId?: string;
  sessionId?: string;
  branchId?: string;
}

export interface ImageSession {
  id: string; // Session ID
  prompt: string;
  totalSteps: number;
  steps: ImageStep[]; // 각 스텝의 이미지들
  isComplete: boolean;
  createdAt: number;
  compositionBboxes?: Array<{
    id: string;
    objectId: string;
    x: number;
    y: number;
    width: number;
    height: number;
    color: string;
  }>; // 구도 설정 시 사용된 BBOX들
  bboxFeedbackHistory?: Record<string, FeedbackRecord[]>; // BBOX별 피드백 히스토리 (bboxId -> FeedbackRecord[])
}

export interface ImageStreamState {
  // 현재 활성 세션
  currentSession: ImageSession | null;

  // 현재 그래프 세션
  currentGraphSession: GraphSession | null;

  // 현재 선택된 노드 ID (그래프에서)
  selectedNodeId: string | null;

  // 현재 hover된 피드백 엣지 (BBOX 표시용)
  hoveredFeedbackEdge: {
    branchId: string;
    bboxFeedbacks: FeedbackRecord[]; // area가 "bbox"인 피드백들만
  } | null;

  // 현재 선택된 스텝 (History에서 클릭한 스텝)
  selectedStepIndex: number | null; // null이면 최신 스텝 표시

  // 생성 상태
  isGenerating: boolean;
  isPaused: boolean; // 일시정지 상태

  // Socket 연결 상태
  isConnected: boolean;
  websocket: WebSocket | null; // 실제 WebSocket 연결

  // 이미지 생성 간격 설정
  generationInterval: number; // 1, 5, 10, 20 스텝 단위

  // 피드백 요청 상태
  feedbackRequest: {
    visible: boolean;
    area: FeedbackArea;
  } | null;

  // 현재 턴의 피드백 리스트
  currentFeedbackList: FeedbackRecord[];

  // 구도 설정 상태
  compositionState: CompositionState;

  // Actions
  startGeneration: (
    prompt: string,
    sessionId: string,
    websocketUrl: string,
    interval?: number
  ) => void;
  setGenerationInterval: (interval: number) => void;
  addImageStep: (sessionId: string, imageStep: ImageStep) => void;
  completeSession: (sessionId: string) => void;
  stopGeneration: () => void;
  pauseGeneration: () => void; // 생성 일시정지
  resumeGeneration: () => void; // 생성 재개
  selectStep: (stepIndex: number | null) => void; // 특정 스텝 선택 또는 최신으로 돌아가기
  resetSession: () => void; // 세션 초기화

  // 피드백 관련 액션
  showFeedbackRequest: (area: FeedbackArea) => void;
  hideFeedbackRequest: () => void;
  addFeedbackToCurrentList: (feedback: FeedbackRecord) => void;
  removeFeedbackFromCurrentList: (feedbackId: string) => void;
  clearCurrentFeedbackList: () => void;

  // 구도 설정 관련 액션
  setObjectList: (objects: ObjectChip[]) => void;
  addObject: (label: string) => void;
  removeObject: (objectId: string) => void;
  selectObject: (objectId: string | null) => void;
  addBbox: (bbox: Omit<BoundingBox, "id">) => void;
  updateBbox: (bboxId: string, updates: Partial<BoundingBox>) => void;
  removeBbox: (bboxId: string) => void;
  clearComposition: () => void;
  startGenerationWithComposition: (
    prompt: string,
    sessionId: string,
    websocketUrl: string,
    bboxes?: BoundingBox[],
    sketchLayers?: SketchLayer[],
    interval?: number
  ) => void;

  // 현재 세션의 구도 BBOX 가져오기
  getCurrentCompositionBboxes: () => Array<{
    id: string;
    objectId: string;
    x: number;
    y: number;
    width: number;
    height: number;
    color: string;
  }> | null;

  // 피드백 기록 관련 (BBOX별)
  addFeedbackToBboxHistory: (bboxId: string, feedback: FeedbackRecord) => void;
  getFeedbackHistoryForBbox: (bboxId: string) => FeedbackRecord[];

  // Socket 시뮬레이션
  simulateImageStream: (
    sessionId: string,
    prompt: string,
    interval?: number
  ) => void;

  // Backend session meta (REST flow)
  backendSessionId: string | null;
  backendActiveBranchId: string | null;
  setBackendSessionMeta: (sessionId: string, activeBranchId: string) => void;
  // Create a branch in the graph with unique ID
  // backendBranchId: The branch ID from the backend (e.g., "B1")
  // backendSessionId: The backend session ID this branch belongs to
  createBranchInGraph: (graphSessionId: string, backendBranchId: string, sourceNodeId: string, backendSessionId: string, feedback?: FeedbackRecord[]) => string; // Returns the unique branch ID
  createMergedBranchWithNode: (
    graphSessionId: string,
    backendBranchId: string, // Backend branch ID (e.g., "B3")
    backendSessionId: string, // Backend session ID
    sourceNodeId1: string, // First source node (e.g., from branch 1)
    sourceNodeId2: string, // Second source node (e.g., from branch 2)
    imageUrl: string,
    step: number,
    position: { x: number; y: number },
    placeholderNodeId?: string // Optional: placeholder node ID to convert
  ) => string; // Returns the new node ID

  // 그래프 세션 관련 액션
  initializeEmptyGraphSession: () => void; // 빈 GraphSession 초기화
  createGraphSession: (
    prompt: string,
    sessionId: string,
    rootNodeId?: string,
    bboxes?: BoundingBox[],
    sketchLayers?: SketchLayer[]
  ) => string;
  // 세션 저장/로드
  saveSessionToServer: (mode: string, participant: number) => Promise<void>;
  loadSessionFromServer: (mode: string, participant: number) => Promise<void>;
  addPlaceholderNode: (
    sessionId: string,
    position: { x: number; y: number },
    onClick?: () => void
  ) => string; // nodeId 반환
  addImageNode: (
    sessionId: string,
    parentNodeId: string,
    imageUrl: string,
    step: number,
    position: { x: number; y: number },
    nodeId?: string,
    explicitBranchId?: string // Optional: explicit unique branch ID for non-main branches
  ) => string; // nodeId 반환
  addImageNodeToBranch: (
    sessionId: string,
    branchId: string,
    imageUrl: string,
    step: number,
    position?: { x: number; y: number },
    nodeId?: string
  ) => string; // nodeId 반환
  addLoadingNode: (
    sessionId: string,
    parentNodeId: string,
    step: number,
    position: { x: number; y: number },
    branchId?: string
  ) => string; // nodeId 반환
  addLoadingNodeToBranch: (
    sessionId: string,
    branchId: string,
    step: number,
    position?: { x: number; y: number }
  ) => string; // nodeId 반환
  removeLoadingNode: (sessionId: string, nodeId: string) => void;
  updateNodePosition: (
    sessionId: string,
    nodeId: string,
    position: { x: number; y: number }
  ) => void;
  updatePlaceholderNodePrompt: (
    sessionId: string,
    nodeId: string,
    prompt: string
  ) => void;
  updatePromptNodePrompt: (
    sessionId: string,
    nodeId: string,
    prompt: string
  ) => void;
  addEmptyPromptNode: (
    sessionId: string,
    position: { x: number; y: number }
  ) => string; // nodeId 반환
  removeImageNodesAfterPrompt: (
    sessionId: string,
    promptNodeId: string
  ) => void; // PromptNode 뒤의 모든 이미지 노드 제거
  addEdge: (
    sessionId: string,
    source: string,
    target: string,
    edgeData?: GraphEdge["data"]
  ) => string; // edgeId 반환
  selectNode: (nodeId: string | null) => void;
  setHoveredFeedbackEdge: (branchId: string | null, bboxFeedbacks?: FeedbackRecord[]) => void;
  simulateGraphImageStream: (
    sessionId: string,
    prompt: string,
    rootNodeId: string,
    interval?: number
  ) => void;
  simulateBranchImageStream: (
    sessionId: string,
    branchId: string,
    interval?: number
  ) => void;
  // 현재 노드에서 루트까지 역방향으로 올라가며 브랜치 피드백 수집
  getBranchFeedbacksForNode: (nodeId: string) => FeedbackRecord[];
  
  // Remove a node and all its descendants (for backtracking)
  removeNodeAndDescendants: (sessionId: string, nodeId: string) => void;
  removePromptNodeAndBranch: (sessionId: string, promptNodeId: string) => void; // 프롬프트 노드와 연결된 모든 노드 및 브랜치 제거
  
  // Parallel session support
  // Map of prompt node ID -> backend session info
  parallelSessions: Map<string, { backendSessionId: string; backendBranchId: string }>;
  
  // Add a new prompt node for parallel session
  addPromptNodeToGraph: (
    prompt: string,
    backendSessionId: string,
    backendBranchId: string,
    bboxes?: BoundingBox[],
    sketchLayers?: SketchLayer[],
    position?: { x: number; y: number },
    placeholderNodeId?: string // Optional: convert placeholder to prompt
  ) => string | null; // Returns the new prompt node ID
  
  // Get backend session info for a node (traces back to its prompt node)
  getBackendSessionForNode: (nodeId: string) => { sessionId: string; branchId: string } | null;
  
  // Register a parallel session for a prompt node
  registerParallelSession: (promptNodeId: string, backendSessionId: string, backendBranchId: string) => void;
  
  // 북마크 관련
  bookmarkedNodeIds: string[];
  toggleBookmark: (nodeId: string) => void;
  isBookmarked: (nodeId: string) => boolean;
}

export const useImageStore = create<ImageStreamState>((set, get) => ({
  // 초기 상태
  currentSession: null,
  currentGraphSession: null,
  selectedNodeId: null,
  hoveredFeedbackEdge: null,
  selectedStepIndex: null,
  isGenerating: false,
  isPaused: false,
  isConnected: false,
  websocket: null,
  generationInterval: 1, // 기본값: 매 스텝마다
  feedbackRequest: null,
  currentFeedbackList: [],
  compositionState: {
    objects: [],
    bboxes: [],
    selectedObjectId: null,
    isConfigured: false,
  },
  backendSessionId: null,
  backendActiveBranchId: null,
  
  // Parallel sessions map: promptNodeId -> { backendSessionId, backendBranchId }
  parallelSessions: new Map(),
  
  // 북마크 관련 초기 상태
  bookmarkedNodeIds: [],
  
  setBackendSessionMeta: (sessionId: string, activeBranchId: string) => {
    set({ backendSessionId: sessionId, backendActiveBranchId: activeBranchId });
  },
  createBranchInGraph: (graphSessionId: string, backendBranchId: string, sourceNodeId: string, backendSessionId: string, feedback?: FeedbackRecord[]): string => {
    const state = get();
    if (!state.currentGraphSession || state.currentGraphSession.id !== graphSessionId) return "";
    
    // Create unique branch ID combining session and backend branch ID
    const uniqueBranchId = createUniqueBranchId(backendSessionId, backendBranchId);
    
    const exists = state.currentGraphSession.branches.find((b) => b.id === uniqueBranchId);
    if (exists) {
      // Update existing branch with feedback if provided
      if (feedback && feedback.length > 0) {
        set({
          currentGraphSession: {
            ...state.currentGraphSession,
            branches: state.currentGraphSession.branches.map((b) =>
              b.id === uniqueBranchId ? { ...b, feedback: [...(b.feedback || []), ...feedback] } : b
            ),
          },
        });
      }
      return uniqueBranchId;
    }
    
    // Debug: Log branch creation
    console.log(`[ImageStore] Creating branch: unique=${uniqueBranchId}, backend=${backendBranchId}, session=${backendSessionId}`);
    console.log(`[ImageStore] Branches BEFORE:`, state.currentGraphSession.branches.map(b => b.id).join(", "));
    
    const newBranch: Branch = {
      id: uniqueBranchId,
      backendBranchId,
      backendSessionId,
      sourceNodeId,
      feedback: feedback || [],
      nodes: [],
    };
    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        branches: [...state.currentGraphSession.branches, newBranch],
      },
      backendActiveBranchId: backendBranchId,
    });
    
    // Debug: Log after creation
    const updatedState = get();
    console.log(`[ImageStore] Branches AFTER:`, updatedState.currentGraphSession!.branches.map(b => b.id).join(", "));
    
    return uniqueBranchId;
  },

  // Create a merged branch and its initial node atomically
  // Connects to both source nodes to visually show the merge
  createMergedBranchWithNode: (
    graphSessionId: string,
    backendBranchId: string, // Backend branch ID (e.g., "B3")
    backendSessionId: string, // Backend session ID
    sourceNodeId1: string, // First source node (e.g., from branch 1)
    sourceNodeId2: string, // Second source node (e.g., from branch 2)
    imageUrl: string,
    step: number,
    position: { x: number; y: number },
    placeholderNodeId?: string // Optional: placeholder node ID to convert
  ): string => {
    const state = get();
    if (!state.currentGraphSession || state.currentGraphSession.id !== graphSessionId) {
      console.warn("[ImageStore] Cannot create merged branch: session not found");
      return "";
    }
    
    // Create unique branch ID
    const uniqueBranchId = createUniqueBranchId(backendSessionId, backendBranchId);
    
    // Check if branch already exists
    const exists = state.currentGraphSession.branches.find((b) => b.id === uniqueBranchId);
    if (exists) {
      console.warn("[ImageStore] Branch already exists:", uniqueBranchId);
      return "";
    }

    // Placeholder node가 있으면 그 node를 변환, 없으면 새 node 생성
    let nodeId: string;
    let finalPosition = position;
    
    if (placeholderNodeId) {
      const placeholderNode = state.currentGraphSession.nodes.find(
        (n) => n.id === placeholderNodeId && n.type === "placeholder"
      );
      if (placeholderNode) {
        nodeId = placeholderNodeId; // 기존 placeholder node ID 사용
        finalPosition = placeholderNode.position; // 기존 위치 사용
        console.log(`[ImageStore] Converting placeholder node ${nodeId} to merge result node`);
      } else {
        // Placeholder node를 찾을 수 없으면 새로 생성
        nodeId = `node_merged_${uniqueBranchId}_${step}_${Date.now()}`;
        console.warn(`[ImageStore] Placeholder node ${placeholderNodeId} not found, creating new node`);
      }
    } else {
      // Generate node ID
      nodeId = `node_merged_${uniqueBranchId}_${step}_${Date.now()}`;
    }
    
    // Find the source nodes to determine which has the larger step
    const node1 = state.currentGraphSession.nodes.find((n) => n.id === sourceNodeId1);
    const node2 = state.currentGraphSession.nodes.find((n) => n.id === sourceNodeId2);
    const step1 = node1?.data?.step ?? 0;
    const step2 = node2?.data?.step ?? 0;
    
    // The source node for the merged branch should be the one with the larger step
    // This ensures the merged branch's first node is positioned at the same column as that node
    const primarySourceNodeId = step1 >= step2 ? sourceNodeId1 : sourceNodeId2;
    const secondarySourceNodeId = step1 >= step2 ? sourceNodeId2 : sourceNodeId1;
    
    // Find PARENT nodes of the source nodes (for edge connections)
    const sourceEdge1 = state.currentGraphSession.edges.find(
      (e) => e.target === sourceNodeId1
    );
    const sourceEdge2 = state.currentGraphSession.edges.find(
      (e) => e.target === sourceNodeId2
    );
    
    // Use parent nodes for edge connections
    const parentNodeId1 = sourceEdge1 ? sourceEdge1.source : sourceNodeId1;
    const parentNodeId2 = sourceEdge2 ? sourceEdge2.source : sourceNodeId2;
    
    // Create the new branch with unique ID
    // sourceNodeId is set to the node with the larger step for correct column positioning
    const newBranch: Branch = {
      id: uniqueBranchId,
      backendBranchId,
      backendSessionId,
      sourceNodeId: primarySourceNodeId, // Use the node with larger step for column calculation
      feedback: [],
      nodes: [nodeId], // Include the new node
    };
    
    console.log(`[ImageStore] Merge: primarySource=${primarySourceNodeId}@${Math.max(step1, step2)}, secondarySource=${secondarySourceNodeId}@${Math.min(step1, step2)}`);

    // Create the new node with merge metadata
    const newNode: GraphNode = {
      id: nodeId,
      type: "image",
      data: { 
        imageUrl, 
        step, 
        sessionId: graphSessionId, 
        backendBranchId, // Backend branch ID (e.g., "B3")
        backendSessionId, // Backend session ID
        uniqueBranchId, // Unique branch ID (e.g., "sess_abc123_B3")
        // Store merge info for potential UI display
        mergedFrom: [sourceNodeId1, sourceNodeId2],
      },
      position,
    };

    // Create edges connecting PARENT nodes (previous step) to the merged node
    const edge1Id = `edge_merge_${parentNodeId1}_${nodeId}`;
    const edge1: GraphEdge = {
      id: edge1Id,
      source: parentNodeId1,
      target: nodeId,
      type: "branch",
      data: { branchId: uniqueBranchId, backendBranchId, isMergeEdge: true },
    };

    // Only create second edge if parent nodes are different
    const edges: GraphEdge[] = [edge1];
    if (parentNodeId1 !== parentNodeId2) {
      const edge2Id = `edge_merge_${parentNodeId2}_${nodeId}`;
      const edge2: GraphEdge = {
        id: edge2Id,
        source: parentNodeId2,
        target: nodeId,
        type: "branch",
        data: { branchId: uniqueBranchId, backendBranchId, isMergeEdge: true },
      };
      edges.push(edge2);
    }

    console.log(`[ImageStore] Creating merged branch: unique=${uniqueBranchId}, backend=${backendBranchId}, node=${nodeId}`);
    console.log(`[ImageStore] Merge edges: ${parentNodeId1} -> ${nodeId}, ${parentNodeId2} -> ${nodeId}`);

    // Update state atomically
    // Placeholder node가 있으면 변환, 없으면 추가
    const updatedNodes = placeholderNodeId
      ? state.currentGraphSession.nodes.map((n) =>
          n.id === placeholderNodeId ? newNode : n
        )
      : [...state.currentGraphSession.nodes, newNode];
    
    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: updatedNodes,
        edges: [...state.currentGraphSession.edges, ...edges],
        branches: [...state.currentGraphSession.branches, newBranch],
      },
      backendActiveBranchId: backendBranchId, // Set backend active branch ID
    });

    console.log(`[ImageStore] Merged branch created successfully: unique=${uniqueBranchId}, backend=${backendBranchId}, node: ${nodeId}`);
    return nodeId;
  },

  // 이미지 생성 시작
  startGeneration: (
    prompt: string,
    sessionId: string,
    websocketUrl: string,
    interval?: number
  ) => {
    const selectedInterval = interval || get().generationInterval;

    const newSession: ImageSession = {
      id: sessionId,
      prompt,
      totalSteps: 50, // 실제 추론 스텝 수
      steps: [],
      isComplete: false,
      createdAt: Date.now(),
    };

    set({
      currentSession: newSession,
      selectedStepIndex: null, // 새 세션 시작 시 선택 초기화
      isGenerating: true,
      isPaused: false,
      isConnected: false, // WebSocket 연결 전
      generationInterval: selectedInterval,
    });

    // Mock 모드 체크
    if (USE_MOCK_MODE) {
      console.log("[ImageStore] Mock 모드: 시뮬레이션 시작");
      // 그래프 세션이 있으면 그래프 시뮬레이션, 없으면 일반 시뮬레이션
      const state = get();
      if (
        state.currentGraphSession &&
        state.currentGraphSession.id === sessionId
      ) {
        const rootNodeId = state.currentGraphSession.nodes.find(
          (n) => n.type === "prompt"
        )?.id;
        if (rootNodeId) {
          get().simulateGraphImageStream(
            sessionId,
            prompt,
            rootNodeId,
            selectedInterval
          );
        }
      } else {
        get().simulateImageStream(sessionId, prompt, selectedInterval);
      }
      return;
    }

    // 실제 WebSocket 연결
    console.log("[ImageStore] WebSocket 연결 시작:", {
      sessionId,
      websocketUrl,
    });
    const ws = connectImageStream(
      sessionId,
      websocketUrl,
      (imageStep) => {
        // 이미지 스텝 수신
        console.log("[ImageStore] 이미지 스텝 수신:", imageStep.step);

        // ImageSession에 추가 (기존 방식)
        get().addImageStep(sessionId, imageStep);

        // GraphSession에도 추가 (그래프 구조)
        const state = get();
        if (
          state.currentGraphSession &&
          state.currentGraphSession.id === sessionId
        ) {
          // 백엔드에서 보낸 nodeId, parentNodeId 사용
          const nodeId = imageStep.nodeId || imageStep.id;
          const parentNodeId = imageStep.parentNodeId;
          const step = imageStep.step || 0;
          const branchId = imageStep.branchId;

          console.log(
            `[ImageStore] 그래프 노드 추가 시도: nodeId=${nodeId}, parentNodeId=${parentNodeId}, step=${step}, branchId=${branchId}`
          );

          if (!parentNodeId) {
            console.warn(
              "[ImageStore] parentNodeId가 없습니다. 그래프 노드 추가 건너뜀"
            );
            return;
          }

          // 부모 노드가 존재하는지 확인
          const parentNode = state.currentGraphSession.nodes.find(
            (n) => n.id === parentNodeId
          );
          if (!parentNode) {
            console.warn(
              `[ImageStore] 부모 노드를 찾을 수 없습니다: ${parentNodeId}`
            );
            console.log(
              `[ImageStore] 현재 노드 목록:`,
              state.currentGraphSession.nodes.map((n) => n.id)
            );
            return;
          }

          // 위치 계산: 항상 오른쪽으로 grow (가로 배치)
          const horizontalSpacing = 400; // 노드 너비(300px) + 간격(100px)
          let position: { x: number; y: number };

          if (!branchId) {
            // 메인 브랜치: 마지막 노드의 오른쪽에 배치
            const mainBranch = state.currentGraphSession.branches.find(
              (b) => !b.sourceNodeId
            );
            if (mainBranch) {
              const mainBranchNodes = state.currentGraphSession.nodes.filter(
                (n) => mainBranch.nodes.includes(n.id)
              );
              // step 순서대로 정렬
              const sortedNodes = mainBranchNodes.sort(
                (a, b) => (a.data.step || 0) - (b.data.step || 0)
              );
              const lastNode = sortedNodes[sortedNodes.length - 1];

              if (lastNode && lastNode.id !== parentNodeId) {
                // 마지막 노드의 오른쪽에 배치
                position = {
                  x: lastNode.position.x + horizontalSpacing,
                  y: lastNode.position.y, // 같은 y 좌표
                };
              } else {
                // 첫 번째 노드 (프롬프트 노드 바로 오른쪽)
                position = {
                  x: parentNode.position.x + horizontalSpacing,
                  y: parentNode.position.y,
                };
              }
            } else {
              // 메인 브랜치가 없으면 부모 노드의 오른쪽에 배치
              position = {
                x: parentNode.position.x + horizontalSpacing,
                y: parentNode.position.y,
              };
            }
          } else {
            // 브랜치: 마지막 노드의 오른쪽에 배치
            const branch = state.currentGraphSession.branches.find(
              (b) => b.id === branchId
            );
            if (branch) {
              const branchNodes = state.currentGraphSession.nodes.filter((n) =>
                branch.nodes.includes(n.id)
              );
              const sortedNodes = branchNodes.sort(
                (a, b) => (a.data.step || 0) - (b.data.step || 0)
              );
              const lastNode = sortedNodes[sortedNodes.length - 1];

              if (lastNode) {
                // 마지막 노드의 오른쪽에 배치
                position = {
                  x: lastNode.position.x + horizontalSpacing,
                  y: lastNode.position.y,
                };
              } else {
                // 첫 번째 노드: 소스 노드의 오른쪽에 배치
                position = {
                  x: parentNode.position.x + horizontalSpacing,
                  y: parentNode.position.y,
                };
              }
            } else {
              // 브랜치를 찾을 수 없으면 부모 노드의 오른쪽에 배치
              position = {
                x: parentNode.position.x + horizontalSpacing,
                y: parentNode.position.y,
              };
            }
          }

          if (branchId) {
            // 브랜치 노드 추가 (백엔드에서 보낸 nodeId 사용)
            console.log(
              `[ImageStore] 브랜치 노드 추가: nodeId=${nodeId}, branchId=${branchId}, imageUrl 길이=${imageStep.url.length}, step=${step}`
            );
            get().addImageNodeToBranch(
              sessionId,
              branchId,
              imageStep.url, // imageUrl 전달
              step,
              position,
              nodeId // 백엔드에서 보낸 nodeId 사용
            );
          } else {
            // 메인 브랜치 찾기
            const mainBranch = state.currentGraphSession.branches.find(
              (b) => !b.sourceNodeId
            );
            if (mainBranch) {
              // 백엔드에서 보낸 nodeId를 사용하여 메인 브랜치에 노드 추가
              get().addImageNodeToBranch(
                sessionId,
                mainBranch.id,
                imageStep.url,
                step,
                position,
                nodeId // 백엔드에서 보낸 nodeId 사용
              );
            } else {
              // 일반 노드 추가 (백엔드에서 보낸 nodeId 사용)
              get().addImageNode(
                sessionId,
                parentNodeId,
                imageStep.url,
                step,
                position,
                nodeId // 백엔드에서 보낸 nodeId 사용
              );
            }
          }
        }
      },
      (error) => {
        // 에러 발생
        console.error("[ImageStore] WebSocket 에러:", error);
        set({ isGenerating: false, isConnected: false, websocket: null });
      },
      () => {
        // 완료
        console.log("[ImageStore] 이미지 생성 완료");
        get().completeSession(sessionId);
        set({ isConnected: false, websocket: null });
      }
    );

    if (ws) {
      // WebSocket 연결 상태 확인
      ws.addEventListener("open", () => {
        console.log("[ImageStore] WebSocket 연결됨");
        set({ isConnected: true });
      });

      ws.addEventListener("close", () => {
        console.log("[ImageStore] WebSocket 연결 종료");
        set({ isConnected: false });
      });

      ws.addEventListener("error", (error) => {
        console.error("[ImageStore] WebSocket 에러 이벤트:", error);
        set({ isConnected: false });
      });
    }

    set({
      websocket: ws,
      isConnected: ws !== null && ws.readyState === WebSocket.OPEN,
    });
  },

  // 생성 간격 설정
  setGenerationInterval: (interval: number) => {
    set({ generationInterval: interval });
  },

  // 새로운 이미지 스텝 추가
  addImageStep: (sessionId: string, imageStep: ImageStep) => {
    set((state) => {
      if (!state.currentSession || state.currentSession.id !== sessionId) {
        return state;
      }

      return {
        currentSession: {
          ...state.currentSession,
          steps: [...state.currentSession.steps, imageStep],
        },
        // 사용자가 특정 스텝을 선택한 경우 그 상태를 유지
        // selectedStepIndex는 변경하지 않음
      };
    });
  },

  // 세션 완료
  completeSession: (sessionId: string) => {
    set((state) => {
      if (!state.currentSession || state.currentSession.id !== sessionId) {
        return state;
      }

      return {
        currentSession: {
          ...state.currentSession,
          isComplete: true,
        },
        isGenerating: false,
        isPaused: false,
      };
    });
  },

  // 생성 중단
  stopGeneration: () => {
    const state = get();
    // WebSocket 연결 종료
    if (state.websocket) {
      disconnectImageStream(state.websocket);
    }
    set({
      isGenerating: false,
      isPaused: false,
      isConnected: false,
      websocket: null,
    });
  },

  // 세션 초기화 (새로운 이미지 생성을 위해)
  resetSession: () => {
    const state = get();
    // WebSocket 연결 종료
    if (state.websocket) {
      disconnectImageStream(state.websocket);
    }
    set({
      currentSession: null,
      selectedStepIndex: null,
      isGenerating: false,
      isPaused: false,
      isConnected: false,
      websocket: null,
    });
  },

  // 생성 일시정지
  pauseGeneration: () => {
    set({ isPaused: true });
  },

  // 생성 재개
  resumeGeneration: () => {
    const state = get();
    if (state.currentSession && !state.currentSession.isComplete) {
      set({ isPaused: false });
      // WebSocket이 이미 연결되어 있으면 자동으로 계속 진행됨
      // 시뮬레이션은 더 이상 사용하지 않음
    }
  },

  // 특정 스텝 선택 또는 최신으로 돌아가기
  selectStep: (stepIndex: number | null) => {
    const state = get();
    console.log("selectStep 호출:", {
      stepIndex,
      currentSession: state.currentSession?.id,
      totalSteps: state.currentSession?.steps.length,
      isValidStep:
        stepIndex !== null &&
        stepIndex >= 0 &&
        stepIndex < (state.currentSession?.steps.length || 0),
      targetImageUrl:
        stepIndex !== null ? state.currentSession?.steps[stepIndex]?.url : null,
    });
    set({ selectedStepIndex: stepIndex });
  },

  // 피드백 요청 표시
  showFeedbackRequest: (area: FeedbackArea) => {
    set({
      feedbackRequest: {
        visible: true,
        area,
      },
    });
  },

  // 피드백 요청 숨기기
  hideFeedbackRequest: () => {
    const state = get();
    set({ feedbackRequest: null });

    // 피드백이 닫히면 생성이 계속됨 (WebSocket이 이미 연결되어 있음)
    if (
      state.currentSession &&
      !state.currentSession.isComplete &&
      state.isGenerating
    ) {
      console.log("피드백 처리 완료, 생성 계속 진행");
      // WebSocket이 이미 연결되어 있으면 자동으로 계속 진행됨
    }
  },

  // 현재 턴의 피드백 리스트에 추가
  addFeedbackToCurrentList: (feedback: FeedbackRecord) => {
    set((state) => ({
      currentFeedbackList: [...state.currentFeedbackList, feedback],
    }));
  },

  // 현재 턴의 피드백 리스트에서 제거
  removeFeedbackFromCurrentList: (feedbackId: string) => {
    set((state) => ({
      currentFeedbackList: state.currentFeedbackList.filter(
        (f) => f.id !== feedbackId
      ),
    }));
  },

  // 현재 턴의 피드백 리스트 초기화
  clearCurrentFeedbackList: () => {
    set({ currentFeedbackList: [] });
  },

  // 객체 리스트 설정
  setObjectList: (objects: ObjectChip[]) => {
    set({
      compositionState: {
        ...get().compositionState,
        objects,
        selectedObjectId: null,
        bboxes: [], // 새로운 객체 리스트가 오면 BBOX 초기화
      },
    });
  },

  // 객체 추가
  addObject: (label: string) => {
    const state = get();
    const colors = [
      "#6366f1",
      "#8b5cf6",
      "#ec4899",
      "#f43f5e",
      "#ef4444",
      "#f59e0b",
      "#eab308",
      "#84cc16",
      "#22c55e",
      "#10b981",
      "#14b8a6",
      "#06b6d4",
      "#0ea5e9",
      "#3b82f6",
      "#6366f1",
    ];
    const existingColors = state.compositionState.objects.map(
      (obj) => obj.color
    );
    const availableColor =
      colors.find((c) => !existingColors.includes(c)) ||
      colors[state.compositionState.objects.length % colors.length];

    const newObject: ObjectChip = {
      id: `obj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      label,
      color: availableColor,
    };

    set({
      compositionState: {
        ...state.compositionState,
        objects: [...state.compositionState.objects, newObject],
      },
    });
  },

  // 객체 삭제
  removeObject: (objectId: string) => {
    const state = get();
    set({
      compositionState: {
        ...state.compositionState,
        objects: state.compositionState.objects.filter(
          (obj) => obj.id !== objectId
        ),
        bboxes: state.compositionState.bboxes.filter(
          (bbox) => bbox.objectId !== objectId
        ),
        selectedObjectId:
          state.compositionState.selectedObjectId === objectId
            ? null
            : state.compositionState.selectedObjectId,
      },
    });
  },

  // 객체 선택
  selectObject: (objectId: string | null) => {
    set({
      compositionState: {
        ...get().compositionState,
        selectedObjectId: objectId,
      },
    });
  },

  // BBOX 추가
  addBbox: (bbox: Omit<BoundingBox, "id">) => {
    const state = get();
    const newBbox: BoundingBox = {
      ...bbox,
      id: `bbox_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };
    set({
      compositionState: {
        ...state.compositionState,
        bboxes: [...state.compositionState.bboxes, newBbox],
      },
    });
  },

  // BBOX 업데이트
  updateBbox: (bboxId: string, updates: Partial<BoundingBox>) => {
    const state = get();
    set({
      compositionState: {
        ...state.compositionState,
        bboxes: state.compositionState.bboxes.map((bbox) =>
          bbox.id === bboxId ? { ...bbox, ...updates } : bbox
        ),
      },
    });
  },

  // BBOX 삭제
  removeBbox: (bboxId: string) => {
    const state = get();
    set({
      compositionState: {
        ...state.compositionState,
        bboxes: state.compositionState.bboxes.filter(
          (bbox) => bbox.id !== bboxId
        ),
      },
    });
  },

  // 구도 설정 초기화
  clearComposition: () => {
    set({
      compositionState: {
        objects: [],
        bboxes: [],
        selectedObjectId: null,
        isConfigured: false,
      },
    });
  },

  // 구도 설정과 함께 생성 시작
  startGenerationWithComposition: (
    prompt: string,
    sessionId: string,
    websocketUrl: string,
    bboxes?: BoundingBox[],
    sketchLayers?: SketchLayer[],
    interval?: number
  ) => {
    const state = get();
    // 구도 설정 완료 표시
    set({
      compositionState: {
        ...state.compositionState,
        isConfigured: true,
      },
    });

    const selectedInterval = interval || state.generationInterval;

    // 그래프 세션이 있으면 루트 노드에 composition 데이터 저장
    if (
      state.currentGraphSession &&
      state.currentGraphSession.id === sessionId
    ) {
      const rootNode = state.currentGraphSession.nodes.find(
        (n) => n.type === "prompt"
      );
      if (rootNode) {
        const updatedNodes = state.currentGraphSession.nodes.map((node) =>
          node.id === rootNode.id
            ? {
                ...node,
                data: {
                  ...node.data,
                  compositionData:
                    (bboxes && bboxes.length > 0) ||
                    (sketchLayers && sketchLayers.length > 0)
                      ? {
                          bboxes: bboxes || [],
                          sketchLayers: sketchLayers || [],
                        }
                      : undefined,
                },
              }
            : node
        );
        set({
          currentGraphSession: {
            ...state.currentGraphSession,
            nodes: updatedNodes,
          },
        });
      }
    }

    const newSession: ImageSession = {
      id: sessionId,
      prompt,
      totalSteps: 50, // 실제 추론 스텝 수
      steps: [],
      isComplete: false,
      createdAt: Date.now(),
      compositionBboxes: bboxes?.map((bbox) => ({
        id: bbox.id,
        objectId: bbox.objectId,
        x: bbox.x,
        y: bbox.y,
        width: bbox.width,
        height: bbox.height,
        color: bbox.color,
      })),
    };

    set({
      currentSession: newSession,
      selectedStepIndex: null,
      isGenerating: true,
      isPaused: false,
      isConnected: false, // WebSocket 연결 전
      generationInterval: selectedInterval,
    });

    // Mock 모드 체크
    if (USE_MOCK_MODE) {
      console.log("[ImageStore] Mock 모드: 시뮬레이션 시작 (구도 포함)");
      // 그래프 세션이 있으면 그래프 시뮬레이션, 없으면 일반 시뮬레이션
      const state = get();
      if (
        state.currentGraphSession &&
        state.currentGraphSession.id === sessionId
      ) {
        const rootNodeId = state.currentGraphSession.nodes.find(
          (n) => n.type === "prompt"
        )?.id;
        if (rootNodeId) {
          get().simulateGraphImageStream(
            sessionId,
            prompt,
            rootNodeId,
            selectedInterval
          );
        }
      } else {
        get().simulateImageStream(sessionId, prompt, selectedInterval);
      }
      return;
    }

    // 실제 WebSocket 연결
    console.log("[ImageStore] WebSocket 연결 시작 (구도 포함):", {
      sessionId,
      websocketUrl,
    });
    const ws = connectImageStream(
      sessionId,
      websocketUrl,
      (imageStep) => {
        // 이미지 스텝 수신
        console.log(
          "[ImageStore] 이미지 스텝 수신 (구도 포함):",
          imageStep.step
        );

        // ImageSession에 추가
        get().addImageStep(sessionId, imageStep);

        // GraphSession에도 추가 (위와 동일한 로직)
        const state = get();
        if (
          state.currentGraphSession &&
          state.currentGraphSession.id === sessionId
        ) {
          const nodeId = imageStep.nodeId || imageStep.id;
          const parentNodeId = imageStep.parentNodeId;
          const step = imageStep.step;
          const branchId = imageStep.branchId;

          if (parentNodeId) {
            const parentNode = state.currentGraphSession.nodes.find(
              (n) => n.id === parentNodeId
            );
            if (parentNode) {
              // 오른쪽으로 grow: 마지막 노드의 오른쪽에 배치
              const horizontalSpacing = 400; // 노드 너비(300px) + 간격(100px)

              // 브랜치의 마지막 노드 찾기
              let lastNode = null;
              if (branchId) {
                const branch = state.currentGraphSession.branches.find(
                  (b) => b.id === branchId
                );
                if (branch) {
                  const branchNodes = state.currentGraphSession.nodes.filter(
                    (n) => branch.nodes.includes(n.id)
                  );
                  const sortedNodes = branchNodes.sort(
                    (a, b) => (a.data.step || 0) - (b.data.step || 0)
                  );
                  lastNode = sortedNodes[sortedNodes.length - 1];
                }
              } else {
                // 메인 브랜치
                const mainBranch = state.currentGraphSession.branches.find(
                  (b) => !b.sourceNodeId
                );
                if (mainBranch) {
                  const mainBranchNodes =
                    state.currentGraphSession.nodes.filter((n) =>
                      mainBranch.nodes.includes(n.id)
                    );
                  const sortedNodes = mainBranchNodes.sort(
                    (a, b) => (a.data.step || 0) - (b.data.step || 0)
                  );
                  lastNode = sortedNodes[sortedNodes.length - 1];
                }
              }

              const position =
                lastNode && lastNode.id !== parentNodeId
                  ? {
                      x: lastNode.position.x + horizontalSpacing,
                      y: lastNode.position.y, // 같은 y 좌표 유지
                    }
                  : {
                      x: parentNode.position.x + horizontalSpacing,
                      y: parentNode.position.y, // 같은 y 좌표 유지
                    };

              if (branchId) {
                // 브랜치 노드 추가 (백엔드에서 보낸 nodeId 사용)
                get().addImageNodeToBranch(
                  sessionId,
                  branchId,
                  imageStep.url,
                  step,
                  position,
                  nodeId // 백엔드에서 보낸 nodeId 사용
                );
              } else {
                const mainBranch = state.currentGraphSession.branches.find(
                  (b) => !b.sourceNodeId
                );
                if (mainBranch) {
                  // 메인 브랜치에 노드 추가 (백엔드에서 보낸 nodeId 사용)
                  get().addImageNodeToBranch(
                    sessionId,
                    mainBranch.id,
                    imageStep.url,
                    step,
                    position,
                    nodeId // 백엔드에서 보낸 nodeId 사용
                  );
                } else {
                  // 일반 노드 추가 (백엔드에서 보낸 nodeId 사용)
                  get().addImageNode(
                    sessionId,
                    parentNodeId,
                    imageStep.url,
                    step,
                    position,
                    nodeId // 백엔드에서 보낸 nodeId 사용
                  );
                }
              }
            }
          }
        }
      },
      (error) => {
        // 에러 발생
        console.error("[ImageStore] WebSocket 에러 (구도 포함):", error);
        set({ isGenerating: false, isConnected: false, websocket: null });
      },
      () => {
        // 완료
        console.log("[ImageStore] 이미지 생성 완료 (구도 포함)");
        get().completeSession(sessionId);
        set({ isConnected: false, websocket: null });
      }
    );

    if (ws) {
      // WebSocket 연결 상태 확인
      ws.addEventListener("open", () => {
        console.log("[ImageStore] WebSocket 연결됨 (구도 포함)");
        set({ isConnected: true });
      });

      ws.addEventListener("close", () => {
        console.log("[ImageStore] WebSocket 연결 종료 (구도 포함)");
        set({ isConnected: false });
      });

      ws.addEventListener("error", (error) => {
        console.error("[ImageStore] WebSocket 에러 이벤트 (구도 포함):", error);
        set({ isConnected: false });
      });
    }

    set({
      websocket: ws,
      isConnected: ws !== null && ws.readyState === WebSocket.OPEN,
    });
  },

  // 현재 세션의 구도 BBOX 가져오기
  getCurrentCompositionBboxes: () => {
    const state = get();
    return state.currentSession?.compositionBboxes || null;
  },

  // BBOX별 피드백 히스토리에 추가
  addFeedbackToBboxHistory: (bboxId: string, feedback: FeedbackRecord) => {
    const state = get();
    if (!state.currentSession) return;

    const currentHistory = state.currentSession.bboxFeedbackHistory || {};
    const bboxHistory = currentHistory[bboxId] || [];

    set({
      currentSession: {
        ...state.currentSession,
        bboxFeedbackHistory: {
          ...currentHistory,
          [bboxId]: [...bboxHistory, feedback],
        },
      },
    });
  },

  // 특정 BBOX의 피드백 히스토리 가져오기
  getFeedbackHistoryForBbox: (bboxId: string) => {
    const state = get();
    if (!state.currentSession?.bboxFeedbackHistory) return [];

    return state.currentSession.bboxFeedbackHistory[bboxId] || [];
  },

  // Socket 시뮬레이션 (실제 서버 대신)
  simulateImageStream: (
    sessionId: string,
    prompt: string,
    interval?: number
  ) => {
    const totalSteps = 20;
    const state = get();
    const selectedInterval = interval || state.generationInterval;

    // 현재 세션의 기존 스텝 수를 확인
    const currentStepCount = state.currentSession?.steps.length || 0;

    console.log(
      `이미지 생성 ${
        currentStepCount > 0 ? "재시작" : "시작"
      }: ${prompt} (Session: ${sessionId}, 현재 스텝: ${currentStepCount}, 간격: ${selectedInterval})`
    );

    // 스텝별로 이미지 추가 시뮬레이션
    let currentStep = currentStepCount; // 기존 스텝 수부터 시작
    let intervalId: NodeJS.Timeout | null = null;

    const addNextStep = () => {
      const currentState = get();

      // 일시정지 상태이거나 세션이 완료된 경우 중단
      if (
        currentState.isPaused ||
        currentState.currentSession?.isComplete ||
        currentState.currentSession?.id !== sessionId
      ) {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        return;
      }

      // 피드백 요청이 있는 경우 일시정지
      if (currentState.feedbackRequest?.visible) {
        console.log("피드백 요청 대기 중... 생성 일시정지");
        return; // 피드백이 처리될 때까지 대기
      }

      // 간격에 따라 스텝 증가
      currentStep += selectedInterval;

      // 더미 이미지 URL 생성 (실제로는 서버에서 받을 이미지)
      const dummyImageUrl = `https://picsum.photos/512/512?random=${sessionId}&step=${currentStep}`;

      const imageStep: ImageStep = {
        id: `step_${sessionId}_${currentStep}`,
        url: dummyImageUrl,
        step: currentStep,
        timestamp: Date.now(),
      };

      console.log(
        `이미지 스텝 추가: ${currentStep}/${totalSteps} (간격: ${selectedInterval})`
      );
      get().addImageStep(sessionId, imageStep);

      // 특정 스텝에서 피드백 요청 트리거
      if (currentStep === 5) {
        console.log("스텝 5에서 피드백 요청 트리거");
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        get().showFeedbackRequest("full");
        return;
      } else if (currentStep === 10) {
        console.log("스텝 10에서 피드백 요청 트리거");
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        // 스텝 10에서는 포인팅 피드백 요청으로 시뮬레이션
        get().showFeedbackRequest("point");
        return;
      }

      // 완료되면 인터벌 정리
      if (currentStep >= totalSteps) {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        get().completeSession(sessionId);
        console.log(`이미지 생성 완료: ${sessionId}`);
      }
    };

    // 이미 완료된 경우 바로 리턴
    if (currentStep >= totalSteps) {
      console.log(`이미지 생성이 이미 완료됨: ${sessionId}`);
      return;
    }

    // 첫 번째 스텝 추가 (기존 스텝이 없는 경우에만)
    if (currentStepCount === 0) {
      addNextStep();
    }

    // 나머지 스텝들을 주기적으로 추가
    intervalId = setInterval(addNextStep, 800); // 800ms마다 업데이트
  },

  // 빈 그래프 세션 초기화
  initializeEmptyGraphSession: () => {
    const emptySession: GraphSession = {
      id: `empty_${Date.now()}`,
      nodes: [],
      edges: [],
      branches: [],
    };
    set({ currentGraphSession: emptySession });
    console.log("[ImageStore] 빈 그래프 세션 초기화");
  },

  // 그래프 세션 생성
  createGraphSession: (
    prompt: string,
    sessionId: string,
    rootNodeId?: string,
    bboxes?: BoundingBox[],
    sketchLayers?: SketchLayer[]
  ) => {
    const state = get();
    const currentSession = state.currentGraphSession;
    
    // 기존 세션에서 placeholder node 찾기
    let placeholderNode: GraphNode | undefined;
    let placeholderPosition = { x: 400, y: 50 }; // 기본 위치
    
    if (currentSession) {
      placeholderNode = currentSession.nodes.find((n) => n.type === "placeholder");
      if (placeholderNode) {
        placeholderPosition = placeholderNode.position;
        console.log("[ImageStore] Placeholder node 발견, 위치:", placeholderPosition);
      }
    }

    // Create unique branch ID for the main branch
    const uniqueMainBranchId = createUniqueBranchId(sessionId, "B0");

    // 프롬프트 노드 생성 (루트 노드)
    // placeholder node가 있으면 그 위치에, 없으면 기본 위치에
    const promptNodeId = rootNodeId || `node_prompt_${Date.now()}`;
    const rootNode: GraphNode = {
      id: promptNodeId,
      type: "prompt",
      data: {
        prompt,
        backendSessionId: sessionId, // Store the backend session ID
        backendBranchId: "B0", // Backend branch ID
        uniqueBranchId: uniqueMainBranchId, // Unique branch ID
        rowIndex: 0, // First prompt node is at row 0
        // Composition 데이터를 루트 노드에 저장
        compositionData:
          (bboxes && bboxes.length > 0) ||
          (sketchLayers && sketchLayers.length > 0)
            ? {
                bboxes: bboxes || [],
                sketchLayers: sketchLayers || [],
              }
            : undefined,
      },
      position: placeholderPosition,
    };

    // 메인 브랜치 생성 with unique ID
    const mainBranch: Branch = {
      id: uniqueMainBranchId,
      backendBranchId: "B0",
      backendSessionId: sessionId,
      sourceNodeId: promptNodeId,
      feedback: [],
      nodes: [],
    };

    // 기존 세션이 있고 placeholder node가 있으면, placeholder node를 제거하고 prompt node로 교체
    let nodes: GraphNode[] = [rootNode];
    let edges: GraphEdge[] = [];
    let branches: Branch[] = [mainBranch];
    
    if (currentSession && placeholderNode) {
      // placeholder node를 제외한 나머지 노드들 유지
      nodes = [
        rootNode,
        ...currentSession.nodes.filter((n) => n.id !== placeholderNode!.id),
      ];
      
      // placeholder node로 연결된 edge들을 prompt node로 재연결
      edges = currentSession.edges.map((e) => {
        if (e.target === placeholderNode!.id) {
          return { ...e, target: promptNodeId };
        }
        if (e.source === placeholderNode!.id) {
          return { ...e, source: promptNodeId };
        }
        return e;
      });
      
      // 기존 브랜치 유지 (새로운 메인 브랜치 추가)
      branches = [mainBranch, ...currentSession.branches];
      
      console.log("[ImageStore] Placeholder node를 prompt node로 변환:", {
        placeholderId: placeholderNode.id,
        promptNodeId,
        position: placeholderPosition,
      });
    }

    const graphSession: GraphSession = {
      id: sessionId,
      nodes,
      edges,
      branches,
    };

    set({ currentGraphSession: graphSession });
    console.log("[ImageStore] 그래프 세션 생성:", sessionId, "main branch:", uniqueMainBranchId);

    return sessionId;
  },

  // Placeholder 노드 추가
  addPlaceholderNode: (
    sessionId: string,
    position: { x: number; y: number },
    onClick?: () => void
  ): string => {
    const state = get();
    const session = state.currentGraphSession;
    if (!session || session.id !== sessionId) {
      console.error("[ImageStore] 세션을 찾을 수 없습니다:", sessionId);
      throw new Error("Session not found");
    }

    const nodeId = `node_placeholder_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const placeholderNode: GraphNode = {
      id: nodeId,
      type: "placeholder",
      data: {
        onClick,
      },
      position,
    };

    const updatedSession: GraphSession = {
      ...session,
      nodes: [...session.nodes, placeholderNode],
    };

    set({ currentGraphSession: updatedSession });
    console.log("[ImageStore] Placeholder 노드 추가:", nodeId);
    return nodeId;
  },

  // 이미지 노드 추가 (nodeId는 선택적, 제공되지 않으면 자동 생성)
  addImageNode: (
    sessionId: string,
    parentNodeId: string,
    imageUrl: string,
    step: number,
    position: { x: number; y: number },
    nodeId?: string,
    explicitBranchId?: string // Optional: explicit unique branch ID for non-main branches
  ): string => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      console.warn("[ImageStore] 그래프 세션을 찾을 수 없습니다:", sessionId);
      return "";
    }

    // FIRST: Get the backend branch ID and session ID from the parent node's session info
    // This must be done BEFORE generating node ID to ensure uniqueness across parallel sessions
    let backendBranchId = "B0";
    let backendSessionId: string | undefined;
    let uniqueBranchId: string | undefined;
    
    // If explicit branch ID is provided, use it (for non-main branches)
    if (explicitBranchId) {
      uniqueBranchId = explicitBranchId;
      backendBranchId = extractBackendBranchId(explicitBranchId);
      // Extract session ID from the unique branch ID if it's in the format "sess_xxx_Bx"
      const parts = explicitBranchId.split('_');
      if (parts.length >= 2 && parts[0] === 'sess') {
        // Find the backend session ID by looking at the branch
        const branch = state.currentGraphSession.branches.find(b => b.id === explicitBranchId);
        if (branch?.backendSessionId) {
          backendSessionId = branch.backendSessionId;
        }
      }
    }
    
    // Find the prompt node by tracing back from parent (to get session info if not provided)
    let currentId: string | null = parentNodeId;
    const visited = new Set<string>();
    while (currentId && !visited.has(currentId)) {
      visited.add(currentId);
      const currentNode = state.currentGraphSession.nodes.find((n) => n.id === currentId);
      if (currentNode?.type === "prompt") {
        // Found the prompt node - get its session info (only if not already set)
        if (!backendSessionId && currentNode.data?.backendSessionId) {
          backendSessionId = currentNode.data.backendSessionId;
        }
        // Only use prompt's branch info if no explicit branch ID was provided
        if (!explicitBranchId) {
          if (currentNode.data?.backendBranchId) {
            backendBranchId = currentNode.data.backendBranchId;
          }
          if (currentNode.data?.uniqueBranchId) {
            uniqueBranchId = currentNode.data.uniqueBranchId;
          }
          // Also check parallelSessions map
          const parallelSession = state.parallelSessions.get(currentId);
          if (parallelSession) {
            backendBranchId = parallelSession.backendBranchId;
            backendSessionId = parallelSession.backendSessionId;
            uniqueBranchId = createUniqueBranchId(backendSessionId, backendBranchId);
          }
        }
        break;
      }
      // Find parent via incoming edge
      const incomingEdge = state.currentGraphSession.edges.find((e) => e.target === currentId);
      currentId = incomingEdge ? incomingEdge.source : null;
    }

    // Create unique branch ID if not found
    if (!uniqueBranchId && backendSessionId) {
      uniqueBranchId = createUniqueBranchId(backendSessionId, backendBranchId);
    }

    // Generate node ID using backend session ID to ensure uniqueness across parallel sessions
    // Format: node_image_{backendSessionId}_{step} or node_image_{graphSessionId}_{step} for backwards compatibility
    const nodeIdPrefix = backendSessionId || sessionId;
    const finalNodeId = nodeId || `node_image_${nodeIdPrefix}_${step}_${Date.now()}`;

    console.log(`[ImageStore] addImageNode called: step=${step}, uniqueBranchId=${uniqueBranchId}, backendBranchId=${backendBranchId}, parentNodeId=${parentNodeId}, explicitBranchId=${explicitBranchId}`);

    // Check for existing node with same step AND same unique branch ID (not just by node ID)
    const existingNode = state.currentGraphSession.nodes.find(
      (n) => n.type === "image" && 
             n.data?.step === step && 
             n.data?.uniqueBranchId === uniqueBranchId
    );
    if (existingNode) {
      console.log(
        `[ImageStore] 노드가 이미 존재합니다 (step=${step}, branch=${uniqueBranchId}): ${existingNode.id}, existingNode.uniqueBranchId=${existingNode.data?.uniqueBranchId}, 이미지 URL 업데이트`
      );
      // Update existing node's image URL
      set({
        currentGraphSession: {
          ...state.currentGraphSession,
          nodes: state.currentGraphSession.nodes.map((n) =>
            n.id === existingNode.id ? { ...n, data: { ...n.data, imageUrl }, type: "image" } : n
          ),
        },
      });
      return existingNode.id;
    }

    // Find loading node with same step AND same unique branch ID
    let loadingNode = state.currentGraphSession.nodes.find(
      (n) => n.type === "loading" && 
             n.data?.step === step &&
             n.data?.uniqueBranchId === uniqueBranchId
    );
    
    // Fallback: If no exact match, find any loading node in the same branch
    // This handles cases where step numbers might not match exactly (e.g., after backtracking)
    if (!loadingNode && uniqueBranchId) {
      loadingNode = state.currentGraphSession.nodes.find(
        (n) => n.type === "loading" && 
               n.data?.uniqueBranchId === uniqueBranchId
      );
      if (loadingNode) {
        console.log(
          `[ImageStore] Found loading node with branch match only (loadingStep=${loadingNode.data?.step}, imageStep=${step}, branch=${uniqueBranchId})`
        );
      }
    }
    
    // Second fallback: Find loading node by backendBranchId if uniqueBranchId doesn't match
    if (!loadingNode && backendBranchId) {
      loadingNode = state.currentGraphSession.nodes.find(
        (n) => n.type === "loading" && 
               n.data?.backendBranchId === backendBranchId
      );
      if (loadingNode) {
        console.log(
          `[ImageStore] Found loading node with backendBranchId match only (loadingStep=${loadingNode.data?.step}, imageStep=${step}, backendBranch=${backendBranchId})`
        );
      }
    }
    
    if (loadingNode) {
      const loadingNodeStep = loadingNode.data?.step;
      console.log(
        `[ImageStore] Loading 노드를 이미지 노드로 교체 (loadingStep=${loadingNodeStep}, imageStep=${step}, branch=${uniqueBranchId}): ${loadingNode.id} -> ${finalNodeId}`
      );
      
      // Loading node를 image node로 교체
      // Use the new position if the step changed, otherwise keep the loading node's position
      const finalPosition = loadingNodeStep !== step ? position : loadingNode.position;
      
      const updatedNodes: GraphNode[] = state.currentGraphSession.nodes.map((n) =>
        n.id === loadingNode.id
          ? {
              ...n,
              id: finalNodeId,
              type: "image" as const,
              position: finalPosition, // Use calculated position
              data: { ...n.data, imageUrl, backendBranchId, backendSessionId, uniqueBranchId, step }, // Update step to actual step
            }
          : n
      );
      
      // Edge의 target도 업데이트
      const updatedEdges = state.currentGraphSession.edges.map((e) =>
        e.target === loadingNode.id
          ? { ...e, target: finalNodeId }
          : e
      );

      set({
        currentGraphSession: {
          ...state.currentGraphSession,
          nodes: updatedNodes,
          edges: updatedEdges,
        },
      });
      return finalNodeId;
    }

    const newNode: GraphNode = {
      id: finalNodeId,
      type: "image",
      data: { 
        imageUrl, 
        step, 
        sessionId: backendSessionId || sessionId, 
        backendBranchId,
        backendSessionId,
        uniqueBranchId,
      },
      position,
    };

    const edgeId = `edge_${parentNodeId}_${finalNodeId}`;
    const newEdge: GraphEdge = {
      id: edgeId,
      source: parentNodeId,
      target: finalNodeId,
      type: "default",
      data: {
        branchId: uniqueBranchId,
        backendBranchId,
      },
    };

    // Find the branch for this node using unique branch ID
    let targetBranch = uniqueBranchId 
      ? state.currentGraphSession.branches.find((b) => b.id === uniqueBranchId)
      : null;
    
    // Fallback: look for branch by backend session ID
    if (!targetBranch && backendSessionId) {
      targetBranch = state.currentGraphSession.branches.find(
        (b) => b.backendSessionId === backendSessionId && b.backendBranchId === backendBranchId
      );
    }
    
    // Final fallback: find any main branch
    if (!targetBranch) {
      targetBranch = state.currentGraphSession.branches.find(
        (b) => extractBackendBranchId(b.id) === "B0"
    );
    }

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: [...state.currentGraphSession.nodes, newNode],
        edges: [...state.currentGraphSession.edges, newEdge],
        branches: targetBranch
          ? state.currentGraphSession.branches.map((b) =>
              b.id === targetBranch!.id
                ? { ...b, nodes: [...b.nodes, finalNodeId] }
                : b
            )
          : state.currentGraphSession.branches,
      },
    });

    console.log(
      `[ImageStore] 이미지 노드 추가: ${finalNodeId}, 부모: ${parentNodeId}, 스텝: ${step}, 브랜치: ${targetBranch?.id || 'none'}`
    );
    return finalNodeId;
  },

  // Loading node 추가 (메인 브랜치)
  // branchId parameter is now the UNIQUE branch ID (e.g., "sess_abc123_B0")
  addLoadingNode: (
    sessionId: string,
    parentNodeId: string,
    step: number,
    position: { x: number; y: number },
    uniqueBranchId?: string
  ): string => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      console.warn("[ImageStore] 그래프 세션을 찾을 수 없습니다:", sessionId);
      return "";
    }

    // Extract backend branch ID from unique branch ID
    const backendBranchId = uniqueBranchId ? extractBackendBranchId(uniqueBranchId) : "B0";
    
    // Find the backend session ID from the branch or parent node
    let backendSessionId: string | undefined;
    if (uniqueBranchId) {
      const branch = state.currentGraphSession.branches.find((b) => b.id === uniqueBranchId);
      backendSessionId = branch?.backendSessionId;
    }
    if (!backendSessionId) {
      // Trace back to prompt node to get session ID
      let currentId: string | null = parentNodeId;
      const visited = new Set<string>();
      while (currentId && !visited.has(currentId)) {
        visited.add(currentId);
        const currentNode = state.currentGraphSession.nodes.find((n) => n.id === currentId);
        if (currentNode?.type === "prompt" && currentNode.data?.backendSessionId) {
          backendSessionId = currentNode.data.backendSessionId;
          break;
        }
        const incomingEdge = state.currentGraphSession.edges.find((e) => e.target === currentId);
        currentId = incomingEdge ? incomingEdge.source : null;
      }
    }

    const nodeId = `node_loading_${backendSessionId || sessionId}_${step}_${Date.now()}`;
    const loadingNode: GraphNode = {
      id: nodeId,
      type: "loading",
      data: { 
        step, 
        sessionId, 
        backendBranchId, 
        backendSessionId,
        uniqueBranchId, // Store unique branch ID for matching
      },
      position,
    };

    const edgeId = `edge_${parentNodeId}_${nodeId}`;
    const newEdge: GraphEdge = {
      id: edgeId,
      source: parentNodeId,
      target: nodeId,
      type: "default",
      data: {
        branchId: uniqueBranchId,
        backendBranchId,
      },
    };

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: [...state.currentGraphSession.nodes, loadingNode],
        edges: [...state.currentGraphSession.edges, newEdge],
      },
    });

    console.log(
      `[ImageStore] Loading 노드 추가: ${nodeId}, 부모: ${parentNodeId}, 스텝: ${step}, branch: ${uniqueBranchId}`
    );
    return nodeId;
  },

  // Loading node 추가 (브랜치)
  // branchId parameter is now the UNIQUE branch ID (e.g., "sess_abc123_B1")
  addLoadingNodeToBranch: (
    sessionId: string,
    uniqueBranchId: string,
    step: number,
    position?: { x: number; y: number }
  ): string => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      console.warn("[ImageStore] 그래프 세션을 찾을 수 없습니다:", sessionId);
      return "";
    }

    const branch = state.currentGraphSession.branches.find(
      (b) => b.id === uniqueBranchId
    );
    if (!branch) {
      console.warn("[ImageStore] 브랜치를 찾을 수 없습니다:", uniqueBranchId);
      return "";
    }

    // Extract backend branch ID and session ID from branch
    const backendBranchId = branch.backendBranchId || extractBackendBranchId(uniqueBranchId);
    const backendSessionId = branch.backendSessionId;

    const nodeId = `node_loading_${backendSessionId || sessionId}_${step}_${Date.now()}`;

    // Calculate position using unified getBranchRowIndex with session base row
    const sessionBaseRow = getSessionBaseRow(backendSessionId, state.currentGraphSession.nodes);
    const rowIndex = getBranchRowIndex(uniqueBranchId, state.currentGraphSession.branches, sessionBaseRow);
    
    let finalPosition: { x: number; y: number };
    if (position) {
      // If position is provided, use it but recalculate y based on rowIndex for consistency
      finalPosition = {
        x: position.x,
        y: GRID_START_Y + rowIndex * GRID_CELL_HEIGHT,
      };
    } else {
      finalPosition = {
        x: GRID_START_X + step * GRID_CELL_WIDTH,
        y: GRID_START_Y + rowIndex * GRID_CELL_HEIGHT,
      };
    }

    // 부모 노드 ID 찾기
      const branchNodes = state.currentGraphSession.nodes.filter((n) =>
        branch.nodes.includes(n.id)
      );
      const lastNode = branchNodes[branchNodes.length - 1];

    let parentNodeId: string;
      if (lastNode) {
      parentNodeId = lastNode.id;
      } else {
      const sourceEdge = state.currentGraphSession.edges.find(
        (e) => e.target === branch.sourceNodeId
        );
      if (sourceEdge) {
        parentNodeId = sourceEdge.source;
      } else {
        parentNodeId = branch.sourceNodeId;
      }
    }

    const loadingNode: GraphNode = {
      id: nodeId,
      type: "loading",
      data: { 
        step, 
        sessionId, 
        backendBranchId,
        backendSessionId,
        uniqueBranchId, // Store unique branch ID for matching
      },
      position: finalPosition,
    };

    const edgeId = `edge_${parentNodeId}_${nodeId}`;
    const branchFeedback = branch.feedback || [];
    const newEdge: GraphEdge = {
      id: edgeId,
      source: parentNodeId,
      target: nodeId,
      type: "branch",
      data: { 
        branchId: uniqueBranchId,
        backendBranchId,
        feedback: branchNodes.length === 0 ? branchFeedback : undefined,
      },
    };

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: [...state.currentGraphSession.nodes, loadingNode],
        edges: [...state.currentGraphSession.edges, newEdge],
      },
    });

    console.log(
      `[ImageStore] Loading 노드 추가 (브랜치): ${nodeId}, uniqueBranchId: ${uniqueBranchId}, backendBranchId: ${backendBranchId}, 스텝: ${step}`
    );
    return nodeId;
  },

  // Loading node 제거
  removeLoadingNode: (sessionId: string, nodeId: string) => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      console.warn("[ImageStore] 그래프 세션을 찾을 수 없습니다:", sessionId);
      return;
    }

    // 노드와 연결된 edge도 제거
    const edgesToRemove = state.currentGraphSession.edges.filter(
      (e) => e.source === nodeId || e.target === nodeId
    );

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: state.currentGraphSession.nodes.filter((n) => n.id !== nodeId),
        edges: state.currentGraphSession.edges.filter(
          (e) => !edgesToRemove.includes(e)
        ),
      },
    });

    console.log(`[ImageStore] Loading 노드 제거: ${nodeId}`);
  },

  // 브랜치에 이미지 노드 추가 (nodeId와 position은 선택적)
  // branchId parameter is now the UNIQUE branch ID (e.g., "sess_abc123_B1")
  addImageNodeToBranch: (
    sessionId: string,
    uniqueBranchId: string,
    imageUrl: string,
    step: number,
    position?: { x: number; y: number },
    nodeId?: string
  ): string => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      console.warn("[ImageStore] 그래프 세션을 찾을 수 없습니다:", sessionId);
      return "";
    }

    const branch = state.currentGraphSession.branches.find(
      (b) => b.id === uniqueBranchId
    );
    if (!branch) {
      console.warn("[ImageStore] 브랜치를 찾을 수 없습니다:", uniqueBranchId);
      return "";
    }

    // Get backend branch ID and session ID from the branch
    const backendBranchId = branch.backendBranchId || extractBackendBranchId(uniqueBranchId);
    const backendSessionId = branch.backendSessionId;

    // Generate node ID using backend session ID for uniqueness
    const finalNodeId =
      nodeId || `node_image_${backendSessionId || sessionId}_${step}_${Date.now()}`;

    // Check for existing node with same step AND same unique branch ID
    const existingNode = state.currentGraphSession.nodes.find(
      (n) => n.type === "image" && 
             n.data?.step === step && 
             n.data?.uniqueBranchId === uniqueBranchId
    );
    if (existingNode) {
      console.log(
        `[ImageStore] 브랜치 노드가 이미 존재합니다 (step=${step}, branch=${uniqueBranchId}): ${existingNode.id}, 이미지 URL 업데이트`
      );
      set({
        currentGraphSession: {
          ...state.currentGraphSession,
          nodes: state.currentGraphSession.nodes.map((n) =>
            n.id === existingNode.id ? { ...n, data: { ...n.data, imageUrl }, type: "image" } : n
          ),
        },
      });
      return existingNode.id;
    }

    // Find loading node with same step AND same unique branch ID
    const loadingNode = state.currentGraphSession.nodes.find(
      (n) => n.type === "loading" && 
             n.data?.step === step &&
             n.data?.uniqueBranchId === uniqueBranchId
    );
    if (loadingNode) {
      console.log(
        `[ImageStore] 브랜치 Loading 노드를 이미지 노드로 교체 (step=${step}, branch=${uniqueBranchId}): ${loadingNode.id} -> ${finalNodeId}`
      );
      // Loading node를 image node로 교체
      const updatedNodes: GraphNode[] = state.currentGraphSession.nodes.map((n) =>
        n.id === loadingNode.id
          ? {
              ...n,
              id: finalNodeId,
              type: "image" as const,
              data: { ...n.data, imageUrl, backendBranchId, backendSessionId, uniqueBranchId },
            }
          : n
      );
      
      // Edge의 target도 업데이트
      const updatedEdges = state.currentGraphSession.edges.map((e) =>
        e.target === loadingNode.id
          ? { ...e, target: finalNodeId }
          : e
      );

      // Update branch nodes list
      const updatedBranches = state.currentGraphSession.branches.map((b) =>
        b.id === uniqueBranchId 
          ? { ...b, nodes: [...b.nodes.filter(id => id !== loadingNode.id), finalNodeId] } 
          : b
      );

      set({
        currentGraphSession: {
          ...state.currentGraphSession,
          nodes: updatedNodes,
          edges: updatedEdges,
          branches: updatedBranches,
        },
      });
      return finalNodeId;
    }

    // Calculate position using unified getBranchRowIndex with session base row
    const sessionBaseRow = getSessionBaseRow(backendSessionId, state.currentGraphSession.nodes);
    const rowIndex = getBranchRowIndex(uniqueBranchId, state.currentGraphSession.branches, sessionBaseRow);
    
    let finalPosition: { x: number; y: number };
    if (position) {
      // If position is provided, use x from position but recalculate y based on rowIndex
      finalPosition = {
        x: position.x,
        y: GRID_START_Y + rowIndex * GRID_CELL_HEIGHT,
      };
    } else {
      // Calculate position based on step and branch row
      finalPosition = {
        x: GRID_START_X + step * GRID_CELL_WIDTH,
        y: GRID_START_Y + rowIndex * GRID_CELL_HEIGHT,
      };
    }

    // 부모 노드 ID 찾기
    const branchNodes = state.currentGraphSession.nodes.filter((n) =>
      branch.nodes.includes(n.id)
    );
    const lastNode = branchNodes[branchNodes.length - 1];
    
    // For the first node in a branch, connect to the PREVIOUS node (parent of source)
    // This creates a proper fork visualization from the previous step
    let parentNodeId: string;
    if (lastNode) {
      // Subsequent nodes connect to the last node in the branch
      parentNodeId = lastNode.id;
    } else {
      // First node in branch - find the parent of the source node
      const sourceEdge = state.currentGraphSession.edges.find(
        (e) => e.target === branch.sourceNodeId
      );
      if (sourceEdge) {
        // Connect to the parent of the source node (previous step)
        parentNodeId = sourceEdge.source;
      } else {
        // Fallback to source node if no parent found (e.g., source is root)
        parentNodeId = branch.sourceNodeId;
      }
    }

    console.log(
      `[ImageStore] 새 브랜치 노드 생성: nodeId=${finalNodeId}, imageUrl 길이=${
        imageUrl ? imageUrl.length : 0
      }, step=${step}, uniqueBranchId=${uniqueBranchId}, backendBranchId=${backendBranchId}`
    );
    const newNode: GraphNode = {
      id: finalNodeId,
      type: "image",
      data: { 
        imageUrl, 
        step, 
        sessionId, 
        backendBranchId, // Backend branch ID (e.g., "B0", "B1")
        backendSessionId, // Backend session ID
        uniqueBranchId, // Unique branch ID (e.g., "sess_abc123_B0")
      },
      position: finalPosition,
    };

    const edgeId = `edge_${parentNodeId}_${finalNodeId}`;
    // Get feedback from branch for the first node in branch
    const branchFeedback = branch.feedback || [];
    const newEdge: GraphEdge = {
      id: edgeId,
      source: parentNodeId,
      target: finalNodeId,
      type: "branch",
      data: { 
        branchId: uniqueBranchId, // Unique branch ID
        backendBranchId, // Backend branch ID
        // Include feedback for the first edge of the branch (from parent to first node)
        feedback: branchNodes.length === 0 ? branchFeedback : undefined,
      },
    };

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: [...state.currentGraphSession.nodes, newNode],
        edges: [...state.currentGraphSession.edges, newEdge],
        branches: state.currentGraphSession.branches.map((b) =>
          b.id === uniqueBranchId ? { ...b, nodes: [...b.nodes, finalNodeId] } : b
        ),
      },
    });

    console.log(
      `[ImageStore] 브랜치 이미지 노드 추가: ${finalNodeId}, 브랜치: ${uniqueBranchId}, 부모: ${parentNodeId}, 스텝: ${step}`
    );
    return finalNodeId;
  },

  // Edge 추가
  addEdge: (
    sessionId: string,
    source: string,
    target: string,
    edgeData?: GraphEdge["data"]
  ): string => {
    const state = get();
    const session = state.currentGraphSession;
    if (!session || session.id !== sessionId) {
      console.error("[ImageStore] 세션을 찾을 수 없습니다:", sessionId);
      throw new Error("Session not found");
    }

    const edgeId = `edge_${source}_${target}_${Date.now()}`;
    const newEdge: GraphEdge = {
      id: edgeId,
      source,
      target,
      type: "default",
      data: edgeData,
    };

    const updatedSession: GraphSession = {
      ...session,
      edges: [...session.edges, newEdge],
    };

    set({ currentGraphSession: updatedSession });
    console.log("[ImageStore] Edge 추가:", edgeId);
    return edgeId;
  },

  // 노드 위치 업데이트
  updateNodePosition: (
    sessionId: string,
    nodeId: string,
    position: { x: number; y: number }
  ) => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return;
    }

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: state.currentGraphSession.nodes.map((node) =>
          node.id === nodeId ? { ...node, position } : node
        ),
      },
    });
  },

  // Placeholder 노드 프롬프트 업데이트
  updatePlaceholderNodePrompt: (
    sessionId: string,
    nodeId: string,
    prompt: string
  ) => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return;
    }

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: state.currentGraphSession.nodes.map((node) =>
          node.id === nodeId && node.type === "placeholder"
            ? { ...node, data: { ...node.data, prompt } }
            : node
        ),
      },
    });
  },

  // Prompt 노드 프롬프트 업데이트
  updatePromptNodePrompt: (
    sessionId: string,
    nodeId: string,
    prompt: string
  ) => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return;
    }

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: state.currentGraphSession.nodes.map((node) =>
          node.id === nodeId && node.type === "prompt"
            ? { ...node, data: { ...node.data, prompt } }
            : node
        ),
      },
    });
  },

  // 빈 Prompt 노드 추가 (백엔드 세션 없이)
  addEmptyPromptNode: (
    sessionId: string,
    position: { x: number; y: number }
  ): string => {
    const state = get();
    const session = state.currentGraphSession;
    if (!session || session.id !== sessionId) {
      console.error("[ImageStore] 세션을 찾을 수 없습니다:", sessionId);
      throw new Error("Session not found");
    }

    // Find existing prompt nodes to calculate row index
    const existingPromptNodes = session.nodes.filter(
      (n) => n.type === "prompt"
    );
    const newRowIndex = existingPromptNodes.length;

    const nodeId = `node_prompt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const promptNode: GraphNode = {
      id: nodeId,
      type: "prompt",
      data: {
        prompt: "",
        rowIndex: newRowIndex,
      },
      position,
    };

    const updatedSession: GraphSession = {
      ...session,
      nodes: [...session.nodes, promptNode],
    };

    set({ currentGraphSession: updatedSession });
    console.log("[ImageStore] 빈 Prompt 노드 추가:", nodeId);
    return nodeId;
  },

  // Prompt 노드 뒤의 모든 이미지 노드 제거 (재생성 시 사용)
  // BFS를 사용하여 Prompt 노드에서 시작해서 연결된 모든 이미지/로딩 노드를 재귀적으로 찾아 제거
  removeImageNodesAfterPrompt: (
    sessionId: string,
    promptNodeId: string
  ): void => {
    const state = get();
    const session = state.currentGraphSession;
    if (!session || session.id !== sessionId) {
      console.error("[ImageStore] 세션을 찾을 수 없습니다:", sessionId);
      return;
    }

    // Prompt 노드 찾기
    const promptNode = session.nodes.find(
      (n) => n.id === promptNodeId && n.type === "prompt"
    );
    if (!promptNode) {
      console.error("[ImageStore] Prompt 노드를 찾을 수 없습니다:", promptNodeId);
      return;
    }

    // BFS를 사용하여 Prompt 노드에서 시작해서 연결된 모든 이미지/로딩 노드 찾기
    const nodesToRemove = new Set<string>();
    const queue: string[] = [promptNodeId];
    const visited = new Set<string>();

    while (queue.length > 0) {
      const currentId = queue.shift()!;
      if (visited.has(currentId)) continue;
      visited.add(currentId);

      // 현재 노드가 이미지나 로딩 노드이면 제거 대상에 추가
      const currentNode = session.nodes.find((n) => n.id === currentId);
      if (currentNode && (currentNode.type === "image" || currentNode.type === "loading")) {
        nodesToRemove.add(currentId);
      }

      // 현재 노드에서 나가는 모든 엣지를 찾아서 자식 노드들을 큐에 추가
      for (const edge of session.edges) {
        if (edge.source === currentId && !visited.has(edge.target)) {
          queue.push(edge.target);
        }
      }
    }

    // 연결된 엣지도 제거 (제거 대상 노드와 관련된 모든 엣지)
    const edgesToRemove = session.edges.filter(
      (e) => nodesToRemove.has(e.source) || nodesToRemove.has(e.target)
    );

    // 노드와 엣지 제거
    const updatedNodes = session.nodes.filter(
      (n) => !nodesToRemove.has(n.id)
    );
    const updatedEdges = session.edges.filter(
      (e) => !edgesToRemove.some((er) => er.id === e.id)
    );

    set({
      currentGraphSession: {
        ...session,
        nodes: updatedNodes,
        edges: updatedEdges,
      },
    });

    console.log(
      `[ImageStore] Prompt 노드 ${promptNodeId} 뒤의 ${nodesToRemove.size}개 노드 제거됨:`,
      Array.from(nodesToRemove)
    );
  },

  // 노드 선택
  selectNode: (nodeId: string | null) => {
    set({ selectedNodeId: nodeId });
  },

  // 피드백 엣지 hover 상태 설정
  setHoveredFeedbackEdge: (branchId: string | null, bboxFeedbacks: FeedbackRecord[] = []) => {
    if (branchId === null) {
      set({ hoveredFeedbackEdge: null });
    } else {
      set({ hoveredFeedbackEdge: { branchId, bboxFeedbacks } });
    }
  },

  // 그래프 구조에 맞는 이미지 스트림 시뮬레이션 (메인 브랜치)
  simulateGraphImageStream: (
    sessionId: string,
    prompt: string,
    rootNodeId: string,
    interval?: number
  ) => {
    const totalSteps = 20;
    const state = get();
    const selectedInterval = interval || state.generationInterval;

    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return;
    }

    // 메인 브랜치 찾기 (rootNodeId를 sourceNodeId로 하는 브랜치)
    const mainBranch = state.currentGraphSession.branches.find(
      (b) => b.sourceNodeId === rootNodeId && b.id.startsWith("branch_main_")
    );

    if (!mainBranch) {
      console.error("메인 브랜치를 찾을 수 없습니다:", rootNodeId);
      return;
    }

    console.log(
      `[ImageStore] 그래프 이미지 생성 시뮬레이션 시작: ${prompt} (Session: ${sessionId}, Root: ${rootNodeId}, Branch: ${mainBranch.id})`
    );

    let currentStep = 0;
    let lastNodeId = rootNodeId;
    let intervalId: NodeJS.Timeout | null = null;

    const addNextStep = () => {
      const currentState = get();

      // 일시정지 상태이거나 세션이 완료된 경우 중단
      if (
        currentState.isPaused ||
        !currentState.currentGraphSession ||
        currentState.currentGraphSession.id !== sessionId
      ) {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        return;
      }

      // 간격에 따라 스텝 증가
      currentStep += selectedInterval;

      // 더미 이미지 URL 생성
      const dummyImageUrl = `https://picsum.photos/512/512?random=${sessionId}&step=${currentStep}`;

      // 메인 브랜치에 이미지 노드 추가
      const newNodeId = get().addImageNodeToBranch(
        sessionId,
        mainBranch.id,
        dummyImageUrl,
        currentStep
      );
      if (newNodeId) {
        lastNodeId = newNodeId;
      }

      console.log(
        `[ImageStore] 이미지 스텝 추가: ${currentStep}/${totalSteps} (간격: ${selectedInterval})`
      );

      // 완료되면 인터벌 정리
      if (currentStep >= totalSteps) {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        set({ isGenerating: false });
        console.log(`[ImageStore] 그래프 이미지 생성 완료: ${sessionId}`);
      }
    };

    // 첫 번째 스텝 추가
    addNextStep();

    // 나머지 스텝들을 주기적으로 추가
    intervalId = setInterval(addNextStep, 800); // 800ms마다 업데이트
  },

  // 브랜치용 이미지 스트림 시뮬레이션
  simulateBranchImageStream: (
    sessionId: string,
    branchId: string,
    interval?: number
  ) => {
    const totalSteps = 20;
    const state = get();
    const selectedInterval = interval || state.generationInterval;

    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return;
    }

    const branch = state.currentGraphSession.branches.find(
      (b) => b.id === branchId
    );
    if (!branch) {
      console.error("[ImageStore] 브랜치를 찾을 수 없습니다:", branchId);
      return;
    }

    console.log(
      `[ImageStore] 브랜치 이미지 생성 시뮬레이션 시작: (Session: ${sessionId}, Branch: ${branchId})`
    );

    let currentStep = 0;
    let lastNodeId = branch.sourceNodeId;
    let intervalId: NodeJS.Timeout | null = null;

    const addNextStep = () => {
      const currentState = get();

      // 일시정지 상태이거나 세션이 완료된 경우 중단
      if (
        currentState.isPaused ||
        !currentState.currentGraphSession ||
        currentState.currentGraphSession.id !== sessionId
      ) {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        return;
      }

      // 브랜치가 여전히 존재하는지 확인
      const currentBranch = currentState.currentGraphSession.branches.find(
        (b) => b.id === branchId
      );
      if (!currentBranch) {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        return;
      }

      // 간격에 따라 스텝 증가
      currentStep += selectedInterval;

      // 더미 이미지 URL 생성
      const dummyImageUrl = `https://picsum.photos/512/512?random=${sessionId}&branch=${branchId}&step=${currentStep}`;

      // 브랜치에 이미지 노드 추가
      const newNodeId = get().addImageNodeToBranch(
        sessionId,
        branchId,
        dummyImageUrl,
        currentStep
      );
      if (newNodeId) {
        lastNodeId = newNodeId;
        console.log(
          `[ImageStore] 브랜치 이미지 스텝 추가: ${currentStep}/${totalSteps} (간격: ${selectedInterval})`
        );
      }

      // 완료되면 인터벌 정리
      if (currentStep >= totalSteps) {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        console.log(`[ImageStore] 브랜치 이미지 생성 완료: ${branchId}`);
      }
    };

    // 첫 번째 스텝 추가
    addNextStep();

    // 나머지 스텝들을 주기적으로 추가
    intervalId = setInterval(addNextStep, 800); // 800ms마다 업데이트
  },

  // 현재 노드에서 루트까지 역방향으로 올라가며 브랜치 피드백 수집
  getBranchFeedbacksForNode: (nodeId: string): FeedbackRecord[] => {
    const state = get();
    if (!state.currentGraphSession) {
      return [];
    }

    const { nodes, edges, branches } = state.currentGraphSession;
    const allFeedbacks: FeedbackRecord[] = [];
    const visitedBranches = new Set<string>();

    // 현재 노드에서 시작하여 루트까지 역방향으로 경로 탐색
    let currentNodeId: string | null = nodeId;
    const visited = new Set<string>();

    while (currentNodeId && !visited.has(currentNodeId)) {
      visited.add(currentNodeId);

      // 현재 노드로 들어오는 엣지 찾기 (부모 노드 찾기)
      const incomingEdges = edges.filter((e) => e.target === currentNodeId);
      if (incomingEdges.length === 0) {
        // 루트 노드에 도달
        break;
      }

      // 각 부모 엣지를 확인하여 브랜치 피드백 수집
      for (const edge of incomingEdges) {
        if (edge.type === "branch" && edge.data?.branchId) {
          const branchId = edge.data.branchId;
          if (!visitedBranches.has(branchId)) {
            visitedBranches.add(branchId);

            // 해당 브랜치의 피드백 찾기
            const branch = branches.find((b) => b.id === branchId);
            if (branch && branch.feedback) {
              allFeedbacks.push(...branch.feedback);
            }
          }
        }
      }

      // 첫 번째 부모 노드로 이동 (일반적으로 하나의 부모만 있음)
      currentNodeId = incomingEdges[0]?.source || null;
    }

    // 시간순으로 정렬 (오래된 것부터)
    return allFeedbacks.sort((a, b) => a.timestamp - b.timestamp);
  },

  // Remove a node and all its descendants (for backtracking)
  removeNodeAndDescendants: (sessionId: string, nodeId: string) => {
    const state = get();
    if (!state.currentGraphSession || state.currentGraphSession.id !== sessionId) {
      console.warn("[ImageStore] Cannot remove node: session not found");
      return;
    }

    const { nodes, edges, branches } = state.currentGraphSession;
    
    // Find all descendant nodes using BFS
    const nodesToRemove = new Set<string>();
    const queue = [nodeId];
    
    while (queue.length > 0) {
      const currentId = queue.shift()!;
      nodesToRemove.add(currentId);
      
      // Find all children (nodes that have edges from currentId)
      for (const edge of edges) {
        if (edge.source === currentId && !nodesToRemove.has(edge.target)) {
          queue.push(edge.target);
        }
      }
    }
    
    console.log(`[ImageStore] Removing nodes: ${Array.from(nodesToRemove).join(", ")}`);
    
    // Filter out removed nodes and edges
    const newNodes = nodes.filter((n) => !nodesToRemove.has(n.id));
    const newEdges = edges.filter(
      (e) => !nodesToRemove.has(e.source) && !nodesToRemove.has(e.target)
    );
    
    // Update branches to remove references to deleted nodes
    let updatedBranches = branches.map((branch) => ({
      ...branch,
      nodes: branch.nodes.filter((nid) => !nodesToRemove.has(nid)),
    }));
    
    // Find branches that are now empty (have no nodes) and are not main branches (B0)
    // Main branches should be kept even if empty (they represent the session's main path)
    const branchesToRemove = updatedBranches.filter((branch) => {
      const backendBranchId = extractBackendBranchId(branch.id);
      // Keep main branches (B0) even if empty
      if (backendBranchId === "B0") return false;
      // Remove non-main branches that have no nodes
      return branch.nodes.length === 0;
    });
    
    if (branchesToRemove.length > 0) {
      console.log(`[ImageStore] Removing empty branches: ${branchesToRemove.map(b => b.id).join(", ")}`);
      const branchIdsToRemove = new Set(branchesToRemove.map(b => b.id));
      updatedBranches = updatedBranches.filter(b => !branchIdsToRemove.has(b.id));
    }
    
    // Clear selection if the selected node was removed
    const newSelectedNodeId = nodesToRemove.has(state.selectedNodeId || "")
      ? null
      : state.selectedNodeId;
    
    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: newNodes,
        edges: newEdges,
        branches: updatedBranches,
      },
      selectedNodeId: newSelectedNodeId,
    });
    
    // Note: Node repositioning is handled by GraphCanvas's nodes memo
    // which recalculates positions based on the updated branches array
    console.log(`[ImageStore] Removed ${nodesToRemove.size} nodes, ${branchesToRemove.length} branches. Remaining: ${newNodes.length} nodes, ${updatedBranches.length} branches`);
  },

  // 프롬프트 노드와 연결된 모든 노드 및 브랜치 제거
  removePromptNodeAndBranch: (sessionId: string, promptNodeId: string) => {
    const state = get();
    if (!state.currentGraphSession || state.currentGraphSession.id !== sessionId) {
      console.warn("[ImageStore] Cannot remove prompt node: session not found");
      return;
    }

    const { nodes, edges, branches } = state.currentGraphSession;
    
    // 프롬프트 노드 찾기
    const promptNode = nodes.find((n) => n.id === promptNodeId && n.type === "prompt");
    if (!promptNode) {
      console.warn("[ImageStore] Prompt node not found:", promptNodeId);
      return;
    }

    // BFS를 사용하여 프롬프트 노드에서 시작해서 연결된 모든 하위 노드 찾기
    const nodesToRemove = new Set<string>();
    const queue = [promptNodeId];
    
    while (queue.length > 0) {
      const currentId = queue.shift()!;
      nodesToRemove.add(currentId);
      
      // Find all children (nodes that have edges from currentId)
      for (const edge of edges) {
        if (edge.source === currentId && !nodesToRemove.has(edge.target)) {
          queue.push(edge.target);
        }
      }
    }
    
    console.log(`[ImageStore] Removing prompt node and descendants: ${Array.from(nodesToRemove).join(", ")}`);
    
    // Filter out removed nodes and edges
    const newNodes = nodes.filter((n) => !nodesToRemove.has(n.id));
    const newEdges = edges.filter(
      (e) => !nodesToRemove.has(e.source) || !nodesToRemove.has(e.target)
    );
    
    // Get unique branch ID from prompt node
    const uniqueBranchId = promptNode.data?.uniqueBranchId as string | undefined;
    
    // Update branches to remove references to deleted nodes
    let updatedBranches = branches.map((branch) => ({
      ...branch,
      nodes: branch.nodes.filter((nid) => !nodesToRemove.has(nid)),
    }));
    
    // Remove the branch associated with this prompt node
    if (uniqueBranchId) {
      const branchToRemove = updatedBranches.find((b) => b.id === uniqueBranchId);
      if (branchToRemove) {
        console.log(`[ImageStore] Removing branch: ${uniqueBranchId}`);
        updatedBranches = updatedBranches.filter((b) => b.id !== uniqueBranchId);
      }
    }
    
    // Also remove any other empty branches (except main branch B0)
    const branchesToRemove = updatedBranches.filter((branch) => {
      const backendBranchId = extractBackendBranchId(branch.id);
      if (backendBranchId === "B0") return false;
      return branch.nodes.length === 0;
    });
    
    if (branchesToRemove.length > 0) {
      console.log(`[ImageStore] Removing empty branches: ${branchesToRemove.map(b => b.id).join(", ")}`);
      const branchIdsToRemove = new Set(branchesToRemove.map(b => b.id));
      updatedBranches = updatedBranches.filter(b => !branchIdsToRemove.has(b.id));
    }
    
    // Remove from parallelSessions
    const newParallelSessions = new Map(state.parallelSessions);
    if (newParallelSessions.has(promptNodeId)) {
      console.log(`[ImageStore] Removing parallel session for prompt node: ${promptNodeId}`);
      newParallelSessions.delete(promptNodeId);
    }
    
    // Clear selection if the selected node was removed
    const newSelectedNodeId = nodesToRemove.has(state.selectedNodeId || "")
      ? null
      : state.selectedNodeId;
    
    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: newNodes,
        edges: newEdges,
        branches: updatedBranches,
      },
      parallelSessions: newParallelSessions,
      selectedNodeId: newSelectedNodeId,
    });
    
    console.log(`[ImageStore] Prompt node ${promptNodeId} and all connected nodes removed`);
  },

  // Register a parallel session for a prompt node
  registerParallelSession: (promptNodeId: string, backendSessionId: string, backendBranchId: string) => {
    const state = get();
    const newMap = new Map(state.parallelSessions);
    newMap.set(promptNodeId, { backendSessionId, backendBranchId });
    set({ parallelSessions: newMap });
    console.log(`[ImageStore] Registered parallel session for prompt ${promptNodeId}: session=${backendSessionId}, branch=${backendBranchId}`);
  },

  // Get backend session info for a node (traces back to its prompt node)
  getBackendSessionForNode: (nodeId: string) => {
    const state = get();
    if (!state.currentGraphSession) return null;
    
    const { nodes, edges } = state.currentGraphSession;
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return null;
    
    // If this is a prompt node with its own backend session, use it
    if (node.type === "prompt") {
      // Check parallelSessions map first
      const parallelSession = state.parallelSessions.get(nodeId);
      if (parallelSession) {
        return { sessionId: parallelSession.backendSessionId, branchId: parallelSession.backendBranchId };
      }
      // Check node data
      if (node.data?.backendSessionId) {
        return { 
          sessionId: node.data.backendSessionId, 
          branchId: node.data.backendBranchId || "B0" 
        };
      }
    }
    
    // If this node has a backendSessionId directly, use it
    if (node.data?.backendSessionId) {
      return { 
        sessionId: node.data.backendSessionId, 
        branchId: node.data.backendBranchId || "B0" 
      };
    }
    
    // Trace back to find the root prompt node for this branch
    let currentId: string | null = nodeId;
    const visited = new Set<string>();
    
    while (currentId && !visited.has(currentId)) {
      visited.add(currentId);
      const currentNode = nodes.find((n) => n.id === currentId);
      
      if (currentNode?.type === "prompt") {
        // Check parallelSessions map
        const parallelSession = state.parallelSessions.get(currentId);
        if (parallelSession) {
          return { sessionId: parallelSession.backendSessionId, branchId: parallelSession.backendBranchId };
        }
        // Check node data
        if (currentNode.data?.backendSessionId) {
          return { 
            sessionId: currentNode.data.backendSessionId, 
            branchId: currentNode.data.backendBranchId || "B0" 
          };
        }
      }
      
      // Find parent via incoming edge
      const incomingEdge = edges.find((e) => e.target === currentId);
      if (incomingEdge) {
        currentId = incomingEdge.source;
      } else {
        break;
      }
    }
    
    // Fall back to the global backend session ID
    if (state.backendSessionId) {
      return { sessionId: state.backendSessionId, branchId: state.backendActiveBranchId || "B0" };
    }
    
    return null;
  },

  // Add a new prompt node for parallel session
  addPromptNodeToGraph: (
    prompt: string,
    backendSessionId: string,
    backendBranchId: string,
    bboxes?: BoundingBox[],
    sketchLayers?: SketchLayer[],
    position?: { x: number; y: number },
    placeholderNodeId?: string
  ): string | null => {
    const state = get();
    if (!state.currentGraphSession) {
      console.warn("[ImageStore] Cannot add prompt node: no graph session");
      return null;
    }

    // Check if we're updating an existing prompt node
    const existingPromptNode = placeholderNodeId 
      ? state.currentGraphSession.nodes.find((n) => n.id === placeholderNodeId && n.type === "prompt")
      : null;
    
    // Calculate row index: use existing rowIndex if updating, otherwise calculate new one
    let newRowIndex: number;
    if (existingPromptNode?.data?.rowIndex !== undefined) {
      // Keep existing rowIndex when updating
      newRowIndex = existingPromptNode.data.rowIndex as number;
    } else {
      // Calculate new rowIndex for new prompt node
      const existingPromptNodes = state.currentGraphSession.nodes.filter(
        (n) => n.type === "prompt"
      );
      newRowIndex = existingPromptNodes.length;
    }
    
    // Calculate position for new prompt node
    let newPosition: { x: number; y: number };
    if (position) {
      newPosition = position;
    } else if (placeholderNodeId) {
      // Use placeholder node's position if converting
      const placeholderNode = state.currentGraphSession.nodes.find(
        (n) => n.id === placeholderNodeId && n.type === "placeholder"
      );
      if (placeholderNode) {
        newPosition = placeholderNode.position;
      } else if (existingPromptNode) {
        // Keep existing position when updating prompt node
        newPosition = existingPromptNode.position;
      } else {
        // Default position based on row index
        newPosition = {
          x: GRID_START_X - GRID_CELL_WIDTH,
          y: GRID_START_Y + newRowIndex * GRID_CELL_HEIGHT,
        };
      }
    } else {
      // Default position based on row index
      newPosition = {
        x: GRID_START_X - GRID_CELL_WIDTH,
        y: GRID_START_Y + newRowIndex * GRID_CELL_HEIGHT,
      };
    }

    // Create unique branch ID for this session
    const uniqueBranchId = createUniqueBranchId(backendSessionId, backendBranchId);

    // Create new prompt node with backend session info
    const promptNodeId = placeholderNodeId || `node_prompt_${Date.now()}`;
    const newPromptNode: GraphNode = {
      id: promptNodeId,
      type: "prompt",
      data: {
        prompt,
        backendSessionId,
        backendBranchId, // Backend branch ID (e.g., "B0")
        uniqueBranchId, // Unique branch ID (e.g., "sess_abc123_B0")
        rowIndex: newRowIndex,
        compositionData:
          (bboxes && bboxes.length > 0) ||
          (sketchLayers && sketchLayers.length > 0)
            ? {
                bboxes: bboxes || [],
                sketchLayers: sketchLayers || [],
              }
            : undefined,
      },
      position: newPosition,
    };

    // Create a branch for this prompt node's session with unique ID
    const newBranch: Branch = {
      id: uniqueBranchId,
      backendBranchId,
      backendSessionId,
      sourceNodeId: promptNodeId,
      feedback: [],
      nodes: [],
    };

    // Update or add the node
    let updatedNodes: GraphNode[];
    if (placeholderNodeId) {
      // Convert placeholder to prompt node or update existing prompt node
      const existingNode = state.currentGraphSession.nodes.find((n) => n.id === placeholderNodeId);
      if (existingNode?.type === "placeholder") {
        // Convert placeholder to prompt node
        updatedNodes = state.currentGraphSession.nodes.map((n) =>
          n.id === placeholderNodeId ? newPromptNode : n
        );
        console.log(`[ImageStore] Converted placeholder ${placeholderNodeId} to prompt node`);
      } else if (existingNode?.type === "prompt") {
        // Update existing prompt node with new session info
        updatedNodes = state.currentGraphSession.nodes.map((n) =>
          n.id === placeholderNodeId ? newPromptNode : n
        );
        console.log(`[ImageStore] Updated prompt node ${placeholderNodeId} with new session info`);
      } else {
        // Add new prompt node
        updatedNodes = [...state.currentGraphSession.nodes, newPromptNode];
      }
    } else {
      // Add new prompt node
      updatedNodes = [...state.currentGraphSession.nodes, newPromptNode];
    }

    // Update the graph session
    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: updatedNodes,
        branches: [...state.currentGraphSession.branches, newBranch],
      },
    });

    // Register the parallel session
    const newMap = new Map(state.parallelSessions);
    newMap.set(promptNodeId, { backendSessionId, backendBranchId });
    set({ parallelSessions: newMap });

    console.log(`[ImageStore] Added new prompt node: ${promptNodeId} with backend session: ${backendSessionId}, branch: ${backendBranchId}, row: ${newRowIndex}`);
    return promptNodeId;
  },
  
  // 북마크 토글
  toggleBookmark: (nodeId: string) => {
    const state = get();
    const isCurrentlyBookmarked = state.bookmarkedNodeIds.includes(nodeId);
    
    if (isCurrentlyBookmarked) {
      set({
        bookmarkedNodeIds: state.bookmarkedNodeIds.filter((id) => id !== nodeId),
      });
    } else {
      set({
        bookmarkedNodeIds: [...state.bookmarkedNodeIds, nodeId],
      });
    }
  },
  
  // 북마크 여부 확인
  isBookmarked: (nodeId: string) => {
    const state = get();
    return state.bookmarkedNodeIds.includes(nodeId);
  },
  
  // 세션 저장/로드
  saveSessionToServer: async (mode: string, participant: number) => {
    const state = get();
    if (!state.currentGraphSession) {
      console.warn("[ImageStore] No currentGraphSession to save");
      return;
    }
    
    try {
      console.log(`[ImageStore] Saving session: mode=${mode}, participant=${participant}, bookmarkedNodeIds=${JSON.stringify(state.bookmarkedNodeIds)}`);
      if (mode === "prompt") {
        await saveSessionPrompt(mode, participant, state.currentGraphSession, state.bookmarkedNodeIds);
      } else {
        await saveSessionStep(mode, participant, state.currentGraphSession, state.bookmarkedNodeIds);
      }
      console.log(`[ImageStore] Session saved: mode=${mode}, participant=${participant}`);
    } catch (error) {
      console.error("[ImageStore] Failed to save session:", error);
      throw error;
    }
  },
  
  loadSessionFromServer: async (mode: string, participant: number) => {
    try {
      const result = mode === "prompt" 
        ? await loadSessionPrompt(mode, participant)
        : await loadSessionStep(mode, participant);
      
      if (!result) {
        console.log(`[ImageStore] No session found: mode=${mode}, participant=${participant}`);
        // null을 반환하면 빈 세션 초기화를 위해 에러를 throw하지 않고 그냥 return
        // GraphCanvas에서 catch 후 initializeEmptyGraphSession을 호출하도록 함
        return;
      }
      
      const { graphSession, lastUpdated, bookmarkedNodeIds } = result;
      console.log(`[ImageStore] Session loaded: mode=${mode}, participant=${participant}, lastUpdated=${lastUpdated}`);
      
      set({
        currentGraphSession: graphSession,
        bookmarkedNodeIds: bookmarkedNodeIds || [],
      });
    } catch (error) {
      console.error("[ImageStore] Failed to load session:", error);
      // 404 에러는 정상적인 경우이므로 throw하지 않고 return
      // 다른 에러만 throw
      if (error instanceof Error && error.message.includes("404")) {
        console.log(`[ImageStore] Session not found (404), will initialize empty session`);
        return;
      }
      throw error;
    }
  },
}));
