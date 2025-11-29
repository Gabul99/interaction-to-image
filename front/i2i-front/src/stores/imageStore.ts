import { create } from "zustand";
import {
  type FeedbackArea,
  type ObjectChip,
  type BoundingBox,
  type CompositionState,
  type FeedbackRecord,
  type GraphNode,
  type GraphEdge,
  type Branch,
} from "../types";

export interface ImageStep {
  id: string; // 서버에서 보내준 UUID
  url: string;
  step: number;
  timestamp: number;
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

export interface GraphSession {
  id: string; // Session ID
  rootNodeId: string; // 프롬프트 노드 ID
  nodes: GraphNode[];
  edges: GraphEdge[];
  branches: Branch[];
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
}

export interface ImageStreamState {
  // 현재 활성 세션 (기존 구조 - 호환성 유지)
  currentSession: ImageSession | null;

  // 그래프 세션 (새 구조)
  currentGraphSession: GraphSession | null;

  // 현재 선택된 노드 ID
  selectedNodeId: string | null;

  // 현재 선택된 스텝 (History에서 클릭한 스텝) - 기존 호환성
  selectedStepIndex: number | null; // null이면 최신 스텝 표시

  // 생성 상태
  isGenerating: boolean;
  isPaused: boolean; // 일시정지 상태

  // Socket 연결 상태
  isConnected: boolean;

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
  startGeneration: (prompt: string, interval?: number) => void;
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
    bboxes?: BoundingBox[],
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

  // 그래프 관련 액션
  createGraphSession: (prompt: string, bboxes?: BoundingBox[]) => string; // 세션 ID 반환
  addPromptNode: (
    sessionId: string,
    prompt: string,
    position: { x: number; y: number }
  ) => string; // 노드 ID 반환
  addImageNode: (
    sessionId: string,
    parentNodeId: string,
    imageUrl: string,
    step: number,
    position: { x: number; y: number }
  ) => string; // 노드 ID 반환
  selectNode: (nodeId: string | null) => void;
  createBranch: (
    sessionId: string,
    sourceNodeId: string,
    feedback: FeedbackRecord[]
  ) => string; // 브랜치 ID 반환
  addImageNodeToBranch: (
    sessionId: string,
    branchId: string,
    imageUrl: string,
    step: number,
    position?: { x: number; y: number }
  ) => string; // 노드 ID 반환
  getNodeById: (sessionId: string, nodeId: string) => GraphNode | null;
  getNodesByBranch: (sessionId: string, branchId: string) => GraphNode[];
  resetGraphSession: () => void;
  calculateAutoLayout: (
    sessionId: string,
    parentNodeId: string
  ) => { x: number; y: number };
  calculateBranchLayout: (
    sessionId: string,
    sourceNodeId: string,
    branchId?: string
  ) => { x: number; y: number };
  updateNodePosition: (
    sessionId: string,
    nodeId: string,
    position: { x: number; y: number }
  ) => void;
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
}

export const useImageStore = create<ImageStreamState>((set, get) => ({
  // 초기 상태
  currentSession: null,
  currentGraphSession: null,
  selectedNodeId: null,
  selectedStepIndex: null,
  isGenerating: false,
  isPaused: false,
  isConnected: false,
  generationInterval: 1, // 기본값: 매 스텝마다
  feedbackRequest: null,
  currentFeedbackList: [],
  compositionState: {
    objects: [],
    bboxes: [],
    selectedObjectId: null,
    isConfigured: false,
  },

  // 이미지 생성 시작
  startGeneration: (prompt: string, interval?: number) => {
    const sessionId = `session_${Date.now()}`;
    const selectedInterval = interval || get().generationInterval;

    const newSession: ImageSession = {
      id: sessionId,
      prompt,
      totalSteps: 20,
      steps: [],
      isComplete: false,
      createdAt: Date.now(),
    };

    set({
      currentSession: newSession,
      selectedStepIndex: null, // 새 세션 시작 시 선택 초기화
      isGenerating: true,
      isPaused: false,
      isConnected: true,
      generationInterval: selectedInterval,
    });

    // Socket 시뮬레이션 시작
    get().simulateImageStream(sessionId, prompt, selectedInterval);
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
    set({
      isGenerating: false,
      isPaused: false,
      isConnected: false,
    });
  },

  // 세션 초기화 (새로운 이미지 생성을 위해)
  resetSession: () => {
    set({
      currentSession: null,
      selectedStepIndex: null,
      isGenerating: false,
      isPaused: false,
      isConnected: false,
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
      // 시뮬레이션 재시작 (현재 간격 설정 사용)
      get().simulateImageStream(
        state.currentSession.id,
        state.currentSession.prompt,
        state.generationInterval
      );
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

    // 피드백이 닫히면 생성이 계속되도록 시뮬레이션 재시작
    if (
      state.currentSession &&
      !state.currentSession.isComplete &&
      state.isGenerating
    ) {
      console.log("피드백 처리 완료, 생성 재개");
      // 시뮬레이션 재시작 (현재 세션과 간격 유지)
      get().simulateImageStream(
        state.currentSession.id,
        state.currentSession.prompt,
        state.generationInterval
      );
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
    bboxes?: BoundingBox[],
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

    // 세션 생성 시 구도 BBOX 정보 포함
    const sessionId = `session_${Date.now()}`;
    const selectedInterval = interval || state.generationInterval;

    const newSession: ImageSession = {
      id: sessionId,
      prompt,
      totalSteps: 20,
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
      isConnected: true,
      generationInterval: selectedInterval,
    });

    // Socket 시뮬레이션 시작
    get().simulateImageStream(sessionId, prompt, selectedInterval);
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

  // 그래프 세션 생성
  createGraphSession: (prompt: string, bboxes?: BoundingBox[]) => {
    const sessionId = `graph_session_${Date.now()}`;
    const rootNodeId = `node_prompt_${Date.now()}`;

    const rootNode: GraphNode = {
      id: rootNodeId,
      type: "prompt",
      data: {
        prompt,
        sessionId,
      },
      position: { x: 250, y: 100 }, // 초기 위치
    };

    // 메인 브랜치 생성 (최초 생성되는 체인)
    const mainBranchId = `branch_main_${Date.now()}`;
    const mainBranch: Branch = {
      id: mainBranchId,
      sourceNodeId: rootNodeId,
      feedback: [], // 메인 브랜치는 피드백 없음
      nodes: [],
    };

    const newGraphSession: GraphSession = {
      id: sessionId,
      rootNodeId,
      nodes: [rootNode],
      edges: [],
      branches: [mainBranch],
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
      currentGraphSession: newGraphSession,
      selectedNodeId: null,
      isGenerating: true,
      isPaused: false,
      isConnected: true,
    });

    return sessionId;
  },

  // 프롬프트 노드 추가
  addPromptNode: (
    sessionId: string,
    prompt: string,
    position: { x: number; y: number }
  ) => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return "";
    }

    const nodeId = `node_prompt_${Date.now()}`;
    const newNode: GraphNode = {
      id: nodeId,
      type: "prompt",
      data: {
        prompt,
        sessionId,
      },
      position,
    };

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: [...state.currentGraphSession.nodes, newNode],
      },
    });

    return nodeId;
  },

  // 자동 레이아웃: 부모 노드의 오른쪽에 가로로 배치
  calculateAutoLayout: (
    sessionId: string,
    parentNodeId: string
  ): { x: number; y: number } => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return { x: 0, y: 0 };
    }

    const parentNode = state.currentGraphSession.nodes.find(
      (n) => n.id === parentNodeId
    );
    if (!parentNode) {
      return { x: 0, y: 0 };
    }

    // 부모 노드에서 나가는 엣지 개수 계산 (이미 생성된 자식 노드 개수)
    const childEdges = state.currentGraphSession.edges.filter(
      (e) => e.source === parentNodeId
    );
    const childCount = childEdges.length;

    // 노드 간 간격 (가로)
    const horizontalSpacing = 400; // 노드 너비(300px) + 간격(20px)

    // 부모 노드의 오른쪽에 가로로 배치
    const newX = parentNode.position.x + horizontalSpacing;
    const newY = parentNode.position.y; // 같은 높이

    return { x: newX, y: newY };
  },

  // 브랜치용 자동 레이아웃: 실제 노드 위치와 크기를 계산해서 겹치지 않게 배치
  calculateBranchLayout: (
    sessionId: string,
    sourceNodeId: string,
    branchId?: string
  ): { x: number; y: number } => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return { x: 0, y: 0 };
    }

    const sourceNode = state.currentGraphSession.nodes.find(
      (n) => n.id === sourceNodeId
    );
    if (!sourceNode) {
      return { x: 0, y: 0 };
    }

    // 노드 간 간격 (가로)
    const horizontalSpacing = 400;
    // 노드 크기 (실제 노드 크기 고려)
    const nodeWidth = 300; // max-width
    const nodeHeight = 300; // 이미지(200px) + 패딩 + 라벨 등
    const verticalSpacing = 50; // 노드 간 최소 간격

    // 기본 위치: sourceNode의 오른쪽
    const newX = sourceNode.position.x + horizontalSpacing;

    // 같은 x 좌표(또는 비슷한 x 좌표)에 있는 노드들 찾기
    // 새로운 노드가 배치될 x 좌표 범위
    const xTolerance = 50; // x 좌표가 이 범위 내에 있으면 같은 열로 간주
    const targetXMin = newX - xTolerance;
    const targetXMax = newX + xTolerance;

    // 같은 열에 있는 노드들 찾기
    const nodesInSameColumn = state.currentGraphSession.nodes.filter((node) => {
      const nodeX = node.position.x;
      return nodeX >= targetXMin && nodeX <= targetXMax;
    });

    // sourceNode의 y 좌표를 기준으로 시작
    const baseY = sourceNode.position.y;

    // 겹치지 않는 y 좌표 찾기
    let newY = baseY;
    let offset = 0;
    const maxOffset = 20; // 최대 20개까지 시도

    while (offset <= maxOffset) {
      // 위쪽과 아래쪽 모두 확인
      const candidateYAbove = baseY - offset * (nodeHeight + verticalSpacing);
      const candidateYBelow = baseY + offset * (nodeHeight + verticalSpacing);

      // 위쪽 위치 확인
      const isOverlappingAbove = nodesInSameColumn.some((node) => {
        const nodeY = node.position.y;
        const nodeTop = nodeY - nodeHeight / 2;
        const nodeBottom = nodeY + nodeHeight / 2;
        const candidateTop = candidateYAbove - nodeHeight / 2;
        const candidateBottom = candidateYAbove + nodeHeight / 2;

        // 겹치는지 확인 (여유 공간 포함)
        return !(
          candidateBottom < nodeTop - verticalSpacing ||
          candidateTop > nodeBottom + verticalSpacing
        );
      });

      if (!isOverlappingAbove) {
        newY = candidateYAbove;
        break;
      }

      // 아래쪽 위치 확인
      const isOverlappingBelow = nodesInSameColumn.some((node) => {
        const nodeY = node.position.y;
        const nodeTop = nodeY - nodeHeight / 2;
        const nodeBottom = nodeY + nodeHeight / 2;
        const candidateTop = candidateYBelow - nodeHeight / 2;
        const candidateBottom = candidateYBelow + nodeHeight / 2;

        // 겹치는지 확인 (여유 공간 포함)
        return !(
          candidateBottom < nodeTop - verticalSpacing ||
          candidateTop > nodeBottom + verticalSpacing
        );
      });

      if (!isOverlappingBelow) {
        newY = candidateYBelow;
        break;
      }

      offset++;
    }

    // 최대 시도 횟수를 넘으면 강제로 배치
    if (offset > maxOffset) {
      newY = baseY + offset * (nodeHeight + verticalSpacing);
    }

    return { x: newX, y: newY };
  },

  // 이미지 노드 추가 (자동 레이아웃)
  addImageNode: (
    sessionId: string,
    parentNodeId: string,
    imageUrl: string,
    step: number,
    position?: { x: number; y: number }
  ) => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return "";
    }

    // position이 제공되지 않으면 자동 레이아웃 계산
    const nodePosition =
      position || get().calculateAutoLayout(sessionId, parentNodeId);

    const nodeId = `node_image_${Date.now()}`;
    const newNode: GraphNode = {
      id: nodeId,
      type: "image",
      data: {
        imageUrl,
        step,
        sessionId,
      },
      position: nodePosition,
    };

    const edgeId = `edge_${parentNodeId}_${nodeId}`;
    const newEdge: GraphEdge = {
      id: edgeId,
      source: parentNodeId,
      target: nodeId,
      type: "default",
    };

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: [...state.currentGraphSession.nodes, newNode],
        edges: [...state.currentGraphSession.edges, newEdge],
      },
    });

    return nodeId;
  },

  // 노드 선택
  selectNode: (nodeId: string | null) => {
    set({ selectedNodeId: nodeId });
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

  // 브랜치 생성
  // TODO: 백엔드 API 연동 필요
  // 1. POST /api/branch/create 호출하여 브랜치 생성 요청
  // 2. 피드백 정보를 서버로 전송
  // 3. 서버에서 브랜치 ID와 WebSocket URL 반환
  // 4. WebSocket 연결하여 브랜치의 이미지 스트림 수신
  // 5. 각 step 이미지를 받아서 addImageNodeToBranch로 노드 추가
  // 6. 여러 브랜치가 동시에 생성될 수 있으므로 병렬 처리 필요
  //
  // 현재는 프론트엔드에서만 브랜치를 생성하고, 이미지 스트림은 시뮬레이션으로 처리
  // 백엔드 연결 시:
  //   - import { createBranch as createBranchAPI } from '../api/branch';
  //   - const { branchId, websocketUrl } = await createBranchAPI(sessionId, sourceNodeId, feedback);
  //   - if (websocketUrl) { connectImageStream(sessionId, websocketUrl, ...); }
  createBranch: (
    sessionId: string,
    sourceNodeId: string,
    feedback: FeedbackRecord[]
  ) => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return "";
    }

    // TODO: 백엔드 API 호출
    // const { branchId, websocketUrl } = await createBranchAPI(sessionId, sourceNodeId, feedback);
    // if (websocketUrl) {
    //   connectImageStream(sessionId, websocketUrl, onImageStep, onError, onComplete);
    // }

    const branchId = `branch_${Date.now()}`;
    const newBranch: Branch = {
      id: branchId,
      sourceNodeId,
      feedback,
      nodes: [],
    };

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        branches: [...state.currentGraphSession.branches, newBranch],
      },
    });

    return branchId;
  },

  // 브랜치에 이미지 노드 추가 (자동 레이아웃)
  addImageNodeToBranch: (
    sessionId: string,
    branchId: string,
    imageUrl: string,
    step: number,
    position?: { x: number; y: number }
  ) => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return "";
    }

    const branch = state.currentGraphSession.branches.find(
      (b) => b.id === branchId
    );
    if (!branch) return "";

    // 브랜치의 마지막 노드에 연결 (또는 sourceNodeId에 연결)
    const lastNodeId =
      branch.nodes.length > 0
        ? branch.nodes[branch.nodes.length - 1]
        : branch.sourceNodeId;

    // position이 제공되지 않으면 자동 레이아웃 계산
    // 첫 번째 노드인 경우 브랜치 레이아웃 사용, 그 이후는 일반 레이아웃
    const nodePosition =
      position ||
      (branch.nodes.length === 0
        ? get().calculateBranchLayout(sessionId, branch.sourceNodeId, branchId)
        : get().calculateAutoLayout(sessionId, lastNodeId));

    const nodeId = `node_image_${Date.now()}`;
    const newNode: GraphNode = {
      id: nodeId,
      type: "image",
      data: {
        imageUrl,
        step,
        sessionId,
      },
      position: nodePosition,
    };

    const edgeId = `edge_${lastNodeId}_${nodeId}`;
    // 첫 번째 노드인 경우에만 피드백 정보 포함 (sourceNodeId에서 나오는 edge)
    const isFirstNode = branch.nodes.length === 0;
    const newEdge: GraphEdge = {
      id: edgeId,
      source: lastNodeId,
      target: nodeId,
      type: "branch",
      data: isFirstNode
        ? {
            branchId,
            feedback: branch.feedback,
          }
        : {
            branchId,
          },
    };

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: [...state.currentGraphSession.nodes, newNode],
        edges: [...state.currentGraphSession.edges, newEdge],
        branches: state.currentGraphSession.branches.map((b) =>
          b.id === branchId ? { ...b, nodes: [...b.nodes, nodeId] } : b
        ),
      },
    });

    return nodeId;
  },

  // 노드 ID로 노드 가져오기
  getNodeById: (sessionId: string, nodeId: string) => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return null;
    }
    return state.currentGraphSession.nodes.find((n) => n.id === nodeId) || null;
  },

  // 브랜치의 노드들 가져오기
  getNodesByBranch: (sessionId: string, branchId: string) => {
    const state = get();
    if (
      !state.currentGraphSession ||
      state.currentGraphSession.id !== sessionId
    ) {
      return [];
    }
    const branch = state.currentGraphSession.branches.find(
      (b) => b.id === branchId
    );
    if (!branch) return [];
    return state.currentGraphSession.nodes.filter((n) =>
      branch.nodes.includes(n.id)
    );
  },

  // 그래프 세션 초기화
  resetGraphSession: () => {
    set({
      currentGraphSession: null,
      selectedNodeId: null,
      isGenerating: false,
      isPaused: false,
      isConnected: false,
    });
  },

  // 그래프 구조에 맞는 이미지 스트림 시뮬레이션
  // TODO: 백엔드 WebSocket 연동 필요
  // 1. startImageGeneration에서 받은 websocketUrl로 WebSocket 연결
  // 2. 서버에서 image_step 이벤트를 받아서 addImageNodeToBranch로 노드 추가
  // 3. generation_complete 이벤트를 받아서 생성 완료 처리
  //
  // 현재는 시뮬레이션으로 더미 이미지를 생성
  // 백엔드 연결 시:
  //   - const ws = connectImageStream(sessionId, websocketUrl,
  //       (step) => { addImageNodeToBranch(sessionId, branchId, step.url, step.step); },
  //       (error) => { console.error(error); },
  //       () => { set({ isGenerating: false }); }
  //     );
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
      `그래프 이미지 생성 시작: ${prompt} (Session: ${sessionId}, Root: ${rootNodeId}, Branch: ${mainBranch.id})`
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
        `이미지 스텝 추가: ${currentStep}/${totalSteps} (간격: ${selectedInterval})`
      );

      // 완료되면 인터벌 정리
      if (currentStep >= totalSteps) {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        set({ isGenerating: false });
        console.log(`그래프 이미지 생성 완료: ${sessionId}`);
      }
    };

    // 첫 번째 스텝 추가
    addNextStep();

    // 나머지 스텝들을 주기적으로 추가
    intervalId = setInterval(addNextStep, 800); // 800ms마다 업데이트
  },

  // 브랜치용 이미지 스트림 시뮬레이션
  // TODO: 백엔드 WebSocket 연동 필요
  // 1. createBranch에서 받은 websocketUrl로 WebSocket 연결
  // 2. 서버에서 image_step 이벤트를 받아서 addImageNodeToBranch로 노드 추가
  // 3. generation_complete 이벤트를 받아서 생성 완료 처리
  // 4. 여러 브랜치가 동시에 생성될 수 있으므로 각 브랜치마다 독립적인 WebSocket 연결 필요
  //
  // 현재는 시뮬레이션으로 더미 이미지를 생성
  // 백엔드 연결 시:
  //   - const ws = connectImageStream(sessionId, websocketUrl,
  //       (step) => { addImageNodeToBranch(sessionId, branchId, step.url, step.step); },
  //       (error) => { console.error(error); },
  //       () => { console.log(`브랜치 생성 완료: ${branchId}`); }
  //     );
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
      console.error("브랜치를 찾을 수 없습니다:", branchId);
      return;
    }

    console.log(
      `브랜치 이미지 생성 시작: (Session: ${sessionId}, Branch: ${branchId})`
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
      const dummyImageUrl = `https://picsum.photos/512/512?random=${sessionId}_${branchId}&step=${currentStep}`;

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
          `브랜치 이미지 스텝 추가: ${currentStep}/${totalSteps} (간격: ${selectedInterval})`
        );
      }

      // 완료되면 인터벌 정리
      if (currentStep >= totalSteps) {
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        console.log(`브랜치 이미지 생성 완료: ${branchId}`);
      }
    };

    // 첫 번째 스텝 추가
    addNextStep();

    // 나머지 스텝들을 주기적으로 추가
    intervalId = setInterval(addNextStep, 800); // 800ms마다 업데이트
  },
}));
