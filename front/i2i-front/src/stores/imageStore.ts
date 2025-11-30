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

  // 그래프 세션 관련 액션
  createGraphSession: (
    prompt: string,
    sessionId: string,
    rootNodeId?: string,
    bboxes?: BoundingBox[],
    sketchLayers?: SketchLayer[]
  ) => string;
  addImageNode: (
    sessionId: string,
    parentNodeId: string,
    imageUrl: string,
    step: number,
    position: { x: number; y: number },
    nodeId?: string
  ) => string; // nodeId 반환
  addImageNodeToBranch: (
    sessionId: string,
    branchId: string,
    imageUrl: string,
    step: number,
    position?: { x: number; y: number },
    nodeId?: string
  ) => string; // nodeId 반환
  updateNodePosition: (
    sessionId: string,
    nodeId: string,
    position: { x: number; y: number }
  ) => void;
  selectNode: (nodeId: string | null) => void;
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

  // 그래프 세션 생성
  createGraphSession: (
    prompt: string,
    sessionId: string,
    rootNodeId?: string,
    bboxes?: BoundingBox[],
    sketchLayers?: SketchLayer[]
  ) => {
    // 프롬프트 노드 생성 (루트 노드)
    const promptNodeId = rootNodeId || `node_prompt_${Date.now()}`;
    const rootNode: GraphNode = {
      id: promptNodeId,
      type: "prompt",
      data: {
        prompt,
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
      position: { x: 400, y: 50 },
    };

    // 메인 브랜치 생성
    const mainBranch: Branch = {
      id: `branch_main_${Date.now()}`,
      sourceNodeId: promptNodeId,
      feedback: [],
      nodes: [],
    };

    const graphSession: GraphSession = {
      id: sessionId,
      nodes: [rootNode],
      edges: [],
      branches: [mainBranch],
    };

    set({ currentGraphSession: graphSession });
    console.log("[ImageStore] 그래프 세션 생성:", sessionId);

    return sessionId;
  },

  // 이미지 노드 추가 (nodeId는 선택적, 제공되지 않으면 자동 생성)
  addImageNode: (
    sessionId: string,
    parentNodeId: string,
    imageUrl: string,
    step: number,
    position: { x: number; y: number },
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

    // 백엔드에서 보낸 nodeId를 사용하거나 새로 생성
    // 백엔드에서 보낸 nodeId 형식: node_image_{session_id}_{step_idx}
    const finalNodeId = nodeId || `node_image_${sessionId}_${step}`;

    // 이미 존재하는 노드인지 확인
    const existingNode = state.currentGraphSession.nodes.find(
      (n) => n.id === finalNodeId
    );
    if (existingNode) {
      console.log(
        `[ImageStore] 노드가 이미 존재합니다: ${finalNodeId}, 이미지 URL 업데이트`
      );
      // 기존 노드의 이미지 URL 업데이트
      set({
        currentGraphSession: {
          ...state.currentGraphSession,
          nodes: state.currentGraphSession.nodes.map((n) =>
            n.id === finalNodeId ? { ...n, data: { ...n.data, imageUrl } } : n
          ),
        },
      });
      return finalNodeId;
    }

    const newNode: GraphNode = {
      id: finalNodeId,
      type: "image",
      data: { imageUrl, step, sessionId },
      position,
    };

    const edgeId = `edge_${parentNodeId}_${finalNodeId}`;
    const newEdge: GraphEdge = {
      id: edgeId,
      source: parentNodeId,
      target: finalNodeId,
      type: "default",
    };

    // 메인 브랜치 찾기 (sourceNodeId가 없는 브랜치)
    const mainBranch = state.currentGraphSession.branches.find(
      (b) => !b.sourceNodeId
    );

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: [...state.currentGraphSession.nodes, newNode],
        edges: [...state.currentGraphSession.edges, newEdge],
        branches: mainBranch
          ? state.currentGraphSession.branches.map((b) =>
              b.id === mainBranch.id
                ? { ...b, nodes: [...b.nodes, finalNodeId] }
                : b
            )
          : state.currentGraphSession.branches,
      },
    });

    console.log(
      `[ImageStore] 이미지 노드 추가: ${finalNodeId}, 부모: ${parentNodeId}, 스텝: ${step}`
    );
    return finalNodeId;
  },

  // 브랜치에 이미지 노드 추가 (nodeId와 position은 선택적)
  addImageNodeToBranch: (
    sessionId: string,
    branchId: string,
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
      (b) => b.id === branchId
    );
    if (!branch) {
      console.warn("[ImageStore] 브랜치를 찾을 수 없습니다:", branchId);
      return "";
    }

    // nodeId가 제공되지 않으면 자동 생성
    const finalNodeId =
      nodeId || `node_image_${sessionId}_${step}_${Date.now()}`;

    // position이 제공되지 않으면 자동 계산
    let finalPosition: { x: number; y: number };
    if (position) {
      finalPosition = position;
    } else {
      // 브랜치의 마지막 노드 찾기
      const branchNodes = state.currentGraphSession.nodes.filter((n) =>
        branch.nodes.includes(n.id)
      );
      const lastNode = branchNodes[branchNodes.length - 1];

      if (lastNode) {
        // 마지막 노드의 오른쪽에 배치 (가로로 grow)
        const horizontalSpacing = 400; // 노드 너비(300px) + 간격(100px)
        finalPosition = {
          x: lastNode.position.x + horizontalSpacing,
          y: lastNode.position.y, // 같은 y 좌표 유지
        };
      } else {
        // 첫 번째 노드인 경우: 소스 노드의 오른쪽에 배치
        const sourceNode = state.currentGraphSession.nodes.find(
          (n) => n.id === branch.sourceNodeId
        );
        const horizontalSpacing = 400;
        finalPosition = sourceNode
          ? {
              x: sourceNode.position.x + horizontalSpacing,
              y: sourceNode.position.y,
            }
          : { x: 0, y: 0 };
      }
    }

    // 부모 노드 ID 찾기
    const branchNodes = state.currentGraphSession.nodes.filter((n) =>
      branch.nodes.includes(n.id)
    );
    const lastNode = branchNodes[branchNodes.length - 1];
    const parentNodeId = lastNode ? lastNode.id : branch.sourceNodeId;

    // 이미 존재하는 노드인지 확인
    const existingNode = state.currentGraphSession.nodes.find(
      (n) => n.id === finalNodeId
    );
    if (existingNode) {
      console.log(
        `[ImageStore] 노드가 이미 존재합니다: ${finalNodeId}, 이미지 URL 업데이트`
      );
      // 기존 노드의 이미지 URL 업데이트
      set({
        currentGraphSession: {
          ...state.currentGraphSession,
          nodes: state.currentGraphSession.nodes.map((n) =>
            n.id === finalNodeId ? { ...n, data: { ...n.data, imageUrl } } : n
          ),
        },
      });
      return finalNodeId;
    }

    console.log(
      `[ImageStore] 새 노드 생성: nodeId=${finalNodeId}, imageUrl 길이=${
        imageUrl ? imageUrl.length : 0
      }, step=${step}`
    );
    const newNode: GraphNode = {
      id: finalNodeId,
      type: "image",
      data: { imageUrl, step, sessionId },
      position: finalPosition,
    };

    const edgeId = `edge_${parentNodeId}_${finalNodeId}`;
    const newEdge: GraphEdge = {
      id: edgeId,
      source: parentNodeId,
      target: finalNodeId,
      type: "branch",
      data: { branchId },
    };

    set({
      currentGraphSession: {
        ...state.currentGraphSession,
        nodes: [...state.currentGraphSession.nodes, newNode],
        edges: [...state.currentGraphSession.edges, newEdge],
        branches: state.currentGraphSession.branches.map((b) =>
          b.id === branchId ? { ...b, nodes: [...b.nodes, finalNodeId] } : b
        ),
      },
    });

    console.log(
      `[ImageStore] 브랜치 이미지 노드 추가: ${finalNodeId}, 브랜치: ${branchId}, 부모: ${parentNodeId}, 스텝: ${step}`
    );
    return finalNodeId;
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

  // 노드 선택
  selectNode: (nodeId: string | null) => {
    set({ selectedNodeId: nodeId });
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
}));
