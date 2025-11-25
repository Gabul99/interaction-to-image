import { create } from "zustand";
import { type FeedbackArea, type ObjectChip, type BoundingBox, type CompositionState, type FeedbackRecord } from "../types";

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

export interface ImageStreamState {
  // 현재 활성 세션
  currentSession: ImageSession | null;

  // 현재 선택된 스텝 (History에서 클릭한 스텝)
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
  startGenerationWithComposition: (prompt: string, bboxes?: BoundingBox[], interval?: number) => void;
  
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
}

export const useImageStore = create<ImageStreamState>((set, get) => ({
  // 초기 상태
  currentSession: null,
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
      "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", "#ef4444",
      "#f59e0b", "#eab308", "#84cc16", "#22c55e", "#10b981",
      "#14b8a6", "#06b6d4", "#0ea5e9", "#3b82f6", "#6366f1",
    ];
    const existingColors = state.compositionState.objects.map(obj => obj.color);
    const availableColor = colors.find(c => !existingColors.includes(c)) || colors[state.compositionState.objects.length % colors.length];
    
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
        objects: state.compositionState.objects.filter(obj => obj.id !== objectId),
        bboxes: state.compositionState.bboxes.filter(bbox => bbox.objectId !== objectId),
        selectedObjectId: state.compositionState.selectedObjectId === objectId ? null : state.compositionState.selectedObjectId,
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
        bboxes: state.compositionState.bboxes.map(bbox =>
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
        bboxes: state.compositionState.bboxes.filter(bbox => bbox.id !== bboxId),
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
  startGenerationWithComposition: (prompt: string, bboxes?: BoundingBox[], interval?: number) => {
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
      compositionBboxes: bboxes?.map(bbox => ({
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
}));
