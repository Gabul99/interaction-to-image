import { useState, useRef, useEffect } from "react";
import styled from "styled-components";
import PromptInput from "./components/PromptInput";
import ImageViewer from "./components/ImageViewer";
import ImageHistoryNavigator from "./components/ImageHistoryNavigator";
import FeedbackPanel from "./components/FeedbackPanel";
import ObjectChipList from "./components/ObjectChipList";
import CompositionCanvas from "./components/CompositionCanvas";
import BboxOverlay from "./components/BboxOverlay";
import { useImageStore } from "./stores/imageStore";
import { requestObjectList, startImageGeneration } from "./api/composition";
import { submitFeedbacks, skipFeedback } from "./api/feedback";
import { type FeedbackData } from "./types";

const AppContainer = styled.div`
  height: 100vh;
  width: 100vw;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  position: relative;
  overflow: hidden;
  margin: 0;
  padding: 0;
`;

const MainContentContainer = styled.div<{ hasFeedbackPanel: boolean }>`
  position: relative;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding-top: 60px;
  min-height: calc(100vh - 60px);
  width: ${(props) => (props.hasFeedbackPanel ? "calc(100% - 400px)" : "100%")};
  transition: width 0.3s ease;
  gap: 0;
`;

const ImageContainer = styled.div`
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  width: fit-content;
  min-width: 512px;
`;

const ProgressContainer = styled.div`
  margin-top: 12px;
  width: 100%;
  max-width: 512px;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: rgba(55, 65, 81, 0.5);
  border-radius: 4px;
  overflow: hidden;
`;

const ProgressFill = styled.div<{ progress: number }>`
  height: 100%;
  width: ${(props) => props.progress}%;
  background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
  border-radius: 4px;
  transition: width 0.3s ease;
`;

const ProgressText = styled.div`
  margin-top: 8px;
  text-align: center;
  font-size: 12px;
  color: #9ca3af;
`;

const LatestImageButton = styled.button<{ visible: boolean }>`
  position: fixed;
  right: 20px;
  top: 50%;
  transform: translateY(-50%);
  z-index: 1000;
  background: rgba(26, 26, 46, 0.9);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 12px 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: #f9fafb;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  opacity: ${(props) => (props.visible ? 1 : 0)};
  pointer-events: ${(props) => (props.visible ? "auto" : "none")};

  &:hover {
    background: rgba(26, 26, 46, 1);
    border-color: #6366f1;
    transform: translateY(-50%) scale(1.05);
  }
`;

const CompleteButton = styled.button<{ visible: boolean }>`
  position: fixed;
  bottom: 40px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 1000;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 16px 32px;
  box-shadow: 0 8px 32px rgba(99, 102, 241, 0.4);
  border: none;
  color: white;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  opacity: ${(props) => (props.visible ? 1 : 0)};
  pointer-events: ${(props) => (props.visible ? "auto" : "none")};

  &:hover {
    transform: translateX(-50%) translateY(-2px);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.5);
  }

  &:active {
    transform: translateX(-50%) translateY(0);
  }
`;

function App() {
  // Zustand store에서 상태 가져오기
  const {
    currentSession,
    selectedStepIndex,
    isGenerating,
    startGenerationWithComposition,
    selectStep,
    resetSession,
    feedbackRequest,
    hideFeedbackRequest,
    compositionState,
    getCurrentCompositionBboxes,
    setObjectList,
    addObject,
    removeObject,
    selectObject,
    addBbox,
    updateBbox,
    removeBbox,
    clearComposition,
    currentFeedbackList,
    clearCurrentFeedbackList,
    addFeedbackToBboxHistory,
  } = useImageStore();

  // 피드백 패널 관련 상태
  const [selectedBbox, setSelectedBbox] = useState<
    { x: number; y: number; width: number; height: number } | undefined
  >();

  // selectedBbox 변경 시 디버깅
  useEffect(() => {
    console.log("App - selectedBbox 상태 변경:", selectedBbox);
  }, [selectedBbox]);

  // 현재 표시할 이미지와 진행률 계산
  const currentImage = (() => {
    if (!currentSession?.steps || currentSession.steps.length === 0)
      return null;

    // 선택된 스텝이 유효한 범위 내에 있는지 확인
    const maxIndex = currentSession.steps.length - 1;
    let targetIndex: number;

    if (
      selectedStepIndex !== null &&
      selectedStepIndex >= 0 &&
      selectedStepIndex <= maxIndex
    ) {
      // 사용자가 특정 스텝을 선택한 경우
      targetIndex = selectedStepIndex;
    } else {
      // 선택된 스텝이 없거나 유효하지 않은 경우 최신 스텝으로
      targetIndex = maxIndex;
    }

    console.log("이미지 선택 로직:", {
      selectedStepIndex,
      maxIndex,
      targetIndex,
      totalSteps: currentSession.steps.length,
      targetUrl: currentSession.steps[targetIndex]?.url,
      isValidSelection:
        selectedStepIndex !== null &&
        selectedStepIndex >= 0 &&
        selectedStepIndex <= maxIndex,
    });

    return currentSession.steps[targetIndex]?.url || null;
  })();

  const progress =
    currentSession && currentSession.steps
      ? (currentSession.steps.length / currentSession.totalSteps) * 100
      : 0;

  const imageRef = useRef<HTMLImageElement>(null);
  const placeholderRef = useRef<HTMLDivElement>(null);

  // 피드백 요청이 변경되면 선택 초기화
  useEffect(() => {
    if (!feedbackRequest?.visible) {
      setSelectedBbox(undefined);
      setSelectedBboxIdForFeedback(null);
    }
  }, [feedbackRequest?.visible]);

  const [isLoadingObjects, setIsLoadingObjects] = useState(false);
  const [currentPrompt, setCurrentPrompt] = useState<string>("");

  const handleSendPrompt = async (newPrompt: string) => {
    console.log("프롬프트 전송:", newPrompt);

    // 이전 상태 초기화
    setCurrentPrompt(newPrompt);
    setIsLoadingObjects(true);
    clearComposition();

    // 피드백 관련 상태 초기화
    setSelectedBbox(undefined);
    setSelectedBboxIdForFeedback(null);

    // 이전 세션 초기화 (새로운 이미지 생성을 위해)
    resetSession();

    try {
      const objects = await requestObjectList(newPrompt);
      setObjectList(objects);
      setIsLoadingObjects(false);
    } catch (error) {
      console.error("객체 리스트 요청 실패:", error);
      setIsLoadingObjects(false);
      // 에러 발생 시에도 기본 객체 리스트로 진행하거나 에러 처리
    }
  };

  const handleCompleteComposition = async () => {
    if (!currentPrompt) return;

    try {
      // API를 통해 이미지 생성 시작
      const bboxes =
        compositionState.bboxes.length > 0
          ? compositionState.bboxes.map((bbox) => ({
              objectId: bbox.objectId,
              x: bbox.x,
              y: bbox.y,
              width: bbox.width,
              height: bbox.height,
            }))
          : undefined;

      const { sessionId, websocketUrl } = await startImageGeneration(
        currentPrompt,
        compositionState.objects.length > 0
          ? compositionState.objects
          : undefined,
        bboxes
      );

      // Store를 통해 이미지 생성 시작 (WebSocket 연결은 추후 구현)
      // 현재는 mockup 모드이므로 기존 시뮬레이션 사용
      startGenerationWithComposition(
        currentPrompt,
        compositionState.bboxes.length > 0 ? compositionState.bboxes : undefined
      );

      console.log("[App] 이미지 생성 세션 시작:", sessionId, websocketUrl);
      setCurrentPrompt("");
    } catch (error) {
      console.error("[App] 이미지 생성 시작 실패:", error);
    }
  };

  // BBOX 선택 상태 관리 (피드백 패널용)
  const [selectedBboxIdForFeedback, setSelectedBboxIdForFeedback] = useState<
    string | null
  >(null);

  // BBOX 클릭 시 선택
  const handleBboxSelectForFeedback = (bboxId: string | null) => {
    setSelectedBboxIdForFeedback(bboxId);
    if (bboxId) {
      const compositionBboxes = getCurrentCompositionBboxes();
      const bbox = compositionBboxes?.find((b) => b.id === bboxId);
      if (bbox) {
        setSelectedBbox({
          x: bbox.x,
          y: bbox.y,
          width: bbox.width,
          height: bbox.height,
        });
      }
    } else {
      setSelectedBbox(undefined);
    }
  };

  // 피드백 제출 핸들러
  const handleFeedbackSubmit = async () => {
    if (!currentSession) {
      console.error("[App] 세션이 없어 피드백을 제출할 수 없습니다.");
      return;
    }

    try {
      // currentFeedbackList의 각 피드백을 FeedbackData로 변환
      const feedbacks: FeedbackData[] = currentFeedbackList.map((feedback) => {
        // imageUrl이 있으면 File 객체로 변환해야 하는데,
        // 현재는 base64 URL이므로 일단 text만 전송하거나
        // 실제 구현에서는 File 객체를 저장해두어야 함
        const feedbackData: FeedbackData = {
          area: feedback.area,
          type: feedback.type,
          text: feedback.text,
          // image는 현재 base64 URL이므로 실제 구현에서는 File 객체를 저장해야 함
          // image: feedback.imageUrl ? ... : undefined,
          bbox: feedback.bbox,
          point: feedback.point,
        };
        return feedbackData;
      });

      // 여러 피드백을 한번에 서버로 전송
      await submitFeedbacks(currentSession.id, feedbacks);

      // BBOX 피드백인 경우 BBOX별 히스토리에 저장
      for (const feedback of currentFeedbackList) {
        if (feedback.area === "bbox" && feedback.bboxId) {
          addFeedbackToBboxHistory(feedback.bboxId, feedback);
        }
      }

      console.log("[App] 피드백 제출 성공");

      // 피드백 리스트 초기화
      clearCurrentFeedbackList();

      // 피드백 패널 닫기
      hideFeedbackRequest();
    } catch (error) {
      console.error("[App] 피드백 제출 실패:", error);
      // 에러 처리 (예: 토스트 메시지 표시)
    }
  };

  // 피드백 건너뛰기 핸들러
  const handleFeedbackSkip = async () => {
    if (!currentSession) {
      console.error("[App] 세션이 없어 피드백을 건너뛸 수 없습니다.");
      return;
    }

    try {
      await skipFeedback(currentSession.id);
      console.log("[App] 피드백 건너뛰기 성공");

      // 피드백 리스트 초기화
      clearCurrentFeedbackList();

      // 피드백 패널 닫기
      hideFeedbackRequest();
    } catch (error) {
      console.error("[App] 피드백 건너뛰기 실패:", error);
      // 에러 처리 (예: 토스트 메시지 표시)
    }
  };

  // 피드백 패널이 열려있을 때만 InteractionCanvas 활성화
  const isFeedbackPanelOpen = feedbackRequest?.visible ?? false;

  return (
    <AppContainer>
      <PromptInput onSendPrompt={handleSendPrompt} disabled={isGenerating} />

      <MainContentContainer hasFeedbackPanel={isFeedbackPanelOpen}>
        {/* 객체 리스트 표시 (세션이 없을 때만) */}
        {compositionState.objects.length > 0 &&
          !isGenerating &&
          !currentSession && (
            <ObjectChipList
              objects={compositionState.objects}
              selectedObjectId={compositionState.selectedObjectId}
              onSelectObject={selectObject}
              onAddObject={addObject}
              onRemoveObject={removeObject}
            />
          )}

        <ImageContainer>
          {/* 구도 설정 모드: CompositionCanvas 사용 (세션이 없을 때만) */}
          {compositionState.objects.length > 0 &&
            !isGenerating &&
            !currentSession && (
              <>
                <ImageViewer
                  imageUrl={undefined}
                  onImageLoad={() => console.log("이미지 로드 완료")}
                  imageRef={imageRef}
                  placeholderRef={placeholderRef}
                />
                <CompositionCanvas
                  bboxes={compositionState.bboxes}
                  selectedObjectId={compositionState.selectedObjectId}
                  selectedObjectColor={
                    compositionState.objects.find(
                      (obj) => obj.id === compositionState.selectedObjectId
                    )?.color || null
                  }
                  onAddBbox={addBbox}
                  onUpdateBbox={updateBbox}
                  onRemoveBbox={removeBbox}
                  imageRef={imageRef}
                  placeholderRef={placeholderRef}
                />
              </>
            )}

          {/* 이미지 생성 모드 및 완료 후: 기존 ImageViewer 사용 */}
          {(isGenerating || currentSession) && (
            <>
              <ImageViewer
                imageUrl={currentImage || undefined}
                onImageLoad={() => console.log("이미지 로드 완료")}
                imageRef={imageRef}
              />

              {/* 구도 BBOX 오버레이 표시 (이미지가 있을 때만) */}
              {currentImage &&
                (() => {
                  const compositionBboxes = getCurrentCompositionBboxes();
                  if (!compositionBboxes || compositionBboxes.length === 0) {
                    return null;
                  }
                  return (
                    <BboxOverlay
                      bboxes={compositionBboxes}
                      objects={compositionState.objects}
                      imageRef={imageRef}
                      selectedBboxId={selectedBboxIdForFeedback}
                      onBboxClick={
                        isFeedbackPanelOpen
                          ? (bboxId) => {
                              handleBboxSelectForFeedback(bboxId);
                            }
                          : undefined
                      }
                      onClearSelection={
                        isFeedbackPanelOpen
                          ? () => {
                              handleBboxSelectForFeedback(null);
                            }
                          : undefined
                      }
                    />
                  );
                })()}

              {/* 피드백 패널이 열려있을 때는 InteractionCanvas 비활성화 */}
              {/* (피드백은 FeedbackPanel에서 직접 관리) */}
            </>
          )}

          {/* 객체 로딩 중 표시 */}
          {isLoadingObjects && (
            <div style={{ color: "#9ca3af", marginTop: "20px" }}>
              객체 리스트를 생성하는 중...
            </div>
          )}

          {/* 이미지 생성 진행률 표시 */}
          {currentSession && (
            <ProgressContainer>
              <ProgressBar>
                <ProgressFill progress={progress} />
              </ProgressBar>
              <ProgressText>
                {isGenerating ? (
                  <>
                    이미지 생성 중... {currentSession.steps.length}/
                    {currentSession.totalSteps} 스텝
                  </>
                ) : (
                  <>
                    완료됨 - {currentSession.steps.length}/
                    {currentSession.totalSteps} 스텝
                  </>
                )}
              </ProgressText>
            </ProgressContainer>
          )}
        </ImageContainer>
      </MainContentContainer>

      <ImageHistoryNavigator />

      {/* 최신 이미지로 가기 버튼 */}
      <LatestImageButton
        visible={
          !!(
            selectedStepIndex !== null &&
            currentSession?.steps &&
            currentSession.steps.length > 0 &&
            !isFeedbackPanelOpen &&
            isGenerating
          )
        }
        onClick={() => selectStep(null)}
      >
        최신 이미지
      </LatestImageButton>

      {/* 구도 설정 완료 버튼 (세션이 없을 때만) */}
      <CompleteButton
        visible={
          compositionState.objects.length > 0 &&
          !isGenerating &&
          !isLoadingObjects &&
          !currentSession
        }
        onClick={handleCompleteComposition}
      >
        {compositionState.bboxes.length > 0
          ? "구도 설정 완료 및 이미지 생성 시작"
          : "구도 없이 이미지 생성 시작"}
      </CompleteButton>

      {/* 피드백 패널 */}
      {feedbackRequest && (
        <FeedbackPanel
          visible={feedbackRequest.visible}
          onClose={hideFeedbackRequest}
          onSubmit={handleFeedbackSubmit}
          onSkip={handleFeedbackSkip}
          objects={compositionState.objects}
          compositionBboxes={getCurrentCompositionBboxes() || []}
          onBboxSelect={handleBboxSelectForFeedback}
          selectedBboxId={selectedBboxIdForFeedback}
        />
      )}
    </AppContainer>
  );
}

export default App;
