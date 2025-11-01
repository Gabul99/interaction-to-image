import { useState, useRef, useEffect } from "react";
import styled from "styled-components";
import PromptInput from "./components/PromptInput";
import ImageViewer from "./components/ImageViewer";
import InteractionCanvas from "./components/InteractionCanvas";
import ImageHistoryNavigator from "./components/ImageHistoryNavigator";
import FeedbackPanel from "./components/FeedbackPanel";
import { useImageStore } from "./stores/imageStore";
import { sendFeedbackToServer, skipFeedback } from "./api/feedback";
import {
  type FeedbackData,
  type ToolMode,
  type InteractionData,
} from "./types";

const AppContainer = styled.div`
  height: 100vh;
  width: 100vw;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  position: relative;
  overflow: hidden;
  margin: 0;
  padding: 0;
`;

const ImageContainer = styled.div<{ hasFeedbackPanel: boolean }>`
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 60px;
  min-height: 60vh;
  width: ${(props) => (props.hasFeedbackPanel ? "calc(100% - 400px)" : "100%")};
  transition: width 0.3s ease;
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

function App() {
  // Zustand store에서 상태 가져오기
  const {
    currentSession,
    selectedStepIndex,
    isGenerating,
    startGeneration,
    selectStep,
    feedbackRequest,
    hideFeedbackRequest,
  } = useImageStore();

  // 피드백 패널 관련 상태
  const [toolMode, setToolMode] = useState<ToolMode>("none");
  const [selectedPoint, setSelectedPoint] = useState<
    { x: number; y: number } | undefined
  >();
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

  // 피드백 요청이 변경되면 선택 초기화
  useEffect(() => {
    if (!feedbackRequest?.visible) {
      setToolMode("none");
      setSelectedPoint(undefined);
      setSelectedBbox(undefined);
    }
  }, [feedbackRequest?.visible]);

  const handleSendPrompt = (newPrompt: string) => {
    console.log("프롬프트 전송:", newPrompt);
    startGeneration(newPrompt);
  };

  // 영역 선택에 따라 도구 모드 변경
  const handleAreaSelect = (area: string) => {
    console.log("handleAreaSelect 호출:", area, "현재 toolMode:", toolMode);

    if (area === "full") {
      setToolMode("none");
      setSelectedPoint(undefined);
      setSelectedBbox(undefined);
    } else if (area === "point") {
      // 포인팅 모드로 변경할 때만 도구 모드 변경
      // 이미 선택된 포인트가 있으면 유지
      if (toolMode !== "point") {
        setToolMode("point");
        setSelectedPoint(undefined); // 모드 변경 시에만 초기화
        setSelectedBbox(undefined);
      }
      // 이미 point 모드면 선택 상태 유지
    } else if (area === "bbox") {
      // BBOX 모드로 변경할 때만 도구 모드 변경
      // 이미 선택된 BBOX가 있으면 유지
      if (toolMode !== "bbox") {
        setToolMode("bbox");
        setSelectedPoint(undefined);
        setSelectedBbox(undefined); // 모드 변경 시에만 초기화
      }
      // 이미 bbox 모드면 선택 상태 유지
    }
  };

  // InteractionCanvas에서 상호작용 발생 시
  const handleInteraction = (data: InteractionData) => {
    console.log("handleInteraction 호출:", data);
    if (data.type === "point") {
      const point = { x: data.x, y: data.y };
      console.log("포인트 선택:", point);
      setSelectedPoint(point);
    } else if (data.type === "bbox" && data.width && data.height) {
      const bbox = {
        x: data.x,
        y: data.y,
        width: data.width,
        height: data.height,
      };
      console.log("BBOX 선택:", bbox);
      setSelectedBbox(bbox);
    } else {
      console.warn("BBOX 데이터가 불완전:", data);
    }
  };

  const handleFeedbackSubmit = async (feedback: FeedbackData) => {
    try {
      await sendFeedbackToServer(feedback);
      console.log("피드백 전송 성공");
      hideFeedbackRequest();
    } catch (error) {
      console.error("피드백 전송 실패:", error);
      // 에러 처리 (예: 토스트 메시지 표시)
    }
  };

  const handleFeedbackSkip = async () => {
    try {
      await skipFeedback();
      console.log("피드백 건너뛰기 성공");
      hideFeedbackRequest();
    } catch (error) {
      console.error("피드백 건너뛰기 실패:", error);
      // 에러 처리 (예: 토스트 메시지 표시)
    }
  };

  // 피드백 패널이 열려있을 때만 InteractionCanvas 활성화
  const isFeedbackPanelOpen = feedbackRequest?.visible ?? false;

  return (
    <AppContainer>
      <PromptInput onSendPrompt={handleSendPrompt} disabled={isGenerating} />

      <ImageContainer hasFeedbackPanel={isFeedbackPanelOpen}>
        <ImageViewer
          imageUrl={currentImage || undefined}
          onImageLoad={() => console.log("이미지 로드 완료")}
          imageRef={imageRef}
        />

        {/* 피드백 패널이 열려있을 때만 InteractionCanvas 활성화 */}
        {currentImage && isFeedbackPanelOpen && (
          <InteractionCanvas
            toolMode={toolMode}
            disabled={false}
            onInteraction={handleInteraction}
            onClearSelection={() => {
              if (toolMode === "point") {
                setSelectedPoint(undefined);
              } else if (toolMode === "bbox") {
                setSelectedBbox(undefined);
              }
            }}
            imageRef={imageRef}
          />
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

      <ImageHistoryNavigator />

      {/* 최신 이미지로 가기 버튼 */}
      <LatestImageButton
        visible={
          !!(
            selectedStepIndex !== null &&
            currentSession?.steps &&
            currentSession.steps.length > 0 &&
            !isFeedbackPanelOpen
          )
        }
        onClick={() => selectStep(null)}
      >
        최신 이미지
      </LatestImageButton>

      {/* 피드백 패널 */}
      {feedbackRequest && (
        <FeedbackPanel
          visible={feedbackRequest.visible}
          onSubmit={handleFeedbackSubmit}
          onSkip={handleFeedbackSkip}
          area={feedbackRequest.area}
          selectedPoint={selectedPoint}
          selectedBbox={selectedBbox}
          onAreaSelect={handleAreaSelect}
        />
      )}
    </AppContainer>
  );
}

export default App;
