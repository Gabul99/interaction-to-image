import { useState, useRef } from "react";
import styled from "styled-components";
import PromptInput from "./components/PromptInput";
import ImageViewer from "./components/ImageViewer";
import InteractionCanvas from "./components/InteractionCanvas";
import FeedbackTooltip, {
  type FeedbackData,
} from "./components/FeedbackTooltip";
import Toolbar from "./components/Toolbar";
import ImageHistoryNavigator from "./components/ImageHistoryNavigator";
import { type ToolMode, type InteractionData } from "./types";
import { useImageStore } from "./stores/imageStore";

const AppContainer = styled.div`
  height: 100vh;
  width: 100vw;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  position: relative;
  overflow: hidden;
  margin: 0;
  padding: 0;
`;

const ImageContainer = styled.div`
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 60px;
  min-height: 60vh;
  width: 100%;
`;

const GlobalFeedbackContainer = styled.div`
  margin-top: 20px;
  width: 100%;
  max-width: 512px; /* 이미지 최대 크기에 맞춤 */
  background: rgba(26, 26, 46, 0.9);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const GlobalFeedbackToggleButton = styled.button`
  width: 100%;
  padding: 12px 16px;
  background: rgba(55, 65, 81, 0.5);
  color: #f9fafb;
  border: 2px solid #374151;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(55, 65, 81, 0.7);
    border-color: #6366f1;
  }
`;

const GlobalFeedbackInputContainer = styled.div`
  display: flex;
  gap: 12px;
  align-items: flex-end;
`;

const GlobalFeedbackCancelButton = styled.button`
  padding: 12px 16px;
  background: transparent;
  color: #9ca3af;
  border: 2px solid #374151;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;

  &:hover {
    background: rgba(55, 65, 81, 0.3);
    border-color: #6b7280;
    color: #d1d5db;
  }
`;

const GlobalFeedbackInput = styled.textarea`
  flex: 1;
  min-height: 50px;
  max-height: 120px;
  padding: 12px 16px;
  border: 2px solid #374151;
  border-radius: 8px;
  font-size: 14px;
  background: rgba(55, 65, 81, 0.5);
  color: #f9fafb;
  resize: vertical;
  outline: none;
  transition: border-color 0.2s ease;

  &:focus {
    border-color: #6366f1;
  }

  &::placeholder {
    color: #9ca3af;
  }
`;

const GlobalSendButton = styled.button`
  padding: 12px 20px;
  background: #6366f1;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;

  &:hover {
    background: #5b5bd6;
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
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

const ControlButtons = styled.div`
  position: fixed;
  right: 20px;
  top: 20px;
  z-index: 1000;
  display: flex;
  gap: 12px;
`;

const ControlButton = styled.button<{ variant: "pause" | "resume" }>`
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
  min-width: 80px;

  background-color: ${(props) =>
    props.variant === "pause"
      ? "rgba(239, 68, 68, 0.2)"
      : "rgba(34, 197, 94, 0.2)"};
  border-color: ${(props) =>
    props.variant === "pause"
      ? "rgba(239, 68, 68, 0.5)"
      : "rgba(34, 197, 94, 0.5)"};

  &:hover {
    background-color: ${(props) =>
      props.variant === "pause"
        ? "rgba(239, 68, 68, 0.3)"
        : "rgba(34, 197, 94, 0.3)"};
    border-color: ${(props) =>
      props.variant === "pause"
        ? "rgba(239, 68, 68, 0.8)"
        : "rgba(34, 197, 94, 0.8)"};
    transform: scale(1.05);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
`;

function App() {
  // Zustand store에서 상태 가져오기
  const {
    currentSession,
    selectedStepIndex,
    isGenerating,
    isPaused,
    startGeneration,
    selectStep,
    pauseGeneration,
    resumeGeneration,
  } = useImageStore();

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

  // 로컬 상태 (UI 관련)
  const [toolMode, setToolMode] = useState<ToolMode>("none");
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const [lastInteraction, setLastInteraction] =
    useState<InteractionData | null>(null);
  const [globalFeedback, setGlobalFeedback] = useState("");
  const [showGlobalFeedback, setShowGlobalFeedback] = useState(false);

  const imageRef = useRef<HTMLImageElement>(null);

  const handleSendPrompt = (newPrompt: string, interval: number) => {
    console.log("프롬프트 전송:", newPrompt, "간격:", interval);
    startGeneration(newPrompt, interval);
  };

  const handleInteraction = (data: InteractionData) => {
    console.log("handleInteraction 호출됨:", data);
    setLastInteraction(data);

    // 툴팁 위치 계산 (이미지 기준)
    if (imageRef.current) {
      const rect = imageRef.current.getBoundingClientRect();

      if (data.type === "point") {
        // 포인트의 경우 클릭한 위치 위에 표시
        const x = rect.left + data.x * rect.width;
        const y = rect.top + data.y * rect.height - 10; // 포인트 위쪽에 10px 여백
        console.log("포인트 툴팁 위치:", { x, y });
        setTooltipPosition({ x, y });
      } else if (data.type === "bbox" && data.width && data.height) {
        // 바운딩 박스의 경우 오른쪽에 표시
        // data.x, data.y는 이미지 내에서의 상대 좌표 (0~1)
        // data.width, data.height는 상대 크기 (0~1)
        const boxRightX = data.x + data.width; // 박스의 오른쪽 끝 (상대 좌표)
        const rightX = rect.left + boxRightX * rect.width + 15; // 박스 오른쪽 + 15px 여백
        const centerY = rect.top + (data.y + data.height / 2) * rect.height;
        console.log("바운딩 박스 툴팁 위치:", {
          boxRightX,
          rightX,
          centerY,
          rect: {
            left: rect.left,
            width: rect.width,
            top: rect.top,
            height: rect.height,
          },
        });
        setTooltipPosition({ x: rightX, y: centerY });
      }

      console.log("툴팁 표시 설정");
      setTooltipVisible(true);
    }
  };

  const handleFeedbackSubmit = (feedback: FeedbackData) => {
    console.log("피드백 제출:", {
      interaction: lastInteraction,
      feedback,
    });

    // 실제로는 서버로 피드백을 전송
    setTooltipVisible(false);
    setLastInteraction(null);
  };

  const handleTooltipClose = () => {
    setTooltipVisible(false);
    setLastInteraction(null);
    // 툴팁을 닫을 때만 모드를 none으로 변경
    setToolMode("none");
  };

  const handleGlobalFeedbackSubmit = () => {
    if (globalFeedback.trim()) {
      console.log("전체 피드백 제출:", {
        type: "global",
        content: globalFeedback.trim(),
        imageUrl: currentImage,
      });

      // 실제로는 서버로 전체 피드백을 전송
      setGlobalFeedback("");
      setShowGlobalFeedback(false); // 전송 후 입력창 닫기
    }
  };

  const handleGlobalFeedbackKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      handleGlobalFeedbackSubmit();
    }
  };

  return (
    <AppContainer>
      <PromptInput onSendPrompt={handleSendPrompt} disabled={isGenerating} />

      <Toolbar
        toolMode={toolMode}
        onToolChange={setToolMode}
        disabled={!currentImage || isGenerating}
      />

      <ImageContainer>
        <ImageViewer
          imageUrl={currentImage || undefined}
          isLoading={isGenerating}
          onImageLoad={() => console.log("이미지 로드 완료")}
          imageRef={imageRef}
        />

        {currentImage && (
          <InteractionCanvas
            toolMode={toolMode}
            disabled={isGenerating}
            onInteraction={handleInteraction}
            onClearSelection={() => {
              // 외부 클릭 시에는 툴팁만 닫고 모드는 유지
              setTooltipVisible(false);
              setLastInteraction(null);
            }}
            imageRef={imageRef}
          />
        )}

        {currentImage && (
          <GlobalFeedbackContainer>
            {!showGlobalFeedback ? (
              <GlobalFeedbackToggleButton
                onClick={() => setShowGlobalFeedback(true)}
              >
                전체 피드백 작성하기
              </GlobalFeedbackToggleButton>
            ) : (
              <GlobalFeedbackInputContainer>
                <GlobalFeedbackInput
                  value={globalFeedback}
                  onChange={(e) => setGlobalFeedback(e.target.value)}
                  onKeyDown={handleGlobalFeedbackKeyDown}
                  placeholder="이미지 전체에 대한 피드백을 입력하세요"
                  autoFocus
                />
                <GlobalSendButton
                  onClick={handleGlobalFeedbackSubmit}
                  disabled={!globalFeedback.trim() || isGenerating}
                >
                  전송
                </GlobalSendButton>
                <GlobalFeedbackCancelButton
                  onClick={() => {
                    setShowGlobalFeedback(false);
                    setGlobalFeedback("");
                  }}
                >
                  취소
                </GlobalFeedbackCancelButton>
              </GlobalFeedbackInputContainer>
            )}
          </GlobalFeedbackContainer>
        )}

        {/* 이미지 생성 진행률 표시 */}
        {currentSession && (
          <ProgressContainer>
            <ProgressBar>
              <ProgressFill progress={progress} />
            </ProgressBar>
            <ProgressText>
              {isPaused ? (
                <>
                  일시정지됨 - {currentSession.steps.length}/
                  {currentSession.totalSteps} 스텝
                </>
              ) : isGenerating ? (
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

      <FeedbackTooltip
        x={tooltipPosition.x}
        y={tooltipPosition.y}
        visible={tooltipVisible}
        type={lastInteraction?.type}
        onClose={handleTooltipClose}
        onSubmit={handleFeedbackSubmit}
      />

      <ImageHistoryNavigator />

      {/* 최신 이미지로 가기 버튼 */}
      <LatestImageButton
        visible={
          !!(
            selectedStepIndex !== null &&
            currentSession?.steps &&
            currentSession.steps.length > 0
          )
        }
        onClick={() => selectStep(null)}
      >
        최신 이미지
      </LatestImageButton>

      {/* 생성 제어 버튼들 */}
      {currentSession && !currentSession.isComplete && (
        <ControlButtons>
          {!isPaused ? (
            <ControlButton
              variant="pause"
              onClick={pauseGeneration}
              disabled={!isGenerating}
            >
              일시정지
            </ControlButton>
          ) : (
            <ControlButton variant="resume" onClick={resumeGeneration}>
              계속하기
            </ControlButton>
          )}
        </ControlButtons>
      )}
    </AppContainer>
  );
}

export default App;
