import React, { useRef, useEffect } from "react";
import styled from "styled-components";
import { useImageStore } from "../stores/imageStore";

const HistoryContainer = styled.div`
  position: fixed;
  bottom: 20px;
  left: 20px;
  right: 20px;
  z-index: 1000;
  background: rgba(26, 26, 46, 0.9);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  height: 120px;
`;

const ThumbnailList = styled.div`
  display: flex;
  height: 80px;
  align-items: stretch;
  gap: 8px;
  overflow-x: auto;
  padding-right: 8px;
`;

const ThumbnailContainer = styled.div<{
  isActive: boolean;
  isGenerating: boolean;
}>`
  position: relative;
  width: 80px;
  height: 100%;
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  border: 2px solid ${(props) => (props.isActive ? "#6366f1" : "transparent")};
  transition: all 0.2s ease;
  opacity: ${(props) => (props.isActive ? 1 : 0.6)};
  filter: ${(props) => (props.isActive ? "none" : "brightness(0.7)")};
  flex-shrink: 0;

  &:hover {
    border-color: ${(props) => (props.isActive ? "#8b5cf6" : "#6366f1")};
    opacity: 1;
    filter: none;
    transform: scale(1.02);
  }
`;

const ThumbnailImage = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
`;

const StepIndicator = styled.div<{ step: number; totalSteps: number }>`
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: rgba(0, 0, 0, 0.3);

  &::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: ${(props) => (props.step / props.totalSteps) * 100}%;
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    transition: width 0.3s ease;
  }
`;

const StepText = styled.div`
  position: absolute;
  top: 4px;
  right: 4px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  font-size: 10px;
  padding: 2px 4px;
  border-radius: 4px;
  font-weight: 600;
`;

const EmptyState = styled.div`
  text-align: center;
  color: #9ca3af;
  font-size: 14px;
  padding: 20px;
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
`;

interface ImageHistoryNavigatorProps {
  className?: string;
}

const ImageHistoryNavigator: React.FC<ImageHistoryNavigatorProps> = ({
  className,
}) => {
  const { currentSession, selectedStepIndex, selectStep } = useImageStore();
  const thumbnailListRef = useRef<HTMLDivElement>(null);

  // 새로운 이미지가 추가될 때 자동으로 오른쪽으로 스크롤
  useEffect(() => {
    if (thumbnailListRef.current && currentSession) {
      thumbnailListRef.current.scrollLeft =
        thumbnailListRef.current.scrollWidth;
    }
  }, [currentSession]);

  const handleThumbnailClick = (stepIndex: number) => {
    console.log("썸네일 클릭:", {
      stepIndex,
      selectedStepIndex,
      currentSession: currentSession?.id,
      totalSteps: currentSession?.steps.length,
      targetImageUrl: currentSession?.steps[stepIndex]?.url,
    });
    selectStep(stepIndex);
  };

  if (!currentSession || currentSession.steps.length === 0) {
    return (
      <HistoryContainer className={className}>
        <EmptyState>아직 생성된 이미지가 없습니다</EmptyState>
      </HistoryContainer>
    );
  }

  return (
    <HistoryContainer className={className}>
      <ThumbnailList ref={thumbnailListRef}>
        {currentSession.steps.map((imageStep, index) => {
          const isLatest = index === currentSession.steps.length - 1;
          const maxIndex = currentSession.steps.length - 1;

          // 선택된 스텝이 유효한 범위 내에 있는지 확인
          const isValidSelection =
            selectedStepIndex !== null && selectedStepIndex <= maxIndex;
          const isSelected = isValidSelection && selectedStepIndex === index;
          const isActive = isValidSelection ? isSelected : isLatest;

          return (
            <ThumbnailContainer
              key={imageStep.id}
              isActive={isActive}
              isGenerating={!currentSession.isComplete && isLatest}
              onClick={() => handleThumbnailClick(index)}
            >
              <ThumbnailImage
                src={imageStep.url}
                alt={`Step ${imageStep.step}`}
              />

              {!currentSession.isComplete && isLatest && (
                <StepIndicator
                  step={currentSession.steps.length}
                  totalSteps={currentSession.totalSteps}
                />
              )}

              <StepText>
                {imageStep.step}/{currentSession.totalSteps}
              </StepText>
            </ThumbnailContainer>
          );
        })}
      </ThumbnailList>
    </HistoryContainer>
  );
};

export default ImageHistoryNavigator;
