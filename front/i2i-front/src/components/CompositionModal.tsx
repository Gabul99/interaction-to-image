import React, { useState } from "react";
import styled from "styled-components";
import { useImageStore } from "../stores/imageStore";
import ObjectChipList from "./ObjectChipList";
import CompositionCanvas from "./CompositionCanvas";
import ImageViewer from "./ImageViewer";
import { requestObjectList, startImageGeneration } from "../api/composition";

const ModalOverlay = styled.div<{ visible: boolean }>`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
  z-index: 2000;
  display: ${(props) => (props.visible ? "flex" : "none")};
  align-items: center;
  justify-content: center;
  padding: 20px;
`;

const ModalContainer = styled.div`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  width: 100%;
  max-width: 50vw;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const ModalHeader = styled.div`
  padding: 24px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ModalTitle = styled.h2`
  color: #f9fafb;
  font-size: 20px;
  font-weight: 600;
  margin: 0;
`;

const CloseButton = styled.button`
  background: transparent;
  border: none;
  color: #9ca3af;
  font-size: 24px;
  cursor: pointer;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #f9fafb;
  }
`;

const ModalContent = styled.div`
  padding: 24px;
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
`;

const TwoColumnLayout = styled.div`
  display: flex;
  gap: 24px;
  flex: 1;
  min-height: 0;
`;

const LeftColumn = styled.div`
  flex: 0 0 300px;
  display: flex;
  flex-direction: column;
  gap: 24px;
`;

const RightColumn = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 24px;
  min-width: 0;
`;

const Section = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const SectionTitle = styled.h3`
  color: #d1d5db;
  font-size: 14px;
  font-weight: 500;
  margin: 0;
`;

const ImageContainer = styled.div`
  position: relative;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  width: 100%;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  overflow: hidden;

  /* 이미지에 맞게 높이 조정 */
  img {
    max-width: 100%;
    height: auto;
    display: block;
  }
`;

const ActionButtonGroup = styled.div`
  display: flex;
  gap: 12px;
  margin-top: auto;
  padding-top: 24px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

const ActionButton = styled.button<{ variant: "submit" | "cancel" }>`
  flex: 1;
  padding: 12px 20px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;

  ${(props) =>
    props.variant === "submit"
      ? `
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;

    &:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }

    &:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
  `
      : `
    background: rgba(55, 65, 81, 0.5);
    color: #f9fafb;
    border: 2px solid #374151;

    &:hover {
      background: rgba(55, 65, 81, 0.7);
      border-color: #6b7280;
    }
  `}
`;

const LoadingText = styled.div`
  color: #9ca3af;
  font-size: 14px;
  text-align: center;
  padding: 20px;
`;

const PromptInput = styled.input`
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #374151;
  border-radius: 8px;
  font-size: 14px;
  background: rgba(55, 65, 81, 0.5);
  color: #f9fafb;
  outline: none;
  transition: border-color 0.2s ease;

  &:focus {
    border-color: #6366f1;
  }

  &::placeholder {
    color: #9ca3af;
  }
`;

const SendPromptButton = styled.button`
  margin-top: 12px;
  width: 100%;
  padding: 12px;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

interface CompositionModalProps {
  visible: boolean;
  onClose: () => void;
  onComplete?: () => void;
}

const CompositionModal: React.FC<CompositionModalProps> = ({
  visible,
  onClose,
  onComplete,
}) => {
  const {
    compositionState,
    setObjectList,
    addObject,
    removeObject,
    selectObject,
    addBbox,
    updateBbox,
    removeBbox,
    clearComposition,
    createGraphSession,
    simulateGraphImageStream,
  } = useImageStore();

  const [currentPrompt, setCurrentPrompt] = useState<string>("");
  const [isLoadingObjects, setIsLoadingObjects] = useState(false);
  const imageRef = React.useRef<HTMLImageElement>(null);
  const placeholderRef = React.useRef<HTMLDivElement>(null);

  const handleSendPrompt = async (newPrompt: string) => {
    console.log("프롬프트 전송:", newPrompt);
    setCurrentPrompt(newPrompt);
    setIsLoadingObjects(true);
    clearComposition();

    try {
      const objects = await requestObjectList(newPrompt);
      setObjectList(objects);
      setIsLoadingObjects(false);
    } catch (error) {
      console.error("객체 리스트 요청 실패:", error);
      setIsLoadingObjects(false);
    }
  };

  const handleComplete = async () => {
    if (!currentPrompt) return;

    try {
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

      await startImageGeneration(
        currentPrompt,
        compositionState.objects.length > 0
          ? compositionState.objects
          : undefined,
        bboxes
      );

      // 그래프 세션 생성
      const graphSessionId = createGraphSession(
        currentPrompt,
        compositionState.bboxes.length > 0 ? compositionState.bboxes : undefined
      );

      // TODO: 백엔드 연동
      // 1. startImageGeneration API 호출하여 세션 생성 및 WebSocket URL 받기
      // 2. WebSocket 연결하여 이미지 스트림 수신
      // 3. 각 step 이미지를 받아서 addImageNodeToBranch로 노드 추가
      //
      // 현재는 시뮬레이션으로 처리
      // 백엔드 연결 시:
      //   const { sessionId, rootNodeId, websocketUrl } = await startImageGeneration(...);
      //   if (websocketUrl) {
      //     connectImageStream(sessionId, websocketUrl, onImageStep, onError, onComplete);
      //   }

      // 프롬프트 노드 ID 가져오기 (세션 생성 후 약간의 지연 후 확인)
      setTimeout(() => {
        const state = useImageStore.getState();
        const rootNodeId = state.currentGraphSession?.rootNodeId;

        if (rootNodeId) {
          // 더미 이미지 스트림 시뮬레이션 시작
          simulateGraphImageStream(graphSessionId, currentPrompt, rootNodeId);
        }
      }, 100);

      console.log("[CompositionModal] 그래프 세션 생성:", graphSessionId);

      if (onComplete) {
        onComplete();
      }
      onClose();
    } catch (error) {
      console.error("[CompositionModal] 이미지 생성 시작 실패:", error);
    }
  };

  const handleCancel = () => {
    clearComposition();
    setCurrentPrompt("");
    onClose();
  };

  if (!visible) return null;

  return (
    <ModalOverlay visible={visible} onClick={handleCancel}>
      <ModalContainer onClick={(e) => e.stopPropagation()}>
        <ModalHeader>
          <ModalTitle>새 이미지 생성</ModalTitle>
          <CloseButton onClick={handleCancel}>×</CloseButton>
        </ModalHeader>
        <ModalContent>
          <Section>
            <SectionTitle>프롬프트</SectionTitle>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (currentPrompt.trim() && !isLoadingObjects) {
                  handleSendPrompt(currentPrompt);
                }
              }}
            >
              <PromptInput
                type="text"
                placeholder="이미지 생성을 위한 프롬프트를 입력하세요..."
                value={currentPrompt}
                onChange={(e) => setCurrentPrompt(e.target.value)}
                disabled={isLoadingObjects}
              />
              <SendPromptButton
                type="submit"
                disabled={isLoadingObjects || !currentPrompt.trim()}
              >
                객체 리스트 생성
              </SendPromptButton>
            </form>
          </Section>

          {isLoadingObjects && (
            <LoadingText>객체 리스트를 생성하는 중...</LoadingText>
          )}

          {compositionState.objects.length > 0 && !isLoadingObjects && (
            <TwoColumnLayout>
              <LeftColumn>
                <Section>
                  <SectionTitle>객체 리스트</SectionTitle>
                  <ObjectChipList
                    objects={compositionState.objects}
                    selectedObjectId={compositionState.selectedObjectId}
                    onSelectObject={selectObject}
                    onAddObject={addObject}
                    onRemoveObject={removeObject}
                  />
                </Section>
              </LeftColumn>
              <RightColumn>
                <Section>
                  <SectionTitle>구도 설정</SectionTitle>
                  <ImageContainer>
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
                  </ImageContainer>
                </Section>
              </RightColumn>
            </TwoColumnLayout>
          )}

          <ActionButtonGroup>
            <ActionButton variant="cancel" onClick={handleCancel}>
              취소
            </ActionButton>
            <ActionButton
              variant="submit"
              onClick={handleComplete}
              disabled={!currentPrompt || isLoadingObjects}
            >
              {compositionState.bboxes.length > 0
                ? "구도 설정 완료 및 이미지 생성 시작"
                : "구도 없이 이미지 생성 시작"}
            </ActionButton>
          </ActionButtonGroup>
        </ModalContent>
      </ModalContainer>
    </ModalOverlay>
  );
};

export default CompositionModal;
