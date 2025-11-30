import React, { useState } from "react";
import styled from "styled-components";
import { useImageStore } from "../stores/imageStore";
import ObjectChipList from "./ObjectChipList";
import CompositionCanvas from "./CompositionCanvas";
import ImageViewer from "./ImageViewer";
import { requestObjectList, startImageGeneration } from "../api/composition";
import { API_BASE_URL } from "../config/api";

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
  } = useImageStore();

  const [currentPrompt, setCurrentPrompt] = useState<string>("");
  const [isLoadingObjects, setIsLoadingObjects] = useState(false);
  const imageRef = React.useRef<HTMLImageElement>(null);
  const placeholderRef = React.useRef<HTMLDivElement>(null);

  const handleSendPrompt = async (newPrompt: string) => {
    console.log("=".repeat(80));
    console.log("[CompositionModal] ========== 객체 리스트 생성 요청 ==========");
    console.log("[CompositionModal] 프롬프트:", newPrompt);
    console.log("=".repeat(80));
    
    setCurrentPrompt(newPrompt);
    setIsLoadingObjects(true);
    clearComposition();

    try {
      const objects = await requestObjectList(newPrompt);
      console.log("[CompositionModal] 객체 리스트 수신:", objects);
      setObjectList(objects);
      setIsLoadingObjects(false);
    } catch (error) {
      console.error("[CompositionModal] 객체 리스트 요청 실패:", error);
      setIsLoadingObjects(false);
    }
  };

  const handleComplete = async () => {
    console.log("=".repeat(80));
    console.log("[CompositionModal] ========== handleComplete 호출됨 ==========");
    console.log("[CompositionModal] currentPrompt:", currentPrompt);
    
    if (!currentPrompt) {
      console.error("[CompositionModal] 프롬프트가 없습니다!");
      return;
    }

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

      // 백엔드로 전송할 데이터 로깅
      console.log("=".repeat(80));
      console.log("[CompositionModal] ========== 백엔드로 전송할 데이터 ==========");
      console.log("[CompositionModal] 프롬프트:", currentPrompt);
      console.log("[CompositionModal] 객체 리스트:", compositionState.objects);
      console.log("[CompositionModal] 바운딩 박스:", bboxes);
      console.log("[CompositionModal] API_BASE_URL:", API_BASE_URL);
      console.log("=".repeat(80));

      // 백엔드 API 호출하여 세션 생성 및 WebSocket URL 받기
      console.log("[CompositionModal] startImageGeneration 호출 시작...");
      const result = await startImageGeneration(
        currentPrompt,
        compositionState.objects.length > 0
          ? compositionState.objects
          : undefined,
        bboxes
      );
      console.log("[CompositionModal] startImageGeneration 완료:", result);

      const { sessionId, rootNodeId, websocketUrl } = result;

      // 그래프 세션 생성 (백엔드에서 받은 sessionId와 rootNodeId 사용)
      const graphSessionId = createGraphSession(
        currentPrompt,
        sessionId,
        rootNodeId,
        compositionState.bboxes.length > 0 ? compositionState.bboxes : undefined
      );

      console.log("[CompositionModal] 그래프 세션 생성 완료:", {
        graphSessionId,
        sessionId,
        rootNodeId
      });

      // WebSocket 연결 (websocketUrl이 있으면 연결)
      console.log("[CompositionModal] WebSocket URL 확인:", websocketUrl);
      if (!websocketUrl) {
        console.error("=".repeat(80));
        console.error("[CompositionModal] WebSocket URL이 없습니다!");
        console.error("[CompositionModal] 백엔드 응답에서 websocketUrl을 받지 못했습니다.");
        console.error("[CompositionModal] 백엔드 서버가 실행 중인지 확인하세요.");
        console.error("[CompositionModal] SSH 터널링이 설정되어 있는지 확인하세요.");
        console.error("=".repeat(80));
        throw new Error("WebSocket URL이 없습니다. 백엔드 서버를 확인하세요.");
      }
      
      console.log("[CompositionModal] WebSocket 연결 시작");
      // imageStore의 startGenerationWithComposition을 사용하여 WebSocket 연결
      // 이 함수는 ImageSession과 GraphSession 모두에 데이터를 추가합니다
      const { startGenerationWithComposition } = useImageStore.getState();
      startGenerationWithComposition(
        currentPrompt,
        sessionId,
        websocketUrl,
        compositionState.bboxes.length > 0 ? compositionState.bboxes : undefined
      );

      console.log("[CompositionModal] 세션 생성 완료:", { sessionId, rootNodeId, websocketUrl, graphSessionId });
      console.log("[CompositionModal] 그래프 세션 생성:", graphSessionId);

      if (onComplete) {
        onComplete();
      }
      onClose();
    } catch (error) {
      console.error("=".repeat(80));
      console.error("[CompositionModal] ========== 이미지 생성 시작 실패 ==========");
      console.error("[CompositionModal] 에러:", error);
      if (error instanceof Error) {
        console.error("[CompositionModal] 에러 메시지:", error.message);
        console.error("[CompositionModal] 에러 스택:", error.stack);
      }
      console.error("=".repeat(80));
      // 사용자에게 알림 (선택적)
      alert(`이미지 생성 시작 실패: ${error instanceof Error ? error.message : String(error)}`);
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
