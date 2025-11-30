import React, { useState, useRef } from "react";
import styled from "styled-components";
import {
  type FeedbackArea,
  type FeedbackType,
  type FeedbackRecord,
  type ObjectChip,
  type InteractionData,
} from "../types";
import { useImageStore } from "../stores/imageStore";
import { createBranch as createBranchAPI } from "../api/branch";
import ImageViewer from "./ImageViewer";
import InteractionCanvas from "./InteractionCanvas";
import BboxOverlay from "./BboxOverlay";

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
  max-width: 1200px;
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
  flex-shrink: 0;
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
  min-height: 0;
`;

const TwoColumnLayout = styled.div`
  display: flex;
  gap: 24px;
  flex: 1;
  min-height: 0;
`;

const LeftColumn = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 24px;
  min-width: 0;
`;

const RightColumn = styled.div`
  flex: 0 0 400px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  overflow-y: auto;
`;

const SectionTitle = styled.h3`
  color: #d1d5db;
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 12px;
  margin-top: 0;

  &:first-child {
    margin-top: 0;
  }
`;

const AddFeedbackButton = styled.button`
  width: 100%;
  padding: 12px 20px;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  border: none;
  border-radius: 8px;
  color: white;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-bottom: 24px;

  &:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const FeedbackList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 24px;
`;

const FeedbackItem = styled.div`
  padding: 16px;
  background: rgba(55, 65, 81, 0.5);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  position: relative;
`;

const FeedbackItemHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
`;

const FeedbackItemInfo = styled.div`
  flex: 1;
`;

const FeedbackAreaBadge = styled.span<{ area: FeedbackArea }>`
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  margin-right: 8px;
  background: ${(props) => {
    if (props.area === "full") return "rgba(99, 102, 241, 0.2)";
    if (props.area === "bbox") return "rgba(139, 92, 246, 0.2)";
    return "rgba(236, 72, 153, 0.2)";
  }};
  color: ${(props) => {
    if (props.area === "full") return "#6366f1";
    if (props.area === "bbox") return "#8b5cf6";
    return "#ec4899";
  }};
  border: 1px solid
    ${(props) => {
      if (props.area === "full") return "rgba(99, 102, 241, 0.3)";
      if (props.area === "bbox") return "rgba(139, 92, 246, 0.3)";
      return "rgba(236, 72, 153, 0.3)";
    }};
`;

const DeleteButton = styled.button`
  padding: 4px 8px;
  background: rgba(239, 68, 68, 0.2);
  border: 1px solid rgba(239, 68, 68, 0.5);
  border-radius: 4px;
  color: #ef4444;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(239, 68, 68, 0.3);
    border-color: #ef4444;
  }
`;

const FeedbackContent = styled.div`
  color: #d1d5db;
  font-size: 14px;
  margin-bottom: 8px;
  white-space: pre-wrap;
  word-break: break-word;
`;

const AddFeedbackForm = styled.div`
  padding: 16px;
  background: rgba(99, 102, 241, 0.1);
  border-radius: 8px;
  border: 1px solid rgba(99, 102, 241, 0.2);
  margin-bottom: 24px;
`;

const OptionGroup = styled.div`
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  flex-wrap: wrap;
`;

const OptionButton = styled.button<{ selected: boolean }>`
  flex: 1;
  min-width: 100px;
  padding: 10px 14px;
  border: 2px solid ${(props) => (props.selected ? "#6366f1" : "#374151")};
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  background: ${(props) =>
    props.selected ? "rgba(99, 102, 241, 0.2)" : "rgba(55, 65, 81, 0.5)"};
  color: ${(props) => (props.selected ? "#6366f1" : "#f9fafb")};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover:not(:disabled) {
    border-color: #6366f1;
    background: rgba(99, 102, 241, 0.1);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const TextArea = styled.textarea`
  width: 100%;
  min-height: 100px;
  padding: 12px;
  background: rgba(55, 65, 81, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  color: #f9fafb;
  font-size: 14px;
  font-family: inherit;
  resize: vertical;
  margin-bottom: 12px;

  &:focus {
    outline: none;
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  }

  &::placeholder {
    color: #6b7280;
  }
`;

const FileInput = styled.input`
  display: none;
`;

const FileUploadButton = styled.button`
  width: 100%;
  padding: 10px;
  background: rgba(55, 65, 81, 0.5);
  border: 1px dashed rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  color: #d1d5db;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-bottom: 12px;

  &:hover {
    background: rgba(55, 65, 81, 0.7);
    border-color: rgba(255, 255, 255, 0.3);
  }
`;

const ImagePreview = styled.div`
  position: relative;
  margin-bottom: 12px;
`;

const PreviewImage = styled.img`
  width: 100%;
  max-height: 200px;
  object-fit: contain;
  border-radius: 6px;
  background: rgba(0, 0, 0, 0.3);
`;

const RemoveImageButton = styled.button`
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 6px 12px;
  background: rgba(239, 68, 68, 0.9);
  border: none;
  border-radius: 4px;
  color: white;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(239, 68, 68, 1);
  }
`;

const FormButtonGroup = styled.div`
  display: flex;
  gap: 8px;
`;

const FormButton = styled.button<{ variant: "submit" | "cancel" }>`
  flex: 1;
  padding: 10px 16px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
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
    border: 1px solid #374151;

    &:hover {
      background: rgba(55, 65, 81, 0.7);
      border-color: #6b7280;
    }
  `}
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

const InstructionText = styled.div`
  color: #9ca3af;
  font-size: 12px;
  margin-bottom: 12px;
  padding: 10px;
  background: rgba(99, 102, 241, 0.1);
  border-radius: 6px;
  border: 1px solid rgba(99, 102, 241, 0.2);
`;

const ImageSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
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

interface BranchingModalProps {
  visible: boolean;
  nodeId: string | null;
  onClose: () => void;
  onBranchCreated?: (branchId: string) => void;
  objects?: ObjectChip[];
  compositionBboxes?: Array<{
    id: string;
    objectId: string;
    x: number;
    y: number;
    width: number;
    height: number;
    color: string;
  }>;
}

const BranchingModal: React.FC<BranchingModalProps> = ({
  visible,
  nodeId,
  onClose,
  onBranchCreated,
  compositionBboxes = [],
}) => {
  const {
    currentFeedbackList,
    addFeedbackToCurrentList,
    removeFeedbackFromCurrentList,
    clearCurrentFeedbackList,
    currentGraphSession,
    getNodeById,
  } = useImageStore();

  const [isAddingFeedback, setIsAddingFeedback] = useState(false);
  const [selectedArea, setSelectedArea] = useState<FeedbackArea>("full");
  const [selectedType, setSelectedType] = useState<FeedbackType>("text");
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [selectedBboxIdForFeedback, setSelectedBboxIdForFeedback] = useState<
    string | null
  >(null);
  const [drawnBboxes, setDrawnBboxes] = useState<
    Array<{
      id: string;
      x: number;
      y: number;
      width: number;
      height: number;
      color: string;
    }>
  >([]);
  const [selectedPoint, setSelectedPoint] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const placeholderRef = useRef<HTMLDivElement>(null);

  // 노드에서 이미지 URL 가져오기
  const nodeImageUrl = React.useMemo(() => {
    if (!nodeId || !currentGraphSession) return null;
    const node = getNodeById(currentGraphSession.id, nodeId);
    return node?.data?.imageUrl || null;
  }, [nodeId, currentGraphSession, getNodeById]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreviewUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFileUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveImage = () => {
    setImageFile(null);
    setImagePreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleInteraction = (data: InteractionData) => {
    if (data.type === "bbox" && data.width && data.height) {
      // BBOX 그리기
      const bboxId = `bbox_${Date.now()}_${Math.random()
        .toString(36)
        .substr(2, 9)}`;
      const newBbox = {
        id: bboxId,
        x: data.x,
        y: data.y,
        width: data.width,
        height: data.height,
        color: "#6366f1",
      };
      setDrawnBboxes([...drawnBboxes, newBbox]);
      setSelectedBboxIdForFeedback(bboxId);
      setSelectedArea("bbox");
    } else if (data.type === "point") {
      // 포인팅
      setSelectedPoint({ x: data.x, y: data.y });
      setSelectedArea("point");
    }
  };

  const handleBboxClick = (bboxId: string) => {
    setSelectedBboxIdForFeedback(bboxId);
    setSelectedArea("bbox");
  };

  const handleSubmitFeedback = () => {
    const hasText = text.trim().length > 0;
    const hasImage = !!imageFile;

    if (!hasText && !hasImage) {
      return;
    }

    // 선택된 BBOX 찾기 (그린 BBOX 또는 compositionBboxes)
    const selectedBbox = selectedBboxIdForFeedback
      ? drawnBboxes.find((b) => b.id === selectedBboxIdForFeedback) ||
        compositionBboxes.find((b) => b.id === selectedBboxIdForFeedback)
      : null;

    const feedback: FeedbackRecord = {
      id: `feedback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      area: selectedArea,
      type: hasImage ? "image" : "text",
      text: hasText ? text : undefined,
      imageUrl: hasImage ? imagePreviewUrl || undefined : undefined,
      bbox:
        selectedArea === "bbox" && selectedBbox
          ? {
              x: selectedBbox.x,
              y: selectedBbox.y,
              width: selectedBbox.width,
              height: selectedBbox.height,
            }
          : undefined,
      point:
        selectedArea === "point" && selectedPoint ? selectedPoint : undefined,
      bboxId:
        selectedArea === "bbox"
          ? selectedBboxIdForFeedback || undefined
          : undefined,
      timestamp: Date.now(),
    };

    addFeedbackToCurrentList(feedback);

    // 폼 초기화
    setText("");
    setImageFile(null);
    setImagePreviewUrl(null);
    setSelectedType("text");
    setSelectedArea("full");
    setSelectedBboxIdForFeedback(null);
    setSelectedPoint(null);
    setIsAddingFeedback(false);
  };

  const handleCancelAdd = () => {
    setText("");
    setImageFile(null);
    setImagePreviewUrl(null);
    setSelectedType("text");
    setSelectedArea("full");
    setSelectedBboxIdForFeedback(null);
    setSelectedPoint(null);
    setIsAddingFeedback(false);
  };

  const handleDeleteFeedback = (feedbackId: string) => {
    removeFeedbackFromCurrentList(feedbackId);
  };

  const handleCreateBranch = async () => {
    if (!nodeId || !currentGraphSession || currentFeedbackList.length === 0) {
      return;
    }

    try {
      // 백엔드 API 호출
      const { branchId, websocketUrl } = await createBranchAPI(
        currentGraphSession.id,
        nodeId,
        currentFeedbackList
      );
      
      clearCurrentFeedbackList();
      onClose();

      if (onBranchCreated) {
        onBranchCreated(branchId, websocketUrl);
      }
    } catch (error) {
      console.error("[BranchingModal] 브랜치 생성 실패:", error);
      alert("브랜치 생성에 실패했습니다. 다시 시도해주세요.");
    }
  };

  const handleCancel = () => {
    clearCurrentFeedbackList();
    setIsAddingFeedback(false);
    setText("");
    setImageFile(null);
    setImagePreviewUrl(null);
    setSelectedType("text");
    setSelectedArea("full");
    setSelectedBboxIdForFeedback(null);
    onClose();
  };

  const canSubmit =
    (text.trim().length > 0 || !!imageFile) &&
    (selectedArea === "full" ||
      (selectedArea === "bbox" && selectedBboxIdForFeedback !== null) ||
      (selectedArea === "point" && selectedPoint !== null));

  const getAreaLabel = (area: FeedbackArea) => {
    if (area === "full") return "전체 이미지";
    if (area === "bbox") return "BBOX";
    return "포인팅";
  };

  if (!visible) return null;

  return (
    <ModalOverlay visible={visible} onClick={handleCancel}>
      <ModalContainer onClick={(e) => e.stopPropagation()}>
        <ModalHeader>
          <ModalTitle>브랜치 생성</ModalTitle>
          <CloseButton onClick={handleCancel}>×</CloseButton>
        </ModalHeader>
        <ModalContent>
          {nodeImageUrl ? (
            <TwoColumnLayout>
              <LeftColumn>
                <ImageSection>
                  <SectionTitle>이미지</SectionTitle>
                  <ImageContainer>
                    <ImageViewer
                      imageUrl={nodeImageUrl}
                      onImageLoad={() => console.log("이미지 로드 완료")}
                      imageRef={imageRef}
                      placeholderRef={placeholderRef}
                    />
                    {/* 인터랙션 캔버스 */}
                    {isAddingFeedback && (
                      <InteractionCanvas
                        toolMode={
                          selectedArea === "bbox"
                            ? "bbox"
                            : selectedArea === "point"
                            ? "point"
                            : "none"
                        }
                        disabled={selectedArea === "full"}
                        onInteraction={handleInteraction}
                        onClearSelection={() => {
                          setSelectedBboxIdForFeedback(null);
                          setSelectedPoint(null);
                        }}
                        imageRef={imageRef}
                      />
                    )}
                    {/* 그린 BBOX 오버레이 */}
                    {drawnBboxes.length > 0 && (
                      <BboxOverlay
                        bboxes={drawnBboxes.map((bbox) => ({
                          id: bbox.id,
                          objectId: "",
                          x: bbox.x,
                          y: bbox.y,
                          width: bbox.width,
                          height: bbox.height,
                          color: bbox.color,
                        }))}
                        objects={[]}
                        imageRef={imageRef}
                        selectedBboxId={selectedBboxIdForFeedback}
                        onBboxClick={handleBboxClick}
                        onClearSelection={() =>
                          setSelectedBboxIdForFeedback(null)
                        }
                      />
                    )}
                  </ImageContainer>
                </ImageSection>
              </LeftColumn>
              <RightColumn>
                <AddFeedbackButton onClick={() => setIsAddingFeedback(true)}>
                  + 피드백 추가하기
                </AddFeedbackButton>

                {isAddingFeedback && (
                  <AddFeedbackForm>
                    <SectionTitle>영역 선택</SectionTitle>
                    <OptionGroup>
                      <OptionButton
                        selected={selectedArea === "full"}
                        onClick={() => {
                          setSelectedArea("full");
                          setSelectedBboxIdForFeedback(null);
                          setSelectedPoint(null);
                        }}
                      >
                        전체 이미지
                      </OptionButton>
                      <OptionButton
                        selected={selectedArea === "bbox"}
                        onClick={() => setSelectedArea("bbox")}
                      >
                        BBOX
                      </OptionButton>
                      <OptionButton
                        selected={selectedArea === "point"}
                        onClick={() => {
                          setSelectedArea("point");
                          setSelectedBboxIdForFeedback(null);
                        }}
                      >
                        포인팅
                      </OptionButton>
                    </OptionGroup>

                    {selectedArea === "bbox" && (
                      <InstructionText>
                        {selectedBboxIdForFeedback
                          ? `BBOX 선택됨`
                          : "이미지 위에서 BBOX를 그려주세요."}
                      </InstructionText>
                    )}
                    {selectedArea === "point" && (
                      <InstructionText>
                        {selectedPoint
                          ? `포인팅 위치: (${Math.round(
                              selectedPoint.x * 100
                            )}%, ${Math.round(selectedPoint.y * 100)}%)`
                          : "이미지 위에서 클릭하여 포인팅하세요."}
                      </InstructionText>
                    )}

                    <SectionTitle>피드백 방식</SectionTitle>
                    <OptionGroup>
                      <OptionButton
                        selected={selectedType === "text"}
                        onClick={() => setSelectedType("text")}
                      >
                        텍스트
                      </OptionButton>
                      <OptionButton
                        selected={selectedType === "image"}
                        onClick={() => setSelectedType("image")}
                      >
                        이미지
                      </OptionButton>
                    </OptionGroup>

                    {selectedType === "text" && (
                      <TextArea
                        placeholder="피드백을 입력하세요..."
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                      />
                    )}

                    {selectedType === "image" && (
                      <>
                        <FileInput
                          ref={fileInputRef}
                          type="file"
                          accept="image/*"
                          onChange={handleFileSelect}
                        />
                        {!imageFile ? (
                          <FileUploadButton onClick={handleFileUploadClick}>
                            이미지 선택
                          </FileUploadButton>
                        ) : (
                          <ImagePreview>
                            <PreviewImage
                              src={imagePreviewUrl || ""}
                              alt="Preview"
                            />
                            <RemoveImageButton onClick={handleRemoveImage}>
                              제거
                            </RemoveImageButton>
                          </ImagePreview>
                        )}
                      </>
                    )}

                    <FormButtonGroup>
                      <FormButton variant="cancel" onClick={handleCancelAdd}>
                        취소
                      </FormButton>
                      <FormButton
                        variant="submit"
                        onClick={handleSubmitFeedback}
                        disabled={!canSubmit}
                      >
                        저장
                      </FormButton>
                    </FormButtonGroup>
                  </AddFeedbackForm>
                )}

                {currentFeedbackList.length > 0 && (
                  <>
                    <SectionTitle>
                      피드백 목록 ({currentFeedbackList.length})
                    </SectionTitle>
                    <FeedbackList>
                      {currentFeedbackList.map((feedback) => (
                        <FeedbackItem key={feedback.id}>
                          <FeedbackItemHeader>
                            <FeedbackItemInfo>
                              <FeedbackAreaBadge area={feedback.area}>
                                {getAreaLabel(feedback.area)}
                              </FeedbackAreaBadge>
                            </FeedbackItemInfo>
                            <DeleteButton
                              onClick={() => handleDeleteFeedback(feedback.id)}
                            >
                              삭제
                            </DeleteButton>
                          </FeedbackItemHeader>
                          {feedback.text && (
                            <FeedbackContent>{feedback.text}</FeedbackContent>
                          )}
                          {feedback.imageUrl && (
                            <div style={{ fontSize: "12px", color: "#9ca3af" }}>
                              [참조 이미지 피드백]
                            </div>
                          )}
                        </FeedbackItem>
                      ))}
                    </FeedbackList>
                  </>
                )}

                {currentFeedbackList.length === 0 && !isAddingFeedback && (
                  <InstructionText>
                    피드백을 추가하여 브랜치를 생성하세요.
                  </InstructionText>
                )}

                <ActionButtonGroup>
                  <ActionButton variant="cancel" onClick={handleCancel}>
                    취소
                  </ActionButton>
                  <ActionButton
                    variant="submit"
                    onClick={handleCreateBranch}
                    disabled={currentFeedbackList.length === 0}
                  >
                    브랜치 생성
                  </ActionButton>
                </ActionButtonGroup>
              </RightColumn>
            </TwoColumnLayout>
          ) : (
            <>
              <AddFeedbackButton onClick={() => setIsAddingFeedback(true)}>
                + 피드백 추가하기
              </AddFeedbackButton>
              {currentFeedbackList.length === 0 && !isAddingFeedback && (
                <InstructionText>
                  피드백을 추가하여 브랜치를 생성하세요.
                </InstructionText>
              )}
              <ActionButtonGroup>
                <ActionButton variant="cancel" onClick={handleCancel}>
                  취소
                </ActionButton>
                <ActionButton
                  variant="submit"
                  onClick={handleCreateBranch}
                  disabled={currentFeedbackList.length === 0}
                >
                  브랜치 생성
                </ActionButton>
              </ActionButtonGroup>
            </>
          )}
        </ModalContent>
      </ModalContainer>
    </ModalOverlay>
  );
};

export default BranchingModal;
