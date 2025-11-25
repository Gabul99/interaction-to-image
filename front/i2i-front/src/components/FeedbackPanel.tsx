import React, { useState, useRef } from "react";
import styled from "styled-components";
import {
  type FeedbackArea,
  type FeedbackType,
  type FeedbackRecord,
  type ObjectChip,
} from "../types";
import { useImageStore } from "../stores/imageStore";

const PanelContainer = styled.div<{ visible: boolean }>`
  position: fixed;
  right: 0;
  top: 0;
  bottom: 0;
  width: 400px;
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  box-shadow: -4px 0 20px rgba(0, 0, 0, 0.3);
  border-left: 1px solid rgba(255, 255, 255, 0.1);
  z-index: 1500;
  display: flex;
  flex-direction: column;
  transform: ${(props) =>
    props.visible ? "translateX(0)" : "translateX(100%)"};
  transition: transform 0.3s ease;
  overflow-y: auto;
`;

const PanelContent = styled.div`
  padding: 24px;
  flex: 1;
  display: flex;
  flex-direction: column;
`;

const PanelTitle = styled.h2`
  color: #f9fafb;
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 24px;
`;

const SectionTitle = styled.h3`
  color: #d1d5db;
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 12px;
  margin-top: 20px;

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

const FeedbackItem = styled.div<{ isReadOnly?: boolean }>`
  padding: 16px;
  background: ${(props) =>
    props.isReadOnly ? "rgba(55, 65, 81, 0.3)" : "rgba(55, 65, 81, 0.5)"};
  border-radius: 8px;
  border: 1px solid
    ${(props) =>
      props.isReadOnly
        ? "rgba(255, 255, 255, 0.05)"
        : "rgba(255, 255, 255, 0.1)"};
  position: relative;
  opacity: ${(props) => (props.isReadOnly ? 0.8 : 1)};
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

const BboxChip = styled.div<{ color: string }>`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  background: ${(props) => `${props.color}20`};
  color: ${(props) => props.color};
  border: 1px solid ${(props) => `${props.color}40`};
`;

const ColorIndicator = styled.div<{ color: string }>`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${(props) => props.color};
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

const FeedbackImagePreview = styled.div`
  margin-top: 8px;
  padding: 8px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
  font-size: 12px;
  color: #9ca3af;
`;

const FeedbackMeta = styled.div`
  font-size: 11px;
  color: #6b7280;
  margin-top: 8px;
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

const ActionButton = styled.button<{ variant: "submit" | "skip" }>`
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

interface FeedbackPanelProps {
  visible: boolean;
  onClose: () => void;
  onSubmit?: () => void;
  onSkip?: () => void;
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
  onBboxSelect?: (bboxId: string | null) => void;
  selectedBboxId?: string | null;
}

const FeedbackPanel: React.FC<FeedbackPanelProps> = ({
  visible,
  onSubmit,
  onSkip,
  objects = [],
  compositionBboxes = [],
  onBboxSelect,
  selectedBboxId,
}) => {
  const {
    currentFeedbackList,
    addFeedbackToCurrentList,
    removeFeedbackFromCurrentList,
    getFeedbackHistoryForBbox,
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
  const fileInputRef = useRef<HTMLInputElement>(null);

  // BBOX 선택 시 자동으로 area를 bbox로 변경
  React.useEffect(() => {
    if (selectedBboxId) {
      setSelectedBboxIdForFeedback(selectedBboxId);
      setSelectedArea("bbox");
    }
  }, [selectedBboxId]);

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

  const handleSubmitFeedback = () => {
    const hasText = text.trim().length > 0;
    const hasImage = !!imageFile;

    if (!hasText && !hasImage) {
      return;
    }

    const selectedBbox = compositionBboxes.find(
      (bbox) => bbox.id === selectedBboxIdForFeedback
    );

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
    setIsAddingFeedback(false);
    if (onBboxSelect) {
      onBboxSelect(null);
    }
  };

  const handleCancelAdd = () => {
    setText("");
    setImageFile(null);
    setImagePreviewUrl(null);
    setSelectedType("text");
    setSelectedArea("full");
    setSelectedBboxIdForFeedback(null);
    setIsAddingFeedback(false);
    if (onBboxSelect) {
      onBboxSelect(null);
    }
  };

  const handleDeleteFeedback = (feedbackId: string) => {
    removeFeedbackFromCurrentList(feedbackId);
  };

  const canSubmit =
    (text.trim().length > 0 || !!imageFile) &&
    (selectedArea !== "bbox" || selectedBboxIdForFeedback !== null);

  const getAreaLabel = (area: FeedbackArea) => {
    if (area === "full") return "전체 이미지";
    if (area === "bbox") return "BBOX";
    return "포인팅";
  };

  const getBboxLabel = (bboxId: string) => {
    const bbox = compositionBboxes.find((b) => b.id === bboxId);
    if (!bbox) return "";
    const object = objects.find((obj) => obj.id === bbox.objectId);
    return object?.label || "";
  };

  return (
    <PanelContainer visible={visible}>
      <PanelContent>
        <PanelTitle>피드백 제공</PanelTitle>

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
                  if (onBboxSelect) {
                    onBboxSelect(null);
                  }
                }}
              >
                전체 이미지
              </OptionButton>
              <OptionButton
                selected={selectedArea === "bbox"}
                onClick={() => {
                  setSelectedArea("bbox");
                }}
                disabled={compositionBboxes.length === 0}
              >
                BBOX
              </OptionButton>
            </OptionGroup>

            {selectedArea === "bbox" && (
              <>
                <InstructionText>
                  {selectedBboxIdForFeedback
                    ? `선택됨: ${getBboxLabel(selectedBboxIdForFeedback)}`
                    : "BBOX를 클릭하여 선택하세요."}
                </InstructionText>
                {/* 선택된 BBOX의 피드백 히스토리 표시 */}
                {selectedBboxIdForFeedback && (
                  <>
                    {(() => {
                      const history = getFeedbackHistoryForBbox(
                        selectedBboxIdForFeedback
                      );
                      if (history.length === 0) return null;
                      return (
                        <div style={{ marginTop: "12px" }}>
                          <SectionTitle style={{ marginTop: "0" }}>
                            이전 피드백 ({history.length})
                          </SectionTitle>
                          <FeedbackList>
                            {history.map((feedback) => (
                              <FeedbackItem key={feedback.id} isReadOnly>
                                <FeedbackItemHeader>
                                  <FeedbackItemInfo>
                                    <FeedbackAreaBadge area={feedback.area}>
                                      {getAreaLabel(feedback.area)}
                                    </FeedbackAreaBadge>
                                  </FeedbackItemInfo>
                                </FeedbackItemHeader>
                                {feedback.text && (
                                  <FeedbackContent>
                                    {feedback.text}
                                  </FeedbackContent>
                                )}
                                {feedback.imageUrl && (
                                  <FeedbackImagePreview>
                                    [참조 이미지 피드백]
                                  </FeedbackImagePreview>
                                )}
                                <FeedbackMeta>
                                  {new Date(feedback.timestamp).toLocaleString(
                                    "ko-KR",
                                    {
                                      month: "short",
                                      day: "numeric",
                                      hour: "2-digit",
                                      minute: "2-digit",
                                    }
                                  )}
                                </FeedbackMeta>
                              </FeedbackItem>
                            ))}
                          </FeedbackList>
                        </div>
                      );
                    })()}
                  </>
                )}
              </>
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
                    <PreviewImage src={imagePreviewUrl || ""} alt="Preview" />
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
              {currentFeedbackList.map((feedback) => {
                const bbox = feedback.bboxId
                  ? compositionBboxes.find((b) => b.id === feedback.bboxId)
                  : null;
                const object = bbox
                  ? objects.find((obj) => obj.id === bbox.objectId)
                  : null;

                return (
                  <FeedbackItem key={feedback.id}>
                    <FeedbackItemHeader>
                      <FeedbackItemInfo>
                        <FeedbackAreaBadge area={feedback.area}>
                          {getAreaLabel(feedback.area)}
                        </FeedbackAreaBadge>
                        {feedback.area === "bbox" && object && (
                          <BboxChip color={object.color}>
                            <ColorIndicator color={object.color} />
                            <span>{object.label}</span>
                          </BboxChip>
                        )}
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
                      <FeedbackImagePreview>
                        [참조 이미지 피드백]
                      </FeedbackImagePreview>
                    )}
                    <FeedbackMeta>
                      {new Date(feedback.timestamp).toLocaleString("ko-KR", {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </FeedbackMeta>
                  </FeedbackItem>
                );
              })}
            </FeedbackList>
          </>
        )}

        {currentFeedbackList.length === 0 && !isAddingFeedback && (
          <InstructionText>
            피드백을 추가하여 이미지 생성에 대한 의견을 남겨주세요.
          </InstructionText>
        )}

        {/* 제출 및 건너뛰기 버튼 */}
        <ActionButtonGroup>
          {onSkip && (
            <ActionButton variant="skip" onClick={onSkip}>
              건너뛰기
            </ActionButton>
          )}
          {onSubmit && (
            <ActionButton
              variant="submit"
              onClick={onSubmit}
              disabled={currentFeedbackList.length === 0}
            >
              피드백 제출
            </ActionButton>
          )}
        </ActionButtonGroup>
      </PanelContent>
    </PanelContainer>
  );
};

export default FeedbackPanel;
