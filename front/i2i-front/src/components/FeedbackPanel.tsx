import React, { useState, useRef } from "react";
import styled from "styled-components";
import {
  type FeedbackArea,
  type FeedbackType,
  type FeedbackData,
} from "../types";

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

const OptionGroup = styled.div`
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  flex-wrap: wrap;
`;

const OptionButton = styled.button<{ selected: boolean }>`
  flex: 1;
  min-width: 100px;
  padding: 12px 16px;
  border: 2px solid ${(props) => (props.selected ? "#6366f1" : "#374151")};
  border-radius: 8px;
  font-size: 14px;
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

const InstructionText = styled.div`
  color: #9ca3af;
  font-size: 13px;
  margin-bottom: 12px;
  padding: 12px;
  background: rgba(99, 102, 241, 0.1);
  border-radius: 8px;
  border: 1px solid rgba(99, 102, 241, 0.2);
`;

const TextInput = styled.textarea`
  width: 100%;
  min-height: 120px;
  max-height: 200px;
  padding: 12px 16px;
  border: 2px solid #374151;
  border-radius: 8px;
  font-size: 14px;
  background: rgba(55, 65, 81, 0.5);
  color: #f9fafb;
  resize: vertical;
  outline: none;
  transition: border-color 0.2s ease;
  font-family: inherit;

  &:focus {
    border-color: #6366f1;
  }

  &::placeholder {
    color: #9ca3af;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    background: rgba(55, 65, 81, 0.3);
  }
`;

const FileUploadArea = styled.div<{ hasFile: boolean }>`
  width: 100%;
  padding: 40px 20px;
  border: 2px dashed ${(props) => (props.hasFile ? "#6366f1" : "#374151")};
  border-radius: 8px;
  background: rgba(55, 65, 81, 0.3);
  text-align: center;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    border-color: #6366f1;
    background: rgba(55, 65, 81, 0.5);
  }
`;

const FileInput = styled.input`
  display: none;
`;

const FileUploadText = styled.div`
  color: #9ca3af;
  font-size: 14px;
  margin-bottom: 8px;
`;

const FileUploadHint = styled.div`
  color: #6b7280;
  font-size: 12px;
`;

const FileName = styled.div`
  color: #6366f1;
  font-size: 14px;
  font-weight: 500;
  margin-top: 8px;
`;

const ImagePreview = styled.div`
  width: 100%;
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const PreviewImage = styled.img`
  width: 100%;
  max-height: 300px;
  object-fit: contain;
  border-radius: 8px;
  border: 2px solid #374151;
  background: rgba(55, 65, 81, 0.3);
`;

const ImagePreviewHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const RemoveImageButton = styled.button`
  padding: 6px 12px;
  background: rgba(239, 68, 68, 0.2);
  border: 1px solid rgba(239, 68, 68, 0.5);
  border-radius: 6px;
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

const ButtonGroup = styled.div`
  display: flex;
  gap: 12px;
  margin-top: auto;
  padding-top: 24px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

const Button = styled.button<{ variant: "submit" | "skip" }>`
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

interface FeedbackPanelProps {
  visible: boolean;
  onSubmit: (feedback: FeedbackData) => void;
  onSkip: () => void;
  // 서버로부터 받은 피드백 요청 정보
  area?: FeedbackArea;
  // 선택된 영역 정보 (InteractionCanvas에서 업데이트)
  selectedPoint?: { x: number; y: number };
  selectedBbox?: { x: number; y: number; width: number; height: number };
  // 영역 선택 콜백
  onAreaSelect: (area: string) => void;
}

const FeedbackPanel: React.FC<FeedbackPanelProps> = ({
  visible,
  onSubmit,
  onSkip,
  area = "full",
  selectedPoint,
  selectedBbox,
  onAreaSelect,
}) => {
  const [selectedArea, setSelectedArea] = useState<FeedbackArea>(area);
  const [selectedType, setSelectedType] = useState<FeedbackType>("text");
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 사용자가 패널에서 직접 영역을 선택했는지 추적
  const isUserSelectingRef = React.useRef(false);

  // area가 변경되면 선택된 영역도 업데이트 및 콜백 호출
  React.useEffect(() => {
    console.log("FeedbackPanel - area prop 변경:", area);
    setSelectedArea(area);
    // area prop이 변경되면 초기 설정이므로 onAreaSelect 호출
    isUserSelectingRef.current = true;
  }, [area]);

  // 영역 선택 시 콜백 호출 (사용자가 직접 선택한 경우에만)
  React.useEffect(() => {
    console.log(
      "FeedbackPanel - 영역 선택 변경:",
      selectedArea,
      "사용자 선택:",
      isUserSelectingRef.current
    );
    if (isUserSelectingRef.current) {
      onAreaSelect(selectedArea);
      isUserSelectingRef.current = false; // 콜백 호출 후 리셋
    }
  }, [selectedArea, onAreaSelect]);

  // selectedBbox 변경 시 디버깅 및 자동 영역 선택
  React.useEffect(() => {
    console.log(
      "FeedbackPanel - selectedBbox 변경:",
      selectedBbox,
      "selectedArea:",
      selectedArea
    );
    // BBOX가 선택되었는데 selectedArea가 bbox가 아니면 자동으로 변경
    if (selectedBbox && selectedArea !== "bbox") {
      console.log(
        "BBOX가 선택되었지만 selectedArea가 bbox가 아님. 자동으로 변경 (콜백 없이)"
      );
      // 자동 변경이므로 onAreaSelect 호출하지 않음
      setSelectedArea("bbox");
    }
  }, [selectedBbox, selectedArea]);

  // selectedPoint 변경 시 디버깅 및 자동 영역 선택
  React.useEffect(() => {
    console.log(
      "FeedbackPanel - selectedPoint 변경:",
      selectedPoint,
      "selectedArea:",
      selectedArea
    );
    // 포인팅이 선택되었는데 selectedArea가 point가 아니면 자동으로 변경
    if (selectedPoint && selectedArea !== "point") {
      console.log(
        "포인팅이 선택되었지만 selectedArea가 point가 아님. 자동으로 변경 (콜백 없이)"
      );
      // 자동 변경이므로 onAreaSelect 호출하지 않음
      setSelectedArea("point");
    }
  }, [selectedPoint, selectedArea]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      // 이미지 미리보기 URL 생성
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
    // 파일 입력 초기화
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = () => {
    // 텍스트 또는 이미지 중 하나 이상 필요
    const hasText = text.trim().length > 0;
    const hasImage = !!imageFile;

    if (!hasText && !hasImage) {
      return;
    }

    // 선택된 영역 정보 사용
    const finalPoint = selectedArea === "point" ? selectedPoint : undefined;
    const finalBbox = selectedArea === "bbox" ? selectedBbox : undefined;

    // 피드백 타입 결정: 이미지가 있으면 image, 텍스트만 있으면 text
    const feedbackType: FeedbackType = hasImage ? "image" : "text";

    const feedback: FeedbackData = {
      area: selectedArea,
      type: feedbackType,
      text: hasText ? text : undefined, // 텍스트가 있으면 포함
      image: hasImage ? imageFile || undefined : undefined,
      point: finalPoint,
      bbox: finalBbox,
    };

    onSubmit(feedback);

    // 폼 초기화
    setText("");
    setImageFile(null);
    setImagePreviewUrl(null);
    setSelectedType("text");
  };

  const handleSkip = () => {
    onSkip();
    // 폼 초기화
    setText("");
    setImageFile(null);
    setImagePreviewUrl(null);
    setSelectedType("text");
  };

  // 영역이 선택되었는지 확인
  const isAreaSelected = (() => {
    if (selectedArea === "full") {
      return true; // 전체 이미지는 항상 활성화
    } else if (selectedArea === "point") {
      const result = !!selectedPoint;
      console.log("포인팅 영역 선택 확인:", { selectedPoint, result });
      return result; // 포인팅은 선택되었을 때만 활성화
    } else if (selectedArea === "bbox") {
      const result = !!selectedBbox;
      console.log("BBOX 영역 선택 확인:", {
        selectedBbox,
        result,
        selectedArea,
      });
      return result; // BBOX는 선택되었을 때만 활성화
    }
    return false;
  })();

  // 텍스트 또는 이미지 중 하나 이상 있으면 제출 가능
  const canSubmit = isAreaSelected && (text.trim().length > 0 || !!imageFile);

  const getInstructionText = () => {
    if (selectedArea === "full") {
      return "전체 이미지에 대한 피드백을 제공하세요.";
    } else if (selectedArea === "point") {
      if (selectedPoint) {
        return "포인팅 위치가 선택되었습니다. 피드백을 입력하세요.";
      }
      return "이미지를 클릭하여 포인팅 위치를 선택하세요.";
    } else if (selectedArea === "bbox") {
      if (selectedBbox) {
        return "바운딩 박스가 선택되었습니다. 피드백을 입력하세요.";
      }
      return "이미지에서 드래그하여 바운딩 박스를 그리세요.";
    }
    return "";
  };

  return (
    <PanelContainer visible={visible}>
      <PanelContent>
        <PanelTitle>피드백 제공</PanelTitle>

        <SectionTitle>영역 선택</SectionTitle>
        <OptionGroup>
          <OptionButton
            selected={selectedArea === "full"}
            onClick={() => {
              isUserSelectingRef.current = true;
              setSelectedArea("full");
            }}
          >
            전체 이미지
          </OptionButton>
          <OptionButton
            selected={selectedArea === "point"}
            onClick={() => {
              isUserSelectingRef.current = true;
              setSelectedArea("point");
            }}
          >
            포인팅
          </OptionButton>
          <OptionButton
            selected={selectedArea === "bbox"}
            onClick={() => {
              isUserSelectingRef.current = true;
              setSelectedArea("bbox");
            }}
          >
            BBOX
          </OptionButton>
        </OptionGroup>

        <InstructionText>{getInstructionText()}</InstructionText>

        <SectionTitle>피드백 방식</SectionTitle>
        <OptionGroup>
          <OptionButton
            selected={selectedType === "text"}
            onClick={() => setSelectedType("text")}
            disabled={!isAreaSelected}
          >
            텍스트
          </OptionButton>
          <OptionButton
            selected={selectedType === "image"}
            onClick={() => setSelectedType("image")}
            disabled={!isAreaSelected}
          >
            참조 이미지
          </OptionButton>
        </OptionGroup>

        {/* 텍스트 입력 (항상 표시) */}
        <TextInput
          placeholder={
            isAreaSelected
              ? "피드백 내용을 입력하세요... (선택사항)"
              : "먼저 영역을 선택해주세요"
          }
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={!isAreaSelected}
          autoFocus={isAreaSelected && selectedType === "text"}
        />

        {/* 이미지 업로드 (참조 이미지 모드일 때만 표시) */}
        {selectedType === "image" && (
          <>
            <FileUploadArea
              hasFile={!!imageFile}
              onClick={isAreaSelected ? handleFileUploadClick : undefined}
              style={{
                cursor: isAreaSelected ? "pointer" : "not-allowed",
                opacity: isAreaSelected ? 1 : 0.5,
                marginTop: "16px",
              }}
            >
              <FileUploadText>
                {isAreaSelected
                  ? imageFile
                    ? "클릭하여 다른 파일 선택"
                    : "클릭하여 파일 선택"
                  : "먼저 영역을 선택해주세요"}
              </FileUploadText>
              <FileUploadHint>
                {isAreaSelected
                  ? "이미지 파일을 업로드하세요 (PNG, JPG, JPEG)"
                  : "영역 선택 후 파일을 업로드할 수 있습니다"}
              </FileUploadHint>
              {imageFile && <FileName>{imageFile.name}</FileName>}
              <FileInput
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                disabled={!isAreaSelected}
              />
            </FileUploadArea>

            {/* 이미지 미리보기 */}
            {imagePreviewUrl && (
              <ImagePreview>
                <ImagePreviewHeader>
                  <span
                    style={{
                      color: "#d1d5db",
                      fontSize: "14px",
                      fontWeight: 500,
                    }}
                  >
                    업로드된 이미지
                  </span>
                  <RemoveImageButton onClick={handleRemoveImage}>
                    제거
                  </RemoveImageButton>
                </ImagePreviewHeader>
                <PreviewImage
                  src={imagePreviewUrl}
                  alt="업로드된 참조 이미지"
                />
              </ImagePreview>
            )}
          </>
        )}

        <ButtonGroup>
          <Button variant="skip" onClick={handleSkip}>
            건너뛰기
          </Button>
          <Button variant="submit" onClick={handleSubmit} disabled={!canSubmit}>
            전송
          </Button>
        </ButtonGroup>
      </PanelContent>
    </PanelContainer>
  );
};

export default FeedbackPanel;
