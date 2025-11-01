import React, { useState, useRef } from "react";
import styled from "styled-components";
import {
  type FeedbackArea,
  type FeedbackType,
  type FeedbackData,
} from "../types";

const ModalOverlay = styled.div<{ visible: boolean }>`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: ${(props) => (props.visible ? 1 : 0)};
  pointer-events: ${(props) => (props.visible ? "auto" : "none")};
  transition: opacity 0.2s ease;
`;

const ModalContainer = styled.div<{ visible: boolean }>`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.1);
  min-width: 500px;
  max-width: 600px;
  max-height: 80vh;
  overflow-y: auto;
  transform: ${(props) => (props.visible ? "scale(1)" : "scale(0.95)")};
  transition: transform 0.2s ease;
`;

const ModalTitle = styled.h2`
  color: #f9fafb;
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 20px;
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
`;

const OptionButton = styled.button<{ selected: boolean }>`
  flex: 1;
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

  &:hover {
    border-color: #6366f1;
    background: rgba(99, 102, 241, 0.1);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
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

const ButtonGroup = styled.div`
  display: flex;
  gap: 12px;
  margin-top: 24px;
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

interface FeedbackModalProps {
  visible: boolean;
  onClose: () => void;
  onSubmit: (feedback: FeedbackData) => void;
  onSkip: () => void;
  // 서버로부터 받은 피드백 요청 정보
  area?: FeedbackArea;
  point?: { x: number; y: number };
  bbox?: { x: number; y: number; width: number; height: number };
}

const FeedbackModal: React.FC<FeedbackModalProps> = ({
  visible,
  onClose,
  onSubmit,
  onSkip,
  area = "full",
  point,
  bbox,
}) => {
  const [selectedArea, setSelectedArea] = useState<FeedbackArea>(area);
  const [selectedType, setSelectedType] = useState<FeedbackType>("text");
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // area가 변경되면 선택된 영역도 업데이트
  React.useEffect(() => {
    setSelectedArea(area);
  }, [area]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
    }
  };

  const handleFileUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleSubmit = () => {
    if (selectedType === "text" && !text.trim()) {
      return; // 텍스트 피드백인데 내용이 없으면 제출 불가
    }
    if (selectedType === "image" && !imageFile) {
      return; // 이미지 피드백인데 파일이 없으면 제출 불가
    }

    const feedback: FeedbackData = {
      area: selectedArea,
      type: selectedType,
      text: selectedType === "text" ? text : undefined,
      image: selectedType === "image" ? imageFile || undefined : undefined,
      point: selectedArea === "point" ? point : undefined,
      bbox: selectedArea === "bbox" ? bbox : undefined,
    };

    onSubmit(feedback);

    // 폼 초기화
    setText("");
    setImageFile(null);
    setSelectedType("text");
  };

  const handleSkip = () => {
    onSkip();
    // 폼 초기화
    setText("");
    setImageFile(null);
    setSelectedType("text");
  };

  const canSubmit =
    (selectedType === "text" && text.trim()) ||
    (selectedType === "image" && imageFile);

  return (
    <ModalOverlay visible={visible} onClick={onClose}>
      <ModalContainer visible={visible} onClick={(e) => e.stopPropagation()}>
        <ModalTitle>피드백 제공</ModalTitle>

        <SectionTitle>영역 선택</SectionTitle>
        <OptionGroup>
          <OptionButton
            selected={selectedArea === "full"}
            onClick={() => setSelectedArea("full")}
          >
            전체 이미지
          </OptionButton>
          <OptionButton
            selected={selectedArea === "point"}
            onClick={() => setSelectedArea("point")}
            disabled={!point}
          >
            포인팅
          </OptionButton>
          <OptionButton
            selected={selectedArea === "bbox"}
            onClick={() => setSelectedArea("bbox")}
            disabled={!bbox}
          >
            BBOX
          </OptionButton>
        </OptionGroup>

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
            참조 이미지
          </OptionButton>
        </OptionGroup>

        {selectedType === "text" ? (
          <TextInput
            placeholder="피드백 내용을 입력하세요..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            autoFocus
          />
        ) : (
          <FileUploadArea hasFile={!!imageFile} onClick={handleFileUploadClick}>
            <FileUploadText>
              {imageFile ? "클릭하여 다른 파일 선택" : "클릭하여 파일 선택"}
            </FileUploadText>
            <FileUploadHint>
              이미지 파일을 업로드하세요 (PNG, JPG, JPEG)
            </FileUploadHint>
            {imageFile && <FileName>{imageFile.name}</FileName>}
            <FileInput
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
            />
          </FileUploadArea>
        )}

        <ButtonGroup>
          <Button variant="skip" onClick={handleSkip}>
            건너뛰기
          </Button>
          <Button variant="submit" onClick={handleSubmit} disabled={!canSubmit}>
            전송
          </Button>
        </ButtonGroup>
      </ModalContainer>
    </ModalOverlay>
  );
};

export default FeedbackModal;
