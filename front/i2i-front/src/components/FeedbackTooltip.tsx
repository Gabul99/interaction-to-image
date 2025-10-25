import React, { useState, useEffect, useRef } from "react";
import styled from "styled-components";

const TooltipContainer = styled.div<{
  x: number;
  y: number;
  type?: "point" | "bbox";
}>`
  position: absolute;
  left: ${(props) => props.x}px;
  top: ${(props) => props.y}px;
  z-index: 1000;
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  min-width: 250px;
  transform: ${
    (props) =>
      props.type === "point"
        ? "translate(-50%, -100%)" // 포인트 위에 표시
        : "translate(0, -50%)" // 바운딩 박스 오른쪽에 표시
  };
`;

const TooltipTitle = styled.h3`
  margin: 0 0 8px 0;
  font-size: 14px;
  font-weight: 600;
  color: #f9fafb;
`;

const FeedbackTypeGrid = styled.div`
  display: flex;
  gap: 6px;
  margin-bottom: 8px;
`;

const FeedbackTypeButton = styled.button<{ active?: boolean }>`
  padding: 6px 10px;
  border: 2px solid ${(props) => (props.active ? "#6366f1" : "#374151")};
  border-radius: 6px;
  background: ${(props) => (props.active ? "#6366f1" : "transparent")};
  color: ${(props) => (props.active ? "white" : "#9ca3af")};
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    border-color: #6366f1;
    background: ${(props) =>
      props.active ? "#6366f1" : "rgba(99, 102, 241, 0.1)"};
  }
`;

const InputContainer = styled.div`
  display: flex;
  gap: 8px;
  align-items: flex-end;
`;

const FeedbackInput = styled.textarea`
  flex: 1;
  min-height: 40px;
  max-height: 80px;
  padding: 8px 12px;
  border: 2px solid #374151;
  border-radius: 6px;
  font-size: 12px;
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

const SendButton = styled.button`
  padding: 8px 12px;
  background: #6366f1;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
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

const CloseButton = styled.button`
  position: absolute;
  top: 8px;
  right: 8px;
  width: 20px;
  height: 20px;
  border: none;
  background: none;
  cursor: pointer;
  color: #9ca3af;
  font-size: 14px;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    color: #f9fafb;
  }
`;

export interface FeedbackData {
  type: FeedbackType;
  content: string;
}

interface FeedbackTooltipProps {
  x: number;
  y: number;
  visible: boolean;
  type?: "point" | "bbox";
  onClose: () => void;
  onSubmit: (data: FeedbackData) => void;
}

const FeedbackTooltip: React.FC<FeedbackTooltipProps> = ({
  x,
  y,
  visible,
  type,
  onClose,
  onSubmit,
}) => {
  const [feedbackType, setFeedbackType] = useState<FeedbackType>("text");
  const [feedbackContent, setFeedbackContent] = useState("");

  const handleSubmit = () => {
    if (feedbackContent.trim()) {
      onSubmit({
        type: feedbackType,
        content: feedbackContent.trim(),
      });
      setFeedbackContent("");
      onClose();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      onClose();
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      handleSubmit();
    }
  };

  if (!visible) return null;

  return (
    <TooltipContainer x={x} y={y} type={type} onKeyDown={handleKeyDown}>
      <CloseButton onClick={onClose}>×</CloseButton>
      <TooltipTitle>피드백 제공</TooltipTitle>

      <FeedbackTypeGrid>
        <FeedbackTypeButton
          active={feedbackType === "text"}
          onClick={() => setFeedbackType("text")}
        >
          텍스트
        </FeedbackTypeButton>
        <FeedbackTypeButton
          active={feedbackType === "image"}
          onClick={() => setFeedbackType("image")}
        >
          이미지
        </FeedbackTypeButton>
      </FeedbackTypeGrid>

      <InputContainer>
        <FeedbackInput
          value={feedbackContent}
          onChange={(e) => setFeedbackContent(e.target.value)}
          placeholder={
            feedbackType === "text"
              ? "이미지에 대한 피드백을 입력하세요..."
              : "이미지 URL 또는 설명을 입력하세요..."
          }
          autoFocus
        />
        <SendButton onClick={handleSubmit} disabled={!feedbackContent.trim()}>
          전송
        </SendButton>
      </InputContainer>
    </TooltipContainer>
  );
};

export default FeedbackTooltip;
