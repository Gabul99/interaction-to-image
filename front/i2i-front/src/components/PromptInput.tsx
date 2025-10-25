import React, { useState } from "react";
import styled from "styled-components";

const PromptContainer = styled.div`
  position: fixed;
  top: 20px;
  left: 20px;
  z-index: 1000;
  background: rgba(26, 26, 46, 0.9);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  min-width: 300px;
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

const IntervalSelector = styled.div`
  margin-top: 12px;
`;

const IntervalLabel = styled.label`
  display: block;
  font-size: 12px;
  color: #9ca3af;
  margin-bottom: 8px;
  font-weight: 500;
`;

const IntervalOptions = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
`;

const IntervalOption = styled.button<{ selected: boolean }>`
  padding: 8px 12px;
  border: 2px solid ${(props) => (props.selected ? "#6366f1" : "#374151")};
  border-radius: 6px;
  font-size: 12px;
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
`;

const SendButton = styled.button`
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

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
  }

  &:active {
    transform: translateY(0);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

interface PromptInputProps {
  onSendPrompt: (prompt: string, interval: number) => void;
  disabled?: boolean;
}

const INTERVAL_OPTIONS = [
  { value: 1, label: "1" },
  { value: 5, label: "5" },
  { value: 10, label: "10" },
  { value: 20, label: "20" },
];

const PromptInputComponent: React.FC<PromptInputProps> = ({
  onSendPrompt,
  disabled = false,
}) => {
  const [prompt, setPrompt] = useState("");
  const [selectedInterval, setSelectedInterval] = useState(1);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !disabled) {
      onSendPrompt(prompt.trim(), selectedInterval);
      setPrompt("");
    }
  };

  return (
    <PromptContainer>
      <form onSubmit={handleSubmit}>
        <PromptInput
          type="text"
          placeholder="이미지 생성을 위한 프롬프트를 입력하세요..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={disabled}
        />

        <IntervalSelector>
          <IntervalLabel>생성 간격 (스텝 단위)</IntervalLabel>
          <IntervalOptions>
            {INTERVAL_OPTIONS.map((option) => (
              <IntervalOption
                key={option.value}
                selected={selectedInterval === option.value}
                onClick={() => setSelectedInterval(option.value)}
                disabled={disabled}
              >
                {option.label}
              </IntervalOption>
            ))}
          </IntervalOptions>
        </IntervalSelector>

        <SendButton type="submit" disabled={disabled || !prompt.trim()}>
          이미지 생성 시작
        </SendButton>
      </form>
    </PromptContainer>
  );
};

export default PromptInputComponent;
