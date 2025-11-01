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
  onSendPrompt: (prompt: string) => void;
  disabled?: boolean;
}

const PromptInputComponent: React.FC<PromptInputProps> = ({
  onSendPrompt,
  disabled = false,
}) => {
  const [prompt, setPrompt] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !disabled) {
      onSendPrompt(prompt.trim());
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

        <SendButton type="submit" disabled={disabled || !prompt.trim()}>
          이미지 생성 시작
        </SendButton>
      </form>
    </PromptContainer>
  );
};

export default PromptInputComponent;
