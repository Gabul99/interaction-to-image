import React from "react";
import styled from "styled-components";
import { type ToolMode } from "../types";

const ToolbarContainer = styled.div`
  position: fixed;
  left: 20px;
  top: 50%;
  transform: translateY(-50%);
  z-index: 1000;
  background: rgba(26, 26, 46, 0.9);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const ToolButton = styled.button<{ active?: boolean }>`
  width: 48px;
  height: 48px;
  border: 2px solid ${(props) => (props.active ? "#6366f1" : "#374151")};
  border-radius: 8px;
  background: ${(props) => (props.active ? "#6366f1" : "transparent")};
  color: ${(props) => (props.active ? "white" : "#9ca3af")};
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  transition: all 0.2s ease;

  &:hover {
    border-color: #6366f1;
    background: ${(props) =>
      props.active ? "#6366f1" : "rgba(99, 102, 241, 0.1)"};
    color: ${(props) => (props.active ? "white" : "#6366f1")};
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const ToolLabel = styled.div`
  font-size: 10px;
  font-weight: 500;
  color: #9ca3af;
  text-align: center;
  margin-top: 4px;
`;

const ToolGroup = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const Divider = styled.div`
  width: 100%;
  height: 1px;
  background: #374151;
  margin: 8px 0;
`;

interface ToolbarProps {
  toolMode: ToolMode;
  onToolChange: (mode: ToolMode) => void;
  disabled?: boolean;
}

const Toolbar: React.FC<ToolbarProps> = ({
  toolMode,
  onToolChange,
  disabled = false,
}) => {
  const tools = [
    {
      mode: "none" as ToolMode,
      icon: "↖",
      label: "선택",
      description: "기본 선택 모드",
    },
    {
      mode: "point" as ToolMode,
      icon: "•",
      label: "포인트",
      description: "이미지에 포인트 선택",
    },
    {
      mode: "bbox" as ToolMode,
      icon: "⊡",
      label: "박스",
      description: "바운딩 박스 선택",
    },
  ];

  return (
    <ToolbarContainer>
      {tools.map((tool, index) => (
        <ToolGroup key={tool.mode}>
          <ToolButton
            active={toolMode === tool.mode}
            onClick={() => onToolChange(tool.mode)}
            disabled={disabled}
            title={tool.description}
          >
            {tool.icon}
          </ToolButton>
          <ToolLabel>{tool.label}</ToolLabel>
          {index < tools.length - 1 && <Divider />}
        </ToolGroup>
      ))}
    </ToolbarContainer>
  );
};

export default Toolbar;
