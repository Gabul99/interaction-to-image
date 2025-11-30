import React from "react";
import styled from "styled-components";
import { type ToolMode } from "../types";

const ToolboxContainer = styled.div<{ visible: boolean }>`
  position: absolute;
  right: 20px;
  top: 50%;
  transform: translateY(-50%);
  z-index: 100;
  display: ${(props) => (props.visible ? "flex" : "none")};
  flex-direction: column;
  gap: 12px;
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const ColorIndicator = styled.div<{ color: string }>`
  width: 40px;
  height: 40px;
  border-radius: 8px;
  background: ${(props) => props.color};
  border: 2px solid rgba(255, 255, 255, 0.3);
  margin-bottom: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
`;

const ToolButton = styled.button<{ active?: boolean; disabled?: boolean }>`
  width: 48px;
  height: 48px;
  border: 2px solid ${(props) => (props.active ? "#6366f1" : "#374151")};
  border-radius: 8px;
  background: ${(props) => (props.active ? "#6366f1" : "transparent")};
  color: ${(props) => (props.active ? "white" : props.disabled ? "#6b7280" : "#9ca3af")};
  cursor: ${(props) => (props.disabled ? "not-allowed" : "pointer")};
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  transition: all 0.2s ease;
  opacity: ${(props) => (props.disabled ? 0.5 : 1)};

  &:hover:not(:disabled) {
    border-color: #6366f1;
    background: ${(props) => (props.active ? "#6366f1" : "rgba(99, 102, 241, 0.1)")};
    color: ${(props) => (props.active ? "white" : "#6366f1")};
  }

  &:disabled {
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

interface FloatingToolboxProps {
  toolMode: ToolMode;
  onToolChange: (mode: ToolMode) => void;
  selectedObjectColor?: string | null;
  disabled?: boolean;
  // 활성화할 도구 목록 (지정하지 않으면 모든 도구 활성화)
  enabledTools?: ToolMode[];
  // 객체 선택이 필수인지 여부 (새 영역 지정 시 false)
  requireObject?: boolean;
}

const FloatingToolbox: React.FC<FloatingToolboxProps> = ({
  toolMode,
  onToolChange,
  selectedObjectColor,
  disabled = false,
  enabledTools,
  requireObject = true,
}) => {
  const hasSelectedObject = !!selectedObjectColor;

  const allTools = [
    {
      mode: "select" as ToolMode,
      icon: "↖",
      label: "선택",
      description: "BBOX 선택 및 이동",
    },
    {
      mode: "bbox" as ToolMode,
      icon: "⊡",
      label: "박스",
      description: "바운딩 박스 그리기",
      requiresObject: true,
    },
    {
      mode: "sketch" as ToolMode,
      icon: "✏️",
      label: "스케치",
      description: "스케치 그리기",
      requiresObject: true,
    },
    {
      mode: "eraser" as ToolMode,
      icon: "🧹",
      label: "지우개",
      description: "스케치 지우기",
      requiresObject: true,
    },
  ];

  // enabledTools가 지정되면 해당 도구만 필터링
  const tools = enabledTools
    ? allTools.filter((tool) => enabledTools.includes(tool.mode))
    : allTools;

  return (
    <ToolboxContainer visible={!disabled}>
      {selectedObjectColor && (
        <>
          <ColorIndicator color={selectedObjectColor} />
          <Divider />
        </>
      )}
      {tools.map((tool, index) => {
        const isDisabled =
          requireObject && tool.requiresObject && !hasSelectedObject;
        return (
          <ToolGroup key={tool.mode}>
            <ToolButton
              active={toolMode === tool.mode}
              disabled={isDisabled || disabled}
              onClick={() => !isDisabled && !disabled && onToolChange(tool.mode)}
              title={tool.description}
            >
              {tool.icon}
            </ToolButton>
            <ToolLabel>{tool.label}</ToolLabel>
            {index < tools.length - 1 && <Divider />}
          </ToolGroup>
        );
      })}
    </ToolboxContainer>
  );
};

export default FloatingToolbox;

