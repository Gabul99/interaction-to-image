import React from "react";
import { Handle, Position } from "reactflow";
import styled from "styled-components";

const NodeContainer = styled.div<{ selected: boolean }>`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border: 2px solid ${(props) => (props.selected ? "#6366f1" : "rgba(255, 255, 255, 0.2)")};
  border-radius: 12px;
  padding: 16px 20px;
  min-width: 200px;
  max-width: 300px;
  box-shadow: ${(props) =>
    props.selected
      ? "0 8px 32px rgba(99, 102, 241, 0.4)"
      : "0 4px 16px rgba(0, 0, 0, 0.2)"};
  transition: all 0.2s ease;
  cursor: grab;
  
  &:active {
    cursor: grabbing;
  }
`;

const PromptText = styled.div`
  color: #f9fafb;
  font-size: 14px;
  font-weight: 500;
  word-wrap: break-word;
  line-height: 1.4;
`;

const NodeLabel = styled.div`
  color: #9ca3af;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
`;

interface PromptNodeData {
  prompt?: string;
  sessionId?: string;
}

interface PromptNodeProps {
  data: PromptNodeData;
  selected?: boolean;
  id: string;
}

const PromptNode: React.FC<PromptNodeProps> = ({ data, selected }) => {
  return (
    <>
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <NodeContainer selected={selected || false}>
        <NodeLabel>프롬프트</NodeLabel>
        <PromptText>{data?.prompt || "프롬프트 없음"}</PromptText>
      </NodeContainer>
      <Handle type="source" position={Position.Right} style={{ background: "#6366f1" }} />
    </>
  );
};

export default PromptNode;

