import React from "react";
import { Handle, Position } from "reactflow";
import styled from "styled-components";

const NodeContainer = styled.div<{ selected: boolean }>`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border: 2px dashed ${(props) => (props.selected ? "#6366f1" : "rgba(255, 255, 255, 0.3)")};
  border-radius: 12px;
  padding: 24px 32px;
  min-width: 200px;
  max-width: 300px;
  box-shadow: ${(props) =>
    props.selected
      ? "0 8px 32px rgba(99, 102, 241, 0.4)"
      : "0 4px 16px rgba(0, 0, 0, 0.2)"};
  transition: all 0.2s ease;
  cursor: pointer;
  text-align: center;

  &:hover {
    border-color: #6366f1;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
  }
`;

const PlusIcon = styled.div`
  color: #9ca3af;
  font-size: 32px;
  font-weight: 300;
  line-height: 1;
  margin-bottom: 8px;
`;

const NodeLabel = styled.div`
  color: #9ca3af;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

interface PlaceholderNodeData {
  onClick?: () => void;
}

interface PlaceholderNodeProps {
  data: PlaceholderNodeData;
  selected?: boolean;
  id: string;
}

const PlaceholderNode: React.FC<PlaceholderNodeProps> = ({ data, selected, id }) => {
  const handleClick = () => {
    if (data?.onClick) {
      data.onClick();
    }
  };

  return (
    <>
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <NodeContainer selected={selected || false} onClick={handleClick}>
        <PlusIcon>+</PlusIcon>
        <NodeLabel>New Prompt</NodeLabel>
      </NodeContainer>
      <Handle type="source" position={Position.Right} style={{ background: "#6366f1" }} />
    </>
  );
};

export default PlaceholderNode;

