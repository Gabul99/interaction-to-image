import React from "react";
import { Handle, Position } from "reactflow";
import styled, { keyframes } from "styled-components";

// Spinner animation
const spin = keyframes`
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
`;

const NodeWrapper = styled.div`
  position: relative;
`;

const NodeContainer = styled.div`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border: 2px solid rgba(99, 102, 241, 0.5);
  border-radius: 12px;
  padding: 8px;
  min-width: 180px;
  max-width: 220px;
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);
  position: relative;
  animation: pulse 2s ease-in-out infinite;

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
      border-color: rgba(99, 102, 241, 0.5);
    }
    50% {
      opacity: 0.8;
      border-color: rgba(99, 102, 241, 0.8);
    }
  }
`;

const ImageWrapper = styled.div`
  position: relative;
  width: 100%;
  aspect-ratio: 1;
  border-radius: 8px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.3);
  min-height: 150px;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const Spinner = styled.div`
  width: 48px;
  height: 48px;
  border: 4px solid rgba(99, 102, 241, 0.2);
  border-top: 4px solid #6366f1;
  border-radius: 50%;
  animation: ${spin} 1s linear infinite;
`;

const LoadingText = styled.div`
  position: absolute;
  bottom: 8px;
  left: 50%;
  transform: translateX(-50%);
  color: #9ca3af;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

interface LoadingNodeData {
  step?: number;
  parentPromptId?: string;
}

interface LoadingNodeProps {
  data: LoadingNodeData;
  id: string;
}

const SimpleLoadingNode: React.FC<LoadingNodeProps> = ({ data, id }) => {
  return (
    <NodeWrapper>
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: "#6366f1", top: "50%" }}
      />

      <NodeContainer>
        <ImageWrapper>
          <Spinner />
        </ImageWrapper>
        <LoadingText>Generating...</LoadingText>
      </NodeContainer>

      <Handle
        type="source"
        position={Position.Right}
        style={{ background: "#6366f1", top: "50%" }}
      />
    </NodeWrapper>
  );
};

export default SimpleLoadingNode;

