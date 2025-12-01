import React from "react";
import { Handle, Position } from "reactflow";
import styled from "styled-components";

// Outer wrapper that includes the hover area for the branching button
const NodeWrapper = styled.div`
  position: relative;
  padding-top: 36px; /* Space for the branching button above */
  
  &:hover .branching-button {
    opacity: 1;
    pointer-events: auto;
  }
`;

const NodeContainer = styled.div<{ selected: boolean; isMergeTarget?: boolean }>`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border: 2px solid ${(props) => (props.selected ? "#6366f1" : "rgba(255, 255, 255, 0.2)")};
  border-radius: 12px;
  padding: 8px;
  min-width: 180px;
  max-width: 220px;
  box-shadow: ${(props) =>
    props.selected
      ? "0 8px 32px rgba(99, 102, 241, 0.4)"
      : "0 4px 16px rgba(0, 0, 0, 0.2)"};
  transition: all 0.2s ease;
  position: relative;
`;

const ImageWrapper = styled.div`
  position: relative;
  width: 100%;
  aspect-ratio: 1;
  border-radius: 8px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.3);
  min-height: 150px;
`;

const PlaceholderImage = styled.div`
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
  font-size: 12px;
  position: absolute;
  top: 0;
  left: 0;
`;

const Image = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
  position: relative;
  z-index: 1;
`;

const StepBadge = styled.div`
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  font-size: 11px;
  font-weight: 600;
  padding: 4px 8px;
  border-radius: 4px;
`;

const NodeLabel = styled.div`
  color: #9ca3af;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-top: 8px;
  text-align: center;
`;

// Branching button positioned ABOVE the node container but within the wrapper's hover area
const BranchingButton = styled.button`
  position: absolute;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  border: none;
  border-radius: 6px;
  padding: 6px 14px;
  color: white;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  opacity: 0;
  pointer-events: none;
  transition: all 0.2s ease;
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
  z-index: 100;
  white-space: nowrap;

  &:hover {
    transform: translateX(-50%) scale(1.05);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.5);
  }

  &:active {
    transform: translateX(-50%) scale(0.98);
  }
`;

// Merge indicator overlay
const MergeIndicator = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  border-radius: 12px;
  padding: 16px 24px;
  color: white;
  font-size: 16px;
  font-weight: 700;
  z-index: 50;
  box-shadow: 0 8px 32px rgba(16, 185, 129, 0.5);
  display: flex;
  align-items: center;
  gap: 8px;
  animation: pulse 1s ease-in-out infinite;

  @keyframes pulse {
    0%, 100% {
      transform: translate(-50%, -50%) scale(1);
    }
    50% {
      transform: translate(-50%, -50%) scale(1.05);
    }
  }
`;

// Arrow indicator for the rightmost (leaf) node - points outward to the right
const ArrowIndicator = styled.div<{ color: string }>`
  position: absolute;
  right: -50px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  align-items: center;
  gap: 4px;
  animation: arrowPulse 1.5s ease-in-out infinite;

  @keyframes arrowPulse {
    0%, 100% {
      opacity: 1;
      transform: translateY(-50%) translateX(0);
    }
    50% {
      opacity: 0.7;
      transform: translateY(-50%) translateX(5px);
    }
  }

  &::before {
    content: '';
    width: 25px;
    height: 4px;
    background: ${props => props.color};
    border-radius: 2px;
  }

  &::after {
    content: '';
    width: 0;
    height: 0;
    border-top: 8px solid transparent;
    border-bottom: 8px solid transparent;
    border-left: 12px solid ${props => props.color};
  }
`;

interface ImageNodeData {
  imageUrl?: string;
  step?: number;
  sessionId?: string;
  onBranchClick?: () => void;
  isMergeTarget?: boolean;
  isInBranch?: boolean;
  isRightmost?: boolean;
  branchColor?: string;
}

interface ImageNodeProps {
  data: ImageNodeData;
  selected?: boolean;
  id: string;
}

const ImageNode: React.FC<ImageNodeProps> = ({ data, selected, id }) => {
  const [imageLoaded, setImageLoaded] = React.useState(false);
  const handleBranchClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (data?.onBranchClick) {
      data.onBranchClick();
    }
  };

  React.useEffect(() => {
    if (data?.imageUrl) {
      console.log(`[ImageNode] 이미지 URL 업데이트: nodeId=${id}, imageUrl=${data.imageUrl.substring(0, 100)}, step=${data.step}`);
      setImageLoaded(false);
    } else {
      console.log(`[ImageNode] 이미지 URL 없음: nodeId=${id}, step=${data?.step}`);
    }
  }, [data?.imageUrl, id, data?.step]);

  return (
    <NodeWrapper>
      <Handle type="target" position={Position.Left} style={{ background: "#6366f1", top: "50%" }} />
      
      {/* Branching button - above the node but within hover area */}
      <BranchingButton className="branching-button" onClick={handleBranchClick}>
        ✨ Branching
      </BranchingButton>
      
      <NodeContainer selected={selected || false} isMergeTarget={data?.isMergeTarget}>
        <ImageWrapper>
          {/* Placeholder - 항상 표시 (이미지 로드 전까지) */}
          {!imageLoaded && (
            <PlaceholderImage>
              {data?.step !== undefined ? `Step ${data.step}` : "로딩 중..."}
            </PlaceholderImage>
          )}
          {/* 실제 이미지 */}
          {data?.imageUrl && (
            <Image
              src={data.imageUrl}
              alt={`Step ${data.step || ""}`}
              onLoad={() => {
                console.log(`[ImageNode] 이미지 로드 완료: nodeId=${id}, step=${data.step}`);
                setImageLoaded(true);
              }}
              onError={(e) => {
                console.error(`[ImageNode] 이미지 로드 실패: nodeId=${id}, step=${data.step}, url=${data.imageUrl}`);
                console.error(e);
              }}
              style={{ opacity: imageLoaded ? 1 : 0 }}
            />
          )}
          {data?.step !== undefined && imageLoaded && (
            <StepBadge>Step {data.step}</StepBadge>
          )}
          
          {/* Merge indicator - shown when this node is a potential merge target */}
          {data?.isMergeTarget && (
            <MergeIndicator>
              🔀 Merge
            </MergeIndicator>
          )}
        </ImageWrapper>
        <NodeLabel>Step {data?.step !== undefined ? data.step : "?"}</NodeLabel>
        
        {/* Arrow indicator for rightmost node - points outward */}
        {data?.isRightmost && data?.branchColor && (
          <ArrowIndicator color={data.branchColor} />
        )}
      </NodeContainer>
      
      <Handle type="source" position={Position.Right} style={{ background: "#6366f1", top: "50%" }} />
    </NodeWrapper>
  );
};

export default ImageNode;
