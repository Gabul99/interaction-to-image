import React from "react";
import { Handle, Position } from "reactflow";
import styled from "styled-components";
import { useImageStore } from "../stores/imageStore";
import { FaBookmark, FaRegBookmark } from "react-icons/fa";

// Outer wrapper - includes hover area for branching button
const NodeWrapper = styled.div`
  position: relative;
  /* Extend hover area above the node for the branching button */
  padding-top: 40px;
  margin-top: -40px;
  
  /* Remove ReactFlow's default selection highlight from wrapper */
  /* ReactFlow applies .react-flow__node.selected class, so we override it */
  &.react-flow__node.selected {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
  }
  
  /* Show branching button on hover */
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

const FeedbackBboxOverlay = styled.div<{ x: number; y: number; width: number; height: number; color: string }>`
  position: absolute;
  left: ${(props) => props.x * 100}%;
  top: ${(props) => props.y * 100}%;
  width: ${(props) => props.width * 100}%;
  height: ${(props) => props.height * 100}%;
  border: 2px solid ${(props) => props.color};
  background: ${(props) => `${props.color}20`};
  pointer-events: none;
  z-index: 10;
  box-shadow: 0 0 8px ${(props) => `${props.color}60`};
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

// Branching button positioned ABOVE the node container
// Now positioned absolutely within the wrapper, not using Portal
const BranchingButton = styled.button`
  position: absolute;
  top: 4px; /* Within the padding-top area */
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

const BookmarkButton = styled.button<{ $isBookmarked: boolean }>`
  position: absolute;
  bottom: 8px;
  right: 8px;
  width: auto;
  height: auto;
  padding: 4px;
  border: none;
  background: transparent;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  z-index: 20;

  svg {
    width: 24px;
    height: 24px;
    color: ${props => props.$isBookmarked 
      ? '#fbbf24' 
      : 'rgba(255, 255, 255, 0.6)'};
    transition: all 0.2s ease;
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.5));
  }

  &:hover {
    svg {
      color: ${props => props.$isBookmarked 
        ? '#f59e0b' 
        : 'rgba(255, 255, 255, 0.9)'};
      transform: scale(1.15);
    }
  }

  &:active {
    svg {
      transform: scale(0.95);
    }
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
  const [isHovered, setIsHovered] = React.useState(false);
  const { hoveredFeedbackEdge, currentGraphSession, toggleBookmark, isBookmarked } = useImageStore();

  const handleBranchClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (data?.onBranchClick) {
      data.onBranchClick();
    }
  };

  // 현재 노드가 hover된 피드백 엣지의 브랜치에 속하는지 확인
  const isInHoveredBranch = React.useMemo(() => {
    if (!hoveredFeedbackEdge || !currentGraphSession) return false;
    
    const branch = currentGraphSession.branches.find((b) => b.id === hoveredFeedbackEdge.branchId);
    if (!branch) return false;
    
    // 브랜치의 nodes 배열에 현재 노드가 포함되어 있는지 확인
    return branch.nodes.includes(id);
  }, [hoveredFeedbackEdge, currentGraphSession, id]);

  // 표시할 BBOX 피드백들 (feedback 정보 포함)
  const visibleBboxes = React.useMemo(() => {
    if (!isInHoveredBranch || !hoveredFeedbackEdge) return [];
    
    return hoveredFeedbackEdge.bboxFeedbacks
      .filter((f) => f.bbox)
      .map((f) => ({
        feedback: f,
        bbox: {
          ...f.bbox!,
          color: "#8b5cf6", // 피드백 BBOX 색상
        },
      }));
  }, [isInHoveredBranch, hoveredFeedbackEdge]);

  React.useEffect(() => {
    if (data?.imageUrl) {
      console.log(`[ImageNode] 이미지 URL 업데이트: nodeId=${id}, imageUrl=${data.imageUrl.substring(0, 100)}, step=${data.step}`);
      setImageLoaded(false);
    } else {
      console.log(`[ImageNode] 이미지 URL 없음: nodeId=${id}, step=${data?.step}`);
    }
  }, [data?.imageUrl, id, data?.step]);

  const bookmarked = isBookmarked(id);
  
  const handleBookmarkClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    toggleBookmark(id);
  };

  return (
    <NodeWrapper
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Branching button - positioned in the padding area above the node, shown on hover */}
      <BranchingButton
        className="branching-button"
        onClick={handleBranchClick}
      >
        ✨ Branching
      </BranchingButton>
      
      <Handle type="target" position={Position.Left} style={{ background: "#6366f1", top: "50%" }} />
      
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

          {/* Feedback BBOX 오버레이 - hover된 피드백 엣지의 브랜치에 속하는 경우 표시 */}
          {isInHoveredBranch && visibleBboxes.map((item, index) => (
            <FeedbackBboxOverlay
              key={item.feedback.id || `feedback-bbox-${index}`}
              x={item.bbox.x}
              y={item.bbox.y}
              width={item.bbox.width}
              height={item.bbox.height}
              color={item.bbox.color}
            />
          ))}
          
          {/* 북마크 버튼 - hover 시 표시 */}
          {isHovered && (
            <BookmarkButton
              $isBookmarked={bookmarked}
              onClick={handleBookmarkClick}
              title={bookmarked ? "북마크 해제" : "북마크 추가"}
            >
              {bookmarked ? <FaBookmark /> : <FaRegBookmark />}
            </BookmarkButton>
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
