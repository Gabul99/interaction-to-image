import React, { useState, useMemo } from "react";
import { useReactFlow } from "reactflow";
import { getBezierPath, BaseEdge, EdgeLabelRenderer, type Position } from "reactflow";
import styled from "styled-components";
import { type FeedbackRecord } from "../types";
import { useImageStore } from "../stores/imageStore";

const EdgeContainer = styled.g`
  cursor: pointer;
`;

const FeedbackAreaBadge = styled.span<{ area: string }>`
  display: inline-block;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 8px;
  font-weight: 600;
  background: ${(props) => {
    if (props.area === "full") return "rgba(99, 102, 241, 0.2)";
    if (props.area === "bbox") return "rgba(139, 92, 246, 0.2)";
    return "rgba(236, 72, 153, 0.2)";
  }};
  color: ${(props) => {
    if (props.area === "full") return "#6366f1";
    if (props.area === "bbox") return "#8b5cf6";
    return "#ec4899";
  }};
`;

const EdgeLabel = styled.div<{ $isHovered?: boolean; $borderColor?: string }>`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(8px);
  border: 2px solid ${(props) => props.$borderColor || "#8b5cf6"};
  border-radius: 8px;
  padding: 0px 0px;
  font-size: 10px;
  color: #f9fafb;
  pointer-events: all;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: ${(props) => props.$isHovered 
    ? `0 4px 12px ${props.$borderColor || "rgba(139, 92, 246, 0.5)"}60` 
    : "0 2px 8px rgba(0, 0, 0, 0.3)"};
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 100px;
  max-width: 180px;
  transform: ${(props) => props.$isHovered ? "scale(1.02)" : "scale(1)"};
  z-index: 1000;

  &:hover {
    background: rgba(26, 26, 46, 1);
    transform: scale(1.02);
  }
`;

const FeedbackImageThumbnail = styled.img`
  width: 100%;
  max-width: 160px;
  height: auto;
  max-height: 80px;
  object-fit: contain;
  border-radius: 4px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: 4px;
  background: rgba(0, 0, 0, 0.2);
`;

const FeedbackItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px 8px;
  background: rgba(55, 65, 81, 0.4);
  border-radius: 4px;
  border: 1px solid rgba(255, 255, 255, 0.05);
`;

const FeedbackItemHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
`;

const GuidanceScaleBadge = styled.span`
  display: inline-block;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 8px;
  font-weight: 600;
  background: rgba(99, 102, 241, 0.2);
  color: #6366f1;
  border: 1px solid rgba(99, 102, 241, 0.3);
`;

const FeedbackTextContent = styled.div`
  color: #d1d5db;
  font-size: 12px;
  line-height: 1.3;
  word-break: break-word;
  max-height: 32px;
  overflow: hidden;
  text-overflow: ellipsis;
`;


interface FeedbackEdgeData {
  feedback?: FeedbackRecord[];
  branchId?: string;
}

interface FeedbackEdgeProps {
  id: string;
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  sourcePosition: Position;
  targetPosition: Position;
  style?: React.CSSProperties;
  data?: FeedbackEdgeData;
  markerEnd?: string;
}

const FeedbackEdge: React.FC<FeedbackEdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
  markerEnd,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const { getEdges } = useReactFlow();
  const { currentGraphSession, setHoveredFeedbackEdge } = useImageStore();

  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const handleMouseEnter = () => {
    setIsHovered(true);
    // BBOX 피드백이 있는 경우 hover 상태 업데이트
    if (branchId && feedbacks.length > 0) {
      const bboxFeedbacks = feedbacks.filter((f) => f.area === "bbox" && f.bbox);
      if (bboxFeedbacks.length > 0) {
        setHoveredFeedbackEdge(branchId, bboxFeedbacks);
      }
    }
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    // hover 해제 시 상태 초기화
    setHoveredFeedbackEdge(null);
  };

  const feedbacks = data?.feedback || [];
  const branchId = data?.branchId;
  const isBranchEdge = branchId && branchId !== "B0";
  
  // Get edge color from style prop (set by GraphCanvas based on branch)
  const edgeColor = style?.stroke as string || "#8b5cf6";
  
  // Check if this is the first edge of the branch (the edge that forks from the main branch)
  const isFirstBranchEdge = useMemo(() => {
    if (!isBranchEdge || !currentGraphSession || !branchId) {
      return false;
    }
    
    // Find the branch
    const branch = currentGraphSession.branches.find((b) => b.id === branchId);
    if (!branch) return false;
    
    // Get this edge from ReactFlow
    const edges = getEdges();
    const thisEdge = edges.find((e) => e.id === id);
    if (!thisEdge) return false;
    
    // Check if target is the first node in the branch
    // The first node in a branch is the one connected from the branch's sourceNodeId
    if (branch.nodes.length > 0) {
      const firstNodeId = branch.nodes[0];
      return thisEdge.target === firstNodeId;
    }
    
    // If branch has no nodes yet, check if this edge connects from the source node
    // This handles the case when the branch is just created
    if (branch.sourceNodeId) {
      // Check if source is the branch's sourceNodeId or its parent
      const sourceEdge = currentGraphSession.edges.find(
        (e) => e.target === branch.sourceNodeId
      );
      const expectedSource = sourceEdge ? sourceEdge.source : branch.sourceNodeId;
      return thisEdge.source === expectedSource;
    }
    
    return false;
  }, [isBranchEdge, currentGraphSession, branchId, id, getEdges]);
  
  // Only show feedback for the first edge of a branch (the fork edge)
  const hasFeedback = feedbacks.length > 0 && isBranchEdge && isFirstBranchEdge;

  const getAreaLabel = (area: string) => {
    if (area === "full") return "Full";
    if (area === "bbox") return "Region";
    if (area === "sketch") return "Sketch";
    if (area === "point") return "Pointing";
    return area;
  };

  // Edge 중간 지점 계산 (라벨 위치)
  const labelX = (sourceX + targetX) / 2;
  const labelY = (sourceY + targetY) / 2;
  

  return (
    <>
      <EdgeContainer
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <BaseEdge
          id={id}
          path={edgePath}
          markerEnd={markerEnd}
          style={style}
        />
      </EdgeContainer>
      {hasFeedback && (
        <>
          <EdgeLabelRenderer>
              <EdgeLabel
              $isHovered={isHovered}
              $borderColor={edgeColor}
              style={{
                position: "absolute",
                transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
                pointerEvents: "all",
                zIndex: 1000,
              }}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
            >
              {feedbacks.map((feedback, index) => (
                <FeedbackItem key={feedback.id || index}>
                  <FeedbackItemHeader>
                    <GuidanceScaleBadge>
                      {getAreaLabel(feedback.area)}
                    </GuidanceScaleBadge>
                    <GuidanceScaleBadge>
                      {feedback.type === "text" ? "Temp" : "Temp"}: {feedback.guidanceScale?.toFixed(1) ?? (feedback.type === "image" ? "5.0" : "2.0")}
                    </GuidanceScaleBadge>
                  </FeedbackItemHeader>
                  {feedback.text && (
                    <FeedbackTextContent>
                      {feedback.text.length > 60 ? feedback.text.substring(0, 60) + "..." : feedback.text}
                    </FeedbackTextContent>
                  )}
                  {feedback.imageUrl && (
                    <FeedbackImageThumbnail
                      src={feedback.imageUrl}
                      alt="피드백 이미지"
                      onError={(e) => {
                        // 이미지 로드 실패 시 숨김 처리
                        e.currentTarget.style.display = "none";
                      }}
                    />
                  )}
                </FeedbackItem>
              ))}
            </EdgeLabel>
          </EdgeLabelRenderer>
        </>
      )}
    </>
  );
};

export default FeedbackEdge;

