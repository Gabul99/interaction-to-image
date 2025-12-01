import React, { useState, useMemo } from "react";
import { createPortal } from "react-dom";
import { useReactFlow } from "reactflow";
import { getBezierPath, BaseEdge, EdgeLabelRenderer, type Position } from "reactflow";
import styled from "styled-components";
import { type FeedbackRecord } from "../types";
import { useImageStore } from "../stores/imageStore";

const EdgeContainer = styled.g`
  cursor: pointer;
`;

const Tooltip = styled.div<{ x: number; y: number; visible: boolean }>`
  position: fixed;
  left: ${(props) => props.x}px;
  top: ${(props) => props.y}px;
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 12px;
  min-width: 200px;
  max-width: 300px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  z-index: 9999;
  pointer-events: none;
  opacity: ${(props) => (props.visible ? 1 : 0)};
  transition: opacity 0.2s ease;
  transform: translate(-50%, -100%) translateY(-8px);
`;

const TooltipTitle = styled.div`
  color: #f9fafb;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  padding-bottom: 8px;
`;

const FeedbackItem = styled.div`
  margin-bottom: 8px;
  padding: 8px;
  background: rgba(55, 65, 81, 0.5);
  border-radius: 4px;
`;

const FeedbackAreaBadge = styled.span<{ area: string }>`
  display: inline-block;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  margin-right: 6px;
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

const FeedbackText = styled.div`
  color: #d1d5db;
  font-size: 11px;
  margin-top: 4px;
  word-wrap: break-word;
`;

const EdgeLabel = styled.div`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border: 2px solid rgba(139, 92, 246, 0.6);
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 11px;
  color: #8b5cf6;
  font-weight: 600;
  pointer-events: all;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
  display: flex;
  align-items: center;
  gap: 6px;
  white-space: nowrap;

  &:hover {
    background: rgba(26, 26, 46, 1);
    border-color: #8b5cf6;
    transform: scale(1.05);
    box-shadow: 0 6px 16px rgba(139, 92, 246, 0.4);
  }
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
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const { getEdges } = useReactFlow();
  const { currentGraphSession } = useImageStore();

  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const handleMouseEnter = (e: React.MouseEvent) => {
    setIsHovered(true);
    setTooltipPosition({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isHovered) {
      setTooltipPosition({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
  };

  const feedbacks = data?.feedback || [];
  const branchId = data?.branchId;
  const isBranchEdge = branchId && branchId !== "B0";
  
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
    if (area === "full") return "전체";
    if (area === "bbox") return "BBOX";
    if (area === "sketch") return "스케치";
    if (area === "point") return "포인팅";
    return area;
  };
  
  const getTypeLabel = (type: string) => {
    if (type === "text") return "텍스트";
    if (type === "image") return "이미지";
    return type;
  };

  // Edge 중간 지점 계산 (라벨 위치)
  const labelX = (sourceX + targetX) / 2;
  const labelY = (sourceY + targetY) / 2;

  // 피드백 요약 텍스트 생성 - 피드백 종류와 개수 표시
  const getFeedbackSummary = () => {
    if (feedbacks.length === 0) return "";
    
    const areaCounts = {
      full: feedbacks.filter(f => f.area === "full").length,
      bbox: feedbacks.filter(f => f.area === "bbox").length,
      point: feedbacks.filter(f => f.area === "point").length,
      sketch: feedbacks.filter(f => f.area === "sketch").length,
    };
    
    const typeCounts = {
      text: feedbacks.filter(f => f.type === "text" && f.text).length,
      image: feedbacks.filter(f => f.type === "image" && f.imageUrl).length,
    };
    
    const parts: string[] = [];
    
    // Area 정보
    if (areaCounts.full > 0) parts.push(`전체`);
    if (areaCounts.bbox > 0) parts.push(`BBOX`);
    if (areaCounts.point > 0) parts.push(`포인팅`);
    if (areaCounts.sketch > 0) parts.push(`스케치`);
    
    // Type 정보
    if (typeCounts.text > 0) parts.push(`텍스트`);
    if (typeCounts.image > 0) parts.push(`이미지`);
    
    return parts.length > 0 ? parts.join(" · ") : "피드백";
  };
  
  // 피드백 아이콘 생성
  const getFeedbackIcon = () => {
    if (feedbacks.length === 0) return "💬";
    const hasText = feedbacks.some(f => f.text);
    const hasImage = feedbacks.some(f => f.imageUrl);
    if (hasText && hasImage) return "💬🖼️";
    if (hasImage) return "🖼️";
    return "💬";
  };

  return (
    <>
      <EdgeContainer
        onMouseEnter={handleMouseEnter}
        onMouseMove={handleMouseMove}
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
              style={{
                position: "absolute",
                transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
                pointerEvents: "all",
              }}
              onMouseEnter={handleMouseEnter}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            >
              <span>{getFeedbackIcon()}</span>
              <span>{getFeedbackSummary()}</span>
            </EdgeLabel>
          </EdgeLabelRenderer>
        </>
      )}
      {/* 피드백이 있는 경우에만 tooltip 표시 - Portal로 body에 렌더링 */}
      {hasFeedback &&
        typeof document !== "undefined" &&
        createPortal(
          <Tooltip
            x={tooltipPosition.x}
            y={tooltipPosition.y}
            visible={isHovered}
          >
            <TooltipTitle>피드백 정보 ({feedbacks.length}개)</TooltipTitle>
            {feedbacks.map((feedback, index) => (
              <FeedbackItem key={feedback.id || index}>
                <div style={{ display: "flex", gap: "6px", alignItems: "center", marginBottom: "4px" }}>
                  <FeedbackAreaBadge area={feedback.area}>
                    {getAreaLabel(feedback.area)}
                  </FeedbackAreaBadge>
                  <span style={{ 
                    fontSize: "10px", 
                    color: "#9ca3af",
                    padding: "2px 6px",
                    background: "rgba(99, 102, 241, 0.2)",
                    borderRadius: "4px"
                  }}>
                    {getTypeLabel(feedback.type)}
                  </span>
                </div>
                {feedback.text && (
                  <FeedbackText>{feedback.text}</FeedbackText>
                )}
                {feedback.imageUrl && (
                  <FeedbackText style={{ color: "#8b5cf6" }}>🖼️ 참조 이미지</FeedbackText>
                )}
                {feedback.bbox && (
                  <FeedbackText style={{ fontSize: "10px", color: "#6b7280" }}>
                    영역: ({feedback.bbox.x.toFixed(2)}, {feedback.bbox.y.toFixed(2)}) 
                    {feedback.bbox.width.toFixed(2)} × {feedback.bbox.height.toFixed(2)}
                  </FeedbackText>
                )}
              </FeedbackItem>
            ))}
          </Tooltip>,
          document.body
        )}
    </>
  );
};

export default FeedbackEdge;

