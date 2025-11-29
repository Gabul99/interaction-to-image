import React, { useState } from "react";
import { createPortal } from "react-dom";
import { getBezierPath, BaseEdge, EdgeLabelRenderer, type Position } from "reactflow";
import styled from "styled-components";
import { type FeedbackRecord } from "../types";

const EdgeContainer = styled.g`
  cursor: pointer;
`;

const EdgePath = styled.path<{ isHovered: boolean }>`
  stroke: ${(props) => (props.isHovered ? "#8b5cf6" : "#6366f1")};
  stroke-width: ${(props) => (props.isHovered ? 3 : 2)};
  fill: none;
  transition: all 0.2s ease;
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
  background: rgba(26, 26, 46, 0.9);
  border: 1px solid rgba(139, 92, 246, 0.5);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 11px;
  color: #8b5cf6;
  font-weight: 600;
  pointer-events: all;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(26, 26, 46, 1);
    border-color: #8b5cf6;
    transform: scale(1.05);
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
  const hasFeedback = feedbacks.length > 0;

  const getAreaLabel = (area: string) => {
    if (area === "full") return "전체";
    if (area === "bbox") return "BBOX";
    return "포인팅";
  };

  // Edge 중간 지점 계산 (라벨 위치)
  const labelX = (sourceX + targetX) / 2;
  const labelY = (sourceY + targetY) / 2;

  // 피드백 요약 텍스트 생성
  const getFeedbackSummary = () => {
    if (feedbacks.length === 0) return "";
    const textCount = feedbacks.filter(f => f.text).length;
    const imageCount = feedbacks.filter(f => f.imageUrl).length;
    const areaCounts = {
      full: feedbacks.filter(f => f.area === "full").length,
      bbox: feedbacks.filter(f => f.area === "bbox").length,
      point: feedbacks.filter(f => f.area === "point").length,
    };
    
    const parts: string[] = [];
    if (areaCounts.full > 0) parts.push(`전체 ${areaCounts.full}`);
    if (areaCounts.bbox > 0) parts.push(`BBOX ${areaCounts.bbox}`);
    if (areaCounts.point > 0) parts.push(`포인팅 ${areaCounts.point}`);
    
    return parts.join(", ");
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
              피드백: {getFeedbackSummary()}
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
            <TooltipTitle>피드백 정보</TooltipTitle>
            {feedbacks.map((feedback, index) => (
              <FeedbackItem key={feedback.id || index}>
                <FeedbackAreaBadge area={feedback.area}>
                  {getAreaLabel(feedback.area)}
                </FeedbackAreaBadge>
                {feedback.text && (
                  <FeedbackText>{feedback.text}</FeedbackText>
                )}
                {feedback.imageUrl && (
                  <FeedbackText>[참조 이미지]</FeedbackText>
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

