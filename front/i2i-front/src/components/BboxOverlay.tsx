import React, { useRef, useEffect, useState } from "react";
import styled from "styled-components";
import { type BoundingBox } from "../types";

const OverlayContainer = styled.div<{
  left: number;
  top: number;
  width: number;
  height: number;
}>`
  position: absolute;
  left: ${(props) => props.left}px;
  top: ${(props) => props.top}px;
  width: ${(props) => props.width}px;
  height: ${(props) => props.height}px;
  pointer-events: ${(props) => (props.onClick ? "auto" : "none")};
  z-index: ${(props) => (props.onClick ? "15" : "10")};
`;

const Canvas = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
`;

interface BboxOverlayProps {
  bboxes: Array<{
    id: string;
    objectId: string;
    x: number;
    y: number;
    width: number;
    height: number;
    color: string;
  }>;
  objects?: Array<{
    id: string;
    label: string;
    color: string;
  }>; // 객체 정보 (라벨 표시용)
  imageRef?: React.RefObject<HTMLImageElement | null>;
  selectedBboxId?: string | null;
  onBboxClick?: (bboxId: string, bbox: { x: number; y: number; width: number; height: number }) => void;
  onClearSelection?: () => void; // 외부 클릭 시 선택 취소
}

const BboxOverlay: React.FC<BboxOverlayProps> = ({
  bboxes,
  objects = [],
  imageRef,
  selectedBboxId,
  onBboxClick,
  onClearSelection,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasPosition, setCanvasPosition] = useState({
    left: 0,
    top: 0,
    width: 0,
    height: 0,
  });

  // 이미지 위치와 크기에 맞춰 Canvas 위치 업데이트
  useEffect(() => {
    const updateCanvasPosition = () => {
      if (!imageRef?.current || !containerRef.current) {
        return;
      }

      const img = imageRef.current;
      const imgRect = img.getBoundingClientRect();

      // ImageContainer는 position: relative이므로, BboxOverlay도 같은 부모 내에 있음
      // ImageContainer를 찾기 (직접 부모가 ImageContainer)
      const parentContainer = containerRef.current.parentElement;
      if (!parentContainer) {
        return;
      }

      const containerRect = parentContainer.getBoundingClientRect();

      const left = imgRect.left - containerRect.left;
      const top = imgRect.top - containerRect.top;

      setCanvasPosition({
        left,
        top,
        width: imgRect.width,
        height: imgRect.height,
      });

      if (canvasRef.current) {
        canvasRef.current.width = imgRect.width;
        canvasRef.current.height = imgRect.height;
      }
    };

    // 초기 업데이트 (이미지가 로드되기 전일 수 있으므로 약간의 지연)
    const timeoutId = setTimeout(updateCanvasPosition, 100);

    const img = imageRef?.current;
    if (img) {
      if (img.complete) {
        updateCanvasPosition();
      } else {
        img.addEventListener("load", updateCanvasPosition);
      }
    }

    window.addEventListener("resize", updateCanvasPosition);

    return () => {
      clearTimeout(timeoutId);
      if (img) {
        img.removeEventListener("load", updateCanvasPosition);
      }
      window.removeEventListener("resize", updateCanvasPosition);
    };
  }, [imageRef]);

  // BBOX 그리기
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvas.width === 0 || canvas.height === 0) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    bboxes.forEach((bbox) => {
      const x = bbox.x * canvas.width;
      const y = bbox.y * canvas.height;
      const width = bbox.width * canvas.width;
      const height = bbox.height * canvas.height;

      // 선택된 BBOX는 더 두껍게
      const isSelected = selectedBboxId === bbox.id;
      
      // 선택된 BBOX는 더 강조된 스타일
      if (isSelected) {
        // 선택된 BBOX: 더 두꺼운 테두리와 강조된 채우기
        ctx.strokeStyle = bbox.color;
        ctx.lineWidth = 4;
        ctx.setLineDash([]);
        ctx.strokeRect(x, y, width, height);
        
        // 외곽선 (글로우 효과)
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 1, y - 1, width + 2, height + 2);
        
        // 채우기 (더 진하게)
        ctx.fillStyle = `${bbox.color}30`;
        ctx.fillRect(x, y, width, height);
      } else {
        // 일반 BBOX
        ctx.strokeStyle = bbox.color;
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.strokeRect(x, y, width, height);
        
        // 채우기
        ctx.fillStyle = `${bbox.color}15`;
        ctx.fillRect(x, y, width, height);
      }

      // 객체 라벨 표시 (왼쪽 상단)
      const object = objects.find((obj) => obj.id === bbox.objectId);
      if (object) {
        const labelText = object.label;
        const padding = 6;
        const fontSize = 14;
        const fontFamily = "system-ui, -apple-system, sans-serif";
        
        ctx.font = `bold ${fontSize}px ${fontFamily}`;
        const textMetrics = ctx.measureText(labelText);
        const textWidth = textMetrics.width;
        const textHeight = fontSize;
        
        // 라벨 배경 (둥근 사각형)
        const labelX = x + padding;
        const labelY = y + padding;
        const labelWidth = textWidth + padding * 2;
        const labelHeight = textHeight + padding;
        
        // 배경 그리기 (둥근 사각형)
        ctx.fillStyle = bbox.color;
        ctx.beginPath();
        const radius = 4;
        ctx.moveTo(labelX + radius, labelY);
        ctx.lineTo(labelX + labelWidth - radius, labelY);
        ctx.quadraticCurveTo(labelX + labelWidth, labelY, labelX + labelWidth, labelY + radius);
        ctx.lineTo(labelX + labelWidth, labelY + labelHeight - radius);
        ctx.quadraticCurveTo(labelX + labelWidth, labelY + labelHeight, labelX + labelWidth - radius, labelY + labelHeight);
        ctx.lineTo(labelX + radius, labelY + labelHeight);
        ctx.quadraticCurveTo(labelX, labelY + labelHeight, labelX, labelY + labelHeight - radius);
        ctx.lineTo(labelX, labelY + radius);
        ctx.quadraticCurveTo(labelX, labelY, labelX + radius, labelY);
        ctx.closePath();
        ctx.fill();
        
        // 텍스트 그리기
        ctx.fillStyle = "#ffffff";
        ctx.textBaseline = "top";
        ctx.textAlign = "left";
        ctx.fillText(labelText, labelX + padding, labelY + padding / 2);
      }
    });
  }, [bboxes, objects, selectedBboxId, canvasPosition]);

  const handleClick = (e: React.MouseEvent) => {
    if (!onBboxClick || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const canvasRect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - canvasRect.left;
    const canvasY = e.clientY - canvasRect.top;

    // 클릭한 위치의 BBOX 찾기 (역순으로 검사)
    for (let i = bboxes.length - 1; i >= 0; i--) {
      const bbox = bboxes[i];
      const bboxX = bbox.x * canvas.width;
      const bboxY = bbox.y * canvas.height;
      const bboxWidth = bbox.width * canvas.width;
      const bboxHeight = bbox.height * canvas.height;

      if (
        canvasX >= bboxX &&
        canvasX <= bboxX + bboxWidth &&
        canvasY >= bboxY &&
        canvasY <= bboxY + bboxHeight
      ) {
        onBboxClick(bbox.id, {
          x: bbox.x,
          y: bbox.y,
          width: bbox.width,
          height: bbox.height,
        });
        return;
      }
    }

    // BBOX 외부를 클릭한 경우 선택 취소
    if (onClearSelection) {
      onClearSelection();
    }
  };

  if (bboxes.length === 0) return null;

  return (
    <OverlayContainer
      ref={containerRef}
      left={canvasPosition.left}
      top={canvasPosition.top}
      width={canvasPosition.width}
      height={canvasPosition.height}
      onClick={(onBboxClick || onClearSelection) ? handleClick : undefined}
      style={{ cursor: (onBboxClick || onClearSelection) ? "pointer" : "default" }}
    >
      <Canvas ref={canvasRef} />
    </OverlayContainer>
  );
};

export default BboxOverlay;

