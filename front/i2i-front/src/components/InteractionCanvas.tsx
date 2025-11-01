import React, { useRef, useEffect, useState, useCallback } from "react";
import styled from "styled-components";
import { type ToolMode, type InteractionData } from "../types";

const CanvasContainer = styled.div<{
  disabled?: boolean;
  toolMode: ToolMode;
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
  pointer-events: ${(props) => (props.disabled ? "none" : "auto")};
  cursor: ${(props) => {
    switch (props.toolMode) {
      case "point":
        return "crosshair";
      case "bbox":
        return "crosshair";
      default:
        return "default";
    }
  }};
`;

const Canvas = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
`;

interface InteractionCanvasProps {
  toolMode: ToolMode;
  disabled?: boolean;
  onInteraction: (data: InteractionData) => void;
  onClearSelection?: () => void;
  imageRef?: React.RefObject<HTMLImageElement | null>;
}

const InteractionCanvas: React.FC<InteractionCanvasProps> = ({
  toolMode,
  disabled = false,
  onInteraction,
  onClearSelection,
  imageRef,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(
    null
  );
  const [canvasPosition, setCanvasPosition] = useState({
    left: 0,
    top: 0,
    width: 0,
    height: 0,
  });

  // 이미지 위치와 크기에 맞춰 Canvas 위치 업데이트
  useEffect(() => {
    const updateCanvasPosition = () => {
      if (!imageRef?.current || !containerRef.current) return;

      const img = imageRef.current;
      const imgRect = img.getBoundingClientRect();

      // 부모 컨테이너 찾기 (ImageContainer)
      const parentContainer = containerRef.current.parentElement;
      if (!parentContainer) return;

      const containerRect = parentContainer.getBoundingClientRect();

      // 컨테이너 기준 이미지 위치
      const left = imgRect.left - containerRect.left;
      const top = imgRect.top - containerRect.top;

      setCanvasPosition({
        left,
        top,
        width: imgRect.width,
        height: imgRect.height,
      });

      // Canvas 크기도 업데이트
      if (canvasRef.current) {
        canvasRef.current.width = imgRect.width;
        canvasRef.current.height = imgRect.height;
      }
    };

    updateCanvasPosition();

    // 이미지 로드 시 업데이트
    const img = imageRef?.current;
    if (img) {
      img.addEventListener("load", updateCanvasPosition);
    }

    // 윈도우 리사이즈 시 업데이트
    window.addEventListener("resize", updateCanvasPosition);

    return () => {
      if (img) {
        img.removeEventListener("load", updateCanvasPosition);
      }
      window.removeEventListener("resize", updateCanvasPosition);
    };
  }, [imageRef]);

  const getImageBounds = useCallback(() => {
    if (!imageRef?.current || !canvasRef.current) return null;

    // Canvas가 이미 이미지와 정확히 같은 크기와 위치이므로
    // 이미지 내에서의 상대 위치는 단순히 Canvas 내 위치를 이미지 크기로 나눈 값
    return {
      left: 0,
      top: 0,
      width: canvasRef.current.width,
      height: canvasRef.current.height,
    };
  }, [imageRef]);

  const getRelativePosition = useCallback(
    (clientX: number, clientY: number) => {
      const bounds = getImageBounds();
      if (!bounds || !canvasRef.current) return null;

      const canvasRect = canvasRef.current.getBoundingClientRect();

      // 캔버스 내에서의 마우스 위치
      const canvasX = clientX - canvasRect.left;
      const canvasY = clientY - canvasRect.top;

      // Canvas가 이미지와 정확히 같은 크기이므로
      // Canvas 내 위치가 이미지 내 위치와 같음
      if (
        canvasX >= 0 &&
        canvasX <= bounds.width &&
        canvasY >= 0 &&
        canvasY <= bounds.height
      ) {
        // 이미지 내에서의 상대적 위치 (0~1)
        const relativeX = canvasX / bounds.width;
        const relativeY = canvasY / bounds.height;

        return {
          x: relativeX,
          y: relativeY,
        };
      }

      return null;
    },
    [getImageBounds]
  );

  const drawPoint = useCallback((x: number, y: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 상대 좌표(0~1)를 캔버스 픽셀 좌표로 변환
    const canvasX = x * canvas.width;
    const canvasY = y * canvas.height;

    // 포인트 그리기
    ctx.fillStyle = "#6366f1";
    ctx.beginPath();
    ctx.arc(canvasX, canvasY, 6, 0, 2 * Math.PI);
    ctx.fill();

    // 외곽선
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();
  }, []);

  const drawBoundingBox = useCallback(
    (startX: number, startY: number, endX: number, endY: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 상대 좌표(0~1)를 캔버스 픽셀 좌표로 변환
      const canvasStartX = startX * canvas.width;
      const canvasStartY = startY * canvas.height;
      const canvasEndX = endX * canvas.width;
      const canvasEndY = endY * canvas.height;

      const x = Math.min(canvasStartX, canvasEndX);
      const y = Math.min(canvasStartY, canvasEndY);
      const width = Math.abs(canvasEndX - canvasStartX);
      const height = Math.abs(canvasEndY - canvasStartY);

      // 바운딩 박스 그리기
      ctx.strokeStyle = "#6366f1";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x, y, width, height);

      // 채우기 (투명도)
      ctx.fillStyle = "rgba(99, 102, 241, 0.1)";
      ctx.fillRect(x, y, width, height);
    },
    []
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (disabled || toolMode === "none") {
        // 이미지 외부 클릭 시 선택 취소
        if (onClearSelection) {
          onClearSelection();
        }
        return;
      }

      const pos = getRelativePosition(e.clientX, e.clientY);
      if (!pos) {
        // 이미지 외부 클릭 시 선택 취소
        if (onClearSelection) {
          onClearSelection();
        }
        return;
      }

      setIsDrawing(true);
      setStartPos(pos);

      if (toolMode === "point") {
        console.log("포인트 클릭:", { x: pos.x, y: pos.y });
        drawPoint(pos.x, pos.y);
        onInteraction({
          type: "point",
          x: pos.x,
          y: pos.y,
        });
      }
    },
    [
      disabled,
      toolMode,
      getRelativePosition,
      drawPoint,
      onInteraction,
      onClearSelection,
    ]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDrawing || !startPos || toolMode !== "bbox") return;

      const pos = getRelativePosition(e.clientX, e.clientY);
      if (!pos) return;

      drawBoundingBox(startPos.x, startPos.y, pos.x, pos.y);
    },
    [isDrawing, startPos, toolMode, getRelativePosition, drawBoundingBox]
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      if (!isDrawing || !startPos || toolMode !== "bbox") return;

      const pos = getRelativePosition(e.clientX, e.clientY);
      if (!pos) return;

      const width = Math.abs(pos.x - startPos.x);
      const height = Math.abs(pos.y - startPos.y);

      if (width > 0.01 && height > 0.01) {
        // 최소 크기 체크
        console.log("바운딩 박스:", {
          x: Math.min(startPos.x, pos.x),
          y: Math.min(startPos.y, pos.y),
          width,
          height,
        });
        onInteraction({
          type: "bbox",
          x: Math.min(startPos.x, pos.x),
          y: Math.min(startPos.y, pos.y),
          width,
          height,
        });
      }

      setIsDrawing(false);
      setStartPos(null);
    },
    [isDrawing, startPos, toolMode, getRelativePosition, onInteraction]
  );

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  useEffect(() => {
    if (toolMode === "none") {
      clearCanvas();
    }
  }, [toolMode, clearCanvas]);

  return (
    <CanvasContainer
      ref={containerRef}
      toolMode={toolMode}
      disabled={disabled}
      left={canvasPosition.left}
      top={canvasPosition.top}
      width={canvasPosition.width}
      height={canvasPosition.height}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <Canvas ref={canvasRef} />
    </CanvasContainer>
  );
};

export default InteractionCanvas;
