import React, { useRef, useEffect, useState, useCallback } from "react";
import styled from "styled-components";
import { type ToolMode, type InteractionData } from "../types";

const CanvasContainer = styled.div<{ disabled?: boolean; toolMode: ToolMode }>`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
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
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(
    null
  );

  const getImageBounds = useCallback(() => {
    if (!imageRef?.current || !canvasRef.current) return null;

    const img = imageRef.current;
    const canvas = canvasRef.current;

    // 이미지의 실제 크기와 표시 크기
    const imgRect = img.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();

    // 캔버스 내에서 이미지의 상대적 위치 계산
    const imageLeft = imgRect.left - canvasRect.left;
    const imageTop = imgRect.top - canvasRect.top;

    console.log("이미지 바운드 계산:", {
      imgRect: {
        left: imgRect.left,
        top: imgRect.top,
        width: imgRect.width,
        height: imgRect.height,
      },
      canvasRect: {
        left: canvasRect.left,
        top: canvasRect.top,
        width: canvasRect.width,
        height: canvasRect.height,
      },
      imageLeft,
      imageTop,
    });

    return {
      left: imageLeft,
      top: imageTop,
      width: imgRect.width,
      height: imgRect.height,
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

      console.log("마우스 위치 계산:", {
        clientX,
        clientY,
        canvasRect: { left: canvasRect.left, top: canvasRect.top },
        canvasX,
        canvasY,
        bounds,
      });

      // 이미지 영역 내부인지 확인
      if (
        canvasX >= bounds.left &&
        canvasX <= bounds.left + bounds.width &&
        canvasY >= bounds.top &&
        canvasY <= bounds.top + bounds.height
      ) {
        // 이미지 내에서의 상대적 위치 (0~1)
        const relativeX = (canvasX - bounds.left) / bounds.width;
        const relativeY = (canvasY - bounds.top) / bounds.height;

        console.log("상대 위치:", { relativeX, relativeY });

        return {
          x: relativeX,
          y: relativeY,
        };
      }

      console.log("이미지 영역 외부 클릭");
      return null;
    },
    [getImageBounds]
  );

  const drawPoint = useCallback(
    (x: number, y: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const bounds = getImageBounds();
      if (!bounds) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 상대 좌표(0~1)를 캔버스 좌표로 변환
      const canvasX = bounds.left + x * bounds.width;
      const canvasY = bounds.top + y * bounds.height;

      // 포인트 그리기
      ctx.fillStyle = "#6366f1";
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 6, 0, 2 * Math.PI);
      ctx.fill();

      // 외곽선
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
    },
    [getImageBounds]
  );

  const drawBoundingBox = useCallback(
    (startX: number, startY: number, endX: number, endY: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const bounds = getImageBounds();
      if (!bounds) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 상대 좌표(0~1)를 캔버스 좌표로 변환
      const canvasStartX = bounds.left + startX * bounds.width;
      const canvasStartY = bounds.top + startY * bounds.height;
      const canvasEndX = bounds.left + endX * bounds.width;
      const canvasEndY = bounds.top + endY * bounds.height;

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
    [getImageBounds]
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
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      const container = canvas.parentElement;
      if (!container) return;

      canvas.width = container.offsetWidth;
      canvas.height = container.offsetHeight;
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    return () => {
      window.removeEventListener("resize", resizeCanvas);
    };
  }, []);

  useEffect(() => {
    if (toolMode === "none") {
      clearCanvas();
    }
  }, [toolMode, clearCanvas]);

  return (
    <CanvasContainer
      toolMode={toolMode}
      disabled={disabled}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <Canvas ref={canvasRef} />
    </CanvasContainer>
  );
};

export default InteractionCanvas;
