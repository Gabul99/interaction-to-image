import React, { useRef, useEffect, useState, useCallback } from "react";
import styled from "styled-components";
import { type BoundingBox } from "../types";

const CanvasContainer = styled.div<{
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
  pointer-events: auto;
  cursor: crosshair;
`;

const Canvas = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
`;

const DeleteButton = styled.button`
  position: absolute;
  top: -8px;
  right: -8px;
  width: 24px;
  height: 24px;
  background: rgba(239, 68, 68, 0.9);
  border: 2px solid #ffffff;
  border-radius: 50%;
  color: white;
  font-size: 14px;
  font-weight: bold;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  transition: all 0.2s ease;

  &:hover {
    background: #ef4444;
    transform: scale(1.1);
  }
`;

interface CompositionCanvasProps {
  bboxes: BoundingBox[];
  selectedObjectId: string | null;
  selectedObjectColor: string | null;
  onAddBbox: (bbox: Omit<BoundingBox, "id">) => void;
  onUpdateBbox: (bboxId: string, updates: Partial<BoundingBox>) => void;
  onRemoveBbox: (bboxId: string) => void;
  imageRef?: React.RefObject<HTMLImageElement | null>;
  placeholderRef?: React.RefObject<HTMLDivElement | null>;
}

type InteractionMode = "draw" | "move" | "resize" | "none";
type ResizeHandle = "nw" | "ne" | "sw" | "se" | null;

const CompositionCanvas: React.FC<CompositionCanvasProps> = ({
  bboxes,
  selectedObjectId,
  selectedObjectColor,
  onAddBbox,
  onUpdateBbox,
  onRemoveBbox,
  imageRef,
  placeholderRef,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasPosition, setCanvasPosition] = useState({
    left: 0,
    top: 0,
    width: 0,
    height: 0,
  });

  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(
    null
  );
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("none");
  const [selectedBboxId, setSelectedBboxId] = useState<string | null>(null);
  const [resizeHandle, setResizeHandle] = useState<ResizeHandle>(null);
  const [dragOffset, setDragOffset] = useState<{ x: number; y: number } | null>(
    null
  );
  const [currentMousePos, setCurrentMousePos] = useState<{ x: number; y: number } | null>(
    null
  );

  // 이미지 또는 placeholder 위치와 크기에 맞춰 Canvas 위치 업데이트
  useEffect(() => {
    const updateCanvasPosition = () => {
      // imageRef 또는 placeholderRef 중 하나를 사용
      const targetElement = imageRef?.current || placeholderRef?.current;
      if (!targetElement || !containerRef.current) return;

      const targetRect = targetElement.getBoundingClientRect();

      const parentContainer = containerRef.current.parentElement;
      if (!parentContainer) return;

      const containerRect = parentContainer.getBoundingClientRect();

      const left = targetRect.left - containerRect.left;
      const top = targetRect.top - containerRect.top;

      setCanvasPosition({
        left,
        top,
        width: targetRect.width,
        height: targetRect.height,
      });

      if (canvasRef.current) {
        canvasRef.current.width = targetRect.width;
        canvasRef.current.height = targetRect.height;
      }
    };

    updateCanvasPosition();

    const targetElement = imageRef?.current || placeholderRef?.current;
    if (targetElement) {
      if (imageRef?.current) {
        imageRef.current.addEventListener("load", updateCanvasPosition);
      }
    }

    window.addEventListener("resize", updateCanvasPosition);

    return () => {
      if (imageRef?.current) {
        imageRef.current.removeEventListener("load", updateCanvasPosition);
      }
      window.removeEventListener("resize", updateCanvasPosition);
    };
  }, [imageRef, placeholderRef]);

  const getRelativePosition = useCallback(
    (clientX: number, clientY: number) => {
      if (!canvasRef.current) return null;

      const canvasRect = canvasRef.current.getBoundingClientRect();
      const canvasX = clientX - canvasRect.left;
      const canvasY = clientY - canvasRect.top;

      if (
        canvasX >= 0 &&
        canvasX <= canvasRect.width &&
        canvasY >= 0 &&
        canvasY <= canvasRect.height
      ) {
        return {
          x: canvasX / canvasRect.width,
          y: canvasY / canvasRect.height,
        };
      }

      return null;
    },
    []
  );

  const getBboxAtPosition = useCallback(
    (x: number, y: number): { bbox: BoundingBox; handle: ResizeHandle } | null => {
      if (!canvasRef.current) return null;

      const canvas = canvasRef.current;
      const canvasX = x * canvas.width;
      const canvasY = y * canvas.height;

      // 역순으로 검사하여 위에 있는 BBOX를 우선 선택
      for (let i = bboxes.length - 1; i >= 0; i--) {
        const bbox = bboxes[i];
        const bboxX = bbox.x * canvas.width;
        const bboxY = bbox.y * canvas.height;
        const bboxWidth = bbox.width * canvas.width;
        const bboxHeight = bbox.height * canvas.height;

        // 리사이즈 핸들 체크 (8px 영역)
        const handleSize = 8;
        const handles = {
          nw: { x: bboxX, y: bboxY },
          ne: { x: bboxX + bboxWidth, y: bboxY },
          sw: { x: bboxX, y: bboxY + bboxHeight },
          se: { x: bboxX + bboxWidth, y: bboxY + bboxHeight },
        };

        for (const [handleName, handlePos] of Object.entries(handles)) {
          if (
            Math.abs(canvasX - handlePos.x) < handleSize &&
            Math.abs(canvasY - handlePos.y) < handleSize
          ) {
            return { bbox, handle: handleName as ResizeHandle };
          }
        }

        // BBOX 내부 체크
        if (
          canvasX >= bboxX &&
          canvasX <= bboxX + bboxWidth &&
          canvasY >= bboxY &&
          canvasY <= bboxY + bboxHeight
        ) {
          return { bbox, handle: null };
        }
      }

      return null;
    },
    [bboxes]
  );

  const drawAllBboxes = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    bboxes.forEach((bbox) => {
      const x = bbox.x * canvas.width;
      const y = bbox.y * canvas.height;
      const width = bbox.width * canvas.width;
      const height = bbox.height * canvas.height;

      // BBOX 그리기
      ctx.strokeStyle = bbox.color;
      ctx.lineWidth = selectedBboxId === bbox.id ? 3 : 2;
      ctx.setLineDash(selectedBboxId === bbox.id ? [] : [5, 5]);
      ctx.strokeRect(x, y, width, height);

      // 채우기
      ctx.fillStyle = `${bbox.color}20`;
      ctx.fillRect(x, y, width, height);

      // 선택된 BBOX의 경우 리사이즈 핸들 표시
      if (selectedBboxId === bbox.id) {
        const handleSize = 8;
        ctx.fillStyle = bbox.color;
        ctx.fillRect(x - handleSize / 2, y - handleSize / 2, handleSize, handleSize); // nw
        ctx.fillRect(x + width - handleSize / 2, y - handleSize / 2, handleSize, handleSize); // ne
        ctx.fillRect(x - handleSize / 2, y + height - handleSize / 2, handleSize, handleSize); // sw
        ctx.fillRect(x + width - handleSize / 2, y + height - handleSize / 2, handleSize, handleSize); // se
      }
    });

    // 그리기 중인 BBOX 표시
    if (isDrawing && startPos && selectedObjectId && selectedObjectColor && currentMousePos) {
      const startX = startPos.x * canvas.width;
      const startY = startPos.y * canvas.height;
      const endX = currentMousePos.x * canvas.width;
      const endY = currentMousePos.y * canvas.height;

      const x = Math.min(startX, endX);
      const y = Math.min(startY, endY);
      const width = Math.abs(endX - startX);
      const height = Math.abs(endY - startY);

      ctx.strokeStyle = selectedObjectColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x, y, width, height);
      ctx.fillStyle = `${selectedObjectColor}20`;
      ctx.fillRect(x, y, width, height);
    }
  }, [bboxes, selectedBboxId, isDrawing, startPos, selectedObjectId, selectedObjectColor, currentMousePos]);

  useEffect(() => {
    drawAllBboxes();
  }, [drawAllBboxes]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!selectedObjectId || !selectedObjectColor) {
        // 객체가 선택되지 않았으면 기존 BBOX 선택/이동만 가능
        const pos = getRelativePosition(e.clientX, e.clientY);
        if (!pos) return;

        const result = getBboxAtPosition(pos.x, pos.y);
        if (result) {
          setSelectedBboxId(result.bbox.id);
          if (result.handle) {
            setInteractionMode("resize");
            setResizeHandle(result.handle);
            setStartPos({ x: result.bbox.x, y: result.bbox.y });
          } else {
            setInteractionMode("move");
            const canvas = canvasRef.current;
            if (canvas) {
              const canvasX = pos.x * canvas.width;
              const canvasY = pos.y * canvas.height;
              const bboxX = result.bbox.x * canvas.width;
              const bboxY = result.bbox.y * canvas.height;
              setDragOffset({
                x: canvasX - bboxX,
                y: canvasY - bboxY,
              });
            }
          }
        } else {
          setSelectedBboxId(null);
        }
        return;
      }

      // 객체가 선택된 경우 새 BBOX 그리기
      const pos = getRelativePosition(e.clientX, e.clientY);
      if (!pos) return;

      // 기존 BBOX와 겹치는지 확인
      const result = getBboxAtPosition(pos.x, pos.y);
      if (result) {
        // 기존 BBOX 선택/이동
        setSelectedBboxId(result.bbox.id);
        if (result.handle) {
          setInteractionMode("resize");
          setResizeHandle(result.handle);
          setStartPos({ x: result.bbox.x, y: result.bbox.y });
        } else {
          setInteractionMode("move");
          const canvas = canvasRef.current;
          if (canvas) {
            const canvasX = pos.x * canvas.width;
            const canvasY = pos.y * canvas.height;
            const bboxX = result.bbox.x * canvas.width;
            const bboxY = result.bbox.y * canvas.height;
            setDragOffset({
              x: canvasX - bboxX,
              y: canvasY - bboxY,
            });
          }
        }
      } else {
        // 새 BBOX 그리기 시작
        setIsDrawing(true);
        setStartPos(pos);
        setInteractionMode("draw");
        setSelectedBboxId(null);
      }
    },
    [selectedObjectId, selectedObjectColor, getRelativePosition, getBboxAtPosition]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const pos = getRelativePosition(e.clientX, e.clientY);
      if (!pos) return;

      if (interactionMode === "draw" && isDrawing && startPos && selectedObjectId && selectedObjectColor) {
        // 그리기 중인 BBOX 미리보기 - 마우스 위치 업데이트
        setCurrentMousePos(pos);
      } else if (interactionMode === "move" && selectedBboxId && dragOffset) {
        // BBOX 이동
        const canvas = canvasRef.current;
        if (!canvas) return;

        const bbox = bboxes.find((b) => b.id === selectedBboxId);
        if (!bbox) return;

        const canvasX = pos.x * canvas.width;
        const canvasY = pos.y * canvas.height;

        const newX = Math.max(0, Math.min(1, (canvasX - dragOffset.x) / canvas.width));
        const newY = Math.max(0, Math.min(1, (canvasY - dragOffset.y) / canvas.height));

        onUpdateBbox(selectedBboxId, { x: newX, y: newY });
      } else if (interactionMode === "resize" && selectedBboxId && resizeHandle && startPos) {
        // BBOX 리사이즈
        const canvas = canvasRef.current;
        if (!canvas) return;

        const bbox = bboxes.find((b) => b.id === selectedBboxId);
        if (!bbox) return;

        let newX = bbox.x;
        let newY = bbox.y;
        let newWidth = bbox.width;
        let newHeight = bbox.height;

        if (resizeHandle === "nw") {
          newX = Math.min(pos.x, bbox.x + bbox.width);
          newY = Math.min(pos.y, bbox.y + bbox.height);
          newWidth = bbox.x + bbox.width - newX;
          newHeight = bbox.y + bbox.height - newY;
        } else if (resizeHandle === "ne") {
          newY = Math.min(pos.y, bbox.y + bbox.height);
          newWidth = Math.max(0.01, pos.x - bbox.x);
          newHeight = bbox.y + bbox.height - newY;
        } else if (resizeHandle === "sw") {
          newX = Math.min(pos.x, bbox.x + bbox.width);
          newWidth = bbox.x + bbox.width - newX;
          newHeight = Math.max(0.01, pos.y - bbox.y);
        } else if (resizeHandle === "se") {
          newWidth = Math.max(0.01, pos.x - bbox.x);
          newHeight = Math.max(0.01, pos.y - bbox.y);
        }

        if (newWidth > 0.01 && newHeight > 0.01) {
          onUpdateBbox(selectedBboxId, {
            x: Math.max(0, Math.min(1, newX)),
            y: Math.max(0, Math.min(1, newY)),
            width: Math.max(0.01, Math.min(1, newWidth)),
            height: Math.max(0.01, Math.min(1, newHeight)),
          });
        }
      } else {
        // 호버 효과를 위한 커서 변경
        setCurrentMousePos(null);
        const result = getBboxAtPosition(pos.x, pos.y);
        if (result && result.handle) {
          const handleCursors: Record<string, string> = {
            nw: "nw-resize",
            ne: "ne-resize",
            sw: "sw-resize",
            se: "se-resize",
          };
          if (canvasRef.current?.parentElement) {
            canvasRef.current.parentElement.style.cursor = handleCursors[result.handle] || "move";
          }
        } else if (result) {
          if (canvasRef.current?.parentElement) {
            canvasRef.current.parentElement.style.cursor = "move";
          }
        } else {
          if (canvasRef.current?.parentElement) {
            canvasRef.current.parentElement.style.cursor = selectedObjectId ? "crosshair" : "default";
          }
        }
      }
    },
    [
      interactionMode,
      isDrawing,
      startPos,
      selectedObjectId,
      selectedObjectColor,
      selectedBboxId,
      dragOffset,
      resizeHandle,
      bboxes,
      onUpdateBbox,
      getRelativePosition,
      getBboxAtPosition,
      drawAllBboxes,
    ]
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      if (interactionMode === "draw" && isDrawing && startPos && selectedObjectId && selectedObjectColor) {
        const pos = getRelativePosition(e.clientX, e.clientY);
        if (!pos) {
          setIsDrawing(false);
          setStartPos(null);
          setInteractionMode("none");
          return;
        }

        const width = Math.abs(pos.x - startPos.x);
        const height = Math.abs(pos.y - startPos.y);

        if (width > 0.01 && height > 0.01) {
          onAddBbox({
            objectId: selectedObjectId,
            x: Math.min(startPos.x, pos.x),
            y: Math.min(startPos.y, pos.y),
            width,
            height,
            color: selectedObjectColor,
          });
        }

        setIsDrawing(false);
        setStartPos(null);
        setCurrentMousePos(null);
        setInteractionMode("none");
      } else if (interactionMode === "move" || interactionMode === "resize") {
        setInteractionMode("none");
        setResizeHandle(null);
        setDragOffset(null);
        setCurrentMousePos(null);
      }
    },
    [interactionMode, isDrawing, startPos, selectedObjectId, selectedObjectColor, onAddBbox, getRelativePosition]
  );

  // 키보드 이벤트: Backspace 또는 Delete로 선택된 BBOX 삭제
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // 입력 필드에 포커스가 있으면 삭제하지 않음
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }

      // Backspace 또는 Delete 키
      if ((e.key === "Backspace" || e.key === "Delete") && selectedBboxId) {
        e.preventDefault();
        onRemoveBbox(selectedBboxId);
        setSelectedBboxId(null);
      }

      // Escape 키로 선택 해제
      if (e.key === "Escape" && selectedBboxId) {
        setSelectedBboxId(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [selectedBboxId, onRemoveBbox]);

  return (
    <CanvasContainer
      ref={containerRef}
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

export default CompositionCanvas;

