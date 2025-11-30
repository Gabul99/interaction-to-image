import React, { useRef, useEffect, useState, useCallback } from "react";
import styled from "styled-components";
import {
  type BoundingBox,
  type ToolMode,
  type InteractionData,
  type SketchLayer,
  type ObjectChip,
} from "../types";

const CanvasContainer = styled.div<{
  left: number;
  top: number;
  width: number;
  height: number;
  editable?: boolean;
  toolMode: ToolMode;
}>`
  position: absolute;
  left: ${(props) => props.left}px;
  top: ${(props) => props.top}px;
  width: ${(props) => props.width}px;
  height: ${(props) => props.height}px;
  pointer-events: ${(props) => (props.editable === false ? "none" : "auto")};
  cursor: ${(props) => {
    switch (props.toolMode) {
      case "select":
        return "default";
      case "bbox":
        return "crosshair";
      case "sketch":
        return "crosshair";
      case "eraser":
        return "grab";
      case "point":
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

// BBOX 관련 타입
type ResizeHandle = "nw" | "ne" | "sw" | "se" | null;
type InteractionMode =
  | "draw"
  | "move"
  | "resize"
  | "sketch"
  | "eraser"
  | "none";

interface UnifiedCanvasProps {
  // BBOX 관련
  bboxes?: BoundingBox[];
  selectedBboxId?: string | null;
  onBboxClick?: (
    bboxId: string,
    bbox?: { x: number; y: number; width: number; height: number }
  ) => void;
  onAddBbox?: (bbox: Omit<BoundingBox, "id">) => void;
  onUpdateBbox?: (bboxId: string, updates: Partial<BoundingBox>) => void;
  onRemoveBbox?: (bboxId: string) => void;

  // 객체 관련
  objects?: ObjectChip[];
  selectedObjectId?: string | null;
  selectedObjectColor?: string | null;

  // 도구 모드
  toolMode: ToolMode;
  editable?: boolean;
  disabled?: boolean;

  // 상호작용 콜백
  onInteraction?: (data: InteractionData) => void;
  onClearSelection?: () => void;

  // 스케치 관련
  sketchLayers?: SketchLayer[];
  onSketchUpdate?: (layers: SketchLayer[]) => void;

  // 이미지 참조
  imageRef?: React.RefObject<HTMLImageElement | null>;
  placeholderRef?: React.RefObject<HTMLDivElement | null>;
}

// Props를 ref에 저장하는 타입
interface LatestProps {
  bboxes: BoundingBox[];
  selectedBboxId: string | null;
  sketchLayers: SketchLayer[];
  toolMode: ToolMode;
  selectedObjectColor: string | null;
  editable: boolean;
  objects: ObjectChip[];
  selectedObjectId: string | null;
}

const UnifiedCanvas: React.FC<UnifiedCanvasProps> = ({
  bboxes = [],
  selectedBboxId: externalSelectedBboxId,
  onBboxClick,
  onAddBbox,
  onUpdateBbox,
  onRemoveBbox,
  objects = [],
  selectedObjectId,
  selectedObjectColor,
  toolMode,
  editable = true,
  disabled = false,
  onInteraction,
  onClearSelection,
  sketchLayers = [],
  onSketchUpdate,
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

  // 내부 상태 (인터랙션 관련만)
  const [internalSelectedBboxId, setInternalSelectedBboxId] = useState<
    string | null
  >(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState<{ x: number; y: number } | null>(
    null
  );
  const [interactionMode, setInteractionMode] =
    useState<InteractionMode>("none");
  const [resizeHandle, setResizeHandle] = useState<ResizeHandle>(null);
  const [dragOffset, setDragOffset] = useState<{ x: number; y: number } | null>(
    null
  );
  const [currentMousePos, setCurrentMousePos] = useState<{
    x: number;
    y: number;
  } | null>(null);

  // 스케치 관련 상태 (현재 그리는 경로만)
  const [currentSketchPath, setCurrentSketchPath] = useState<
    { x: number; y: number }[]
  >([]);

  // 선택된 BBOX ID (외부 prop 우선)
  const selectedBboxId =
    externalSelectedBboxId !== undefined
      ? externalSelectedBboxId
      : internalSelectedBboxId;

  // Props를 ref에 저장 (의존성 제거를 위해)
  const latestPropsRef = useRef<LatestProps>({
    bboxes,
    selectedBboxId,
    sketchLayers,
    toolMode,
    selectedObjectColor: selectedObjectColor || null,
    editable,
    objects,
    selectedObjectId: selectedObjectId || null,
  });

  // 이전 sketchLayers prop을 추적하여 실제 변경 여부 확인
  const prevSketchLayersPropRef = useRef<SketchLayer[]>(sketchLayers);

  // Props 변경 시 ref 업데이트 및 Canvas 재그리기
  useEffect(() => {
    // sketchLayers의 경우, prop이 실제로 변경되었는지 확인
    // prop이 빈 배열이고 ref에 값이 있으면, 낙관적 업데이트로 인한 것일 수 있으므로 덮어쓰지 않음
    const sketchLayersToUse =
      sketchLayers.length === 0 &&
      latestPropsRef.current.sketchLayers.length > 0 &&
      prevSketchLayersPropRef.current.length === 0
        ? latestPropsRef.current.sketchLayers // 낙관적 업데이트로 ref에 저장된 값 유지
        : sketchLayers; // prop이 실제로 변경되었거나, 명시적으로 빈 배열로 설정된 경우

    // 이전 prop 값 업데이트
    prevSketchLayersPropRef.current = sketchLayers;

    latestPropsRef.current = {
      bboxes,
      selectedBboxId,
      sketchLayers: sketchLayersToUse,
      toolMode,
      selectedObjectColor: selectedObjectColor || null,
      editable,
      objects,
      selectedObjectId: selectedObjectId || null,
    };

    // requestAnimationFrame을 사용하여 다음 프레임에 그리기
    const rafId = requestAnimationFrame(() => {
      drawCanvasRef.current();
    });

    return () => {
      cancelAnimationFrame(rafId);
    };
  }, [
    bboxes,
    selectedBboxId,
    sketchLayers,
    toolMode,
    selectedObjectColor,
    editable,
    objects,
    selectedObjectId,
  ]);

  // 이미지 또는 placeholder 위치와 크기에 맞춰 Canvas 위치 업데이트
  useEffect(() => {
    const updateCanvasPosition = () => {
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
        canvasRef.current.width = targetRect.width;
        canvasRef.current.height = targetRect.height;
        // 크기 변경 시 즉시 재그리기
        requestAnimationFrame(() => {
          drawCanvasRef.current();
        });
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

  // Canvas 그리기 함수를 ref에 저장 (의존성 문제 방지)
  const drawCanvasRef = useRef<() => void>(() => {});

  // Canvas 그리기 함수 (의존성 없이 ref만 참조)
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvas.width === 0 || canvas.height === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const props = latestPropsRef.current;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 스케치 레이어 그리기 (ref에서 직접 참조하여 최신 값 보장)
    const sketchLayersToDraw = latestPropsRef.current.sketchLayers;
    console.log("[drawCanvas] Sketch layers to draw:", sketchLayersToDraw);
    console.log("[drawCanvas] Sketch layers count:", sketchLayersToDraw.length);

    sketchLayersToDraw.forEach((layer, layerIndex) => {
      console.log(`[drawCanvas] Drawing layer ${layerIndex}:`, layer);
      ctx.strokeStyle = layer.color;
      ctx.lineWidth = 10; // 색연필 느낌의 두꺼운 선
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      layer.paths.forEach((path, pathIndex) => {
        if (path.length < 2) {
          console.log(
            `[drawCanvas] Layer ${layerIndex}, Path ${pathIndex}: Skipping (length < 2)`
          );
          return;
        }

        console.log(
          `[drawCanvas] Layer ${layerIndex}, Path ${pathIndex}: Drawing ${path.length} points`
        );
        ctx.beginPath();
        ctx.moveTo(path[0].x * canvas.width, path[0].y * canvas.height);

        for (let i = 1; i < path.length; i++) {
          ctx.lineTo(path[i].x * canvas.width, path[i].y * canvas.height);
        }

        ctx.stroke();
      });
    });

    // 현재 그리는 스케치 경로
    if (props.toolMode === "sketch" && currentSketchPath.length > 1) {
      // 객체가 선택되지 않았으면 기본 색상 사용
      const colorToUse = props.selectedObjectColor || "#6366f1";
      ctx.strokeStyle = colorToUse;
      ctx.lineWidth = 10;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      ctx.beginPath();
      ctx.moveTo(
        currentSketchPath[0].x * canvas.width,
        currentSketchPath[0].y * canvas.height
      );

      for (let i = 1; i < currentSketchPath.length; i++) {
        ctx.lineTo(
          currentSketchPath[i].x * canvas.width,
          currentSketchPath[i].y * canvas.height
        );
      }

      ctx.stroke();
    }

    // BBOX 그리기
    props.bboxes.forEach((bbox) => {
      const x = bbox.x * canvas.width;
      const y = bbox.y * canvas.height;
      const width = bbox.width * canvas.width;
      const height = bbox.height * canvas.height;

      const isSelected = props.selectedBboxId === bbox.id;

      // BBOX 그리기
      ctx.strokeStyle = bbox.color;
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.setLineDash(isSelected ? [] : [5, 5]);
      ctx.strokeRect(x, y, width, height);

      // 채우기
      ctx.fillStyle = `${bbox.color}20`;
      ctx.fillRect(x, y, width, height);

      // 선택된 BBOX의 경우 리사이즈 핸들 표시
      if (isSelected && props.editable && props.toolMode === "select") {
        const handleSize = 8;
        ctx.fillStyle = bbox.color;
        ctx.fillRect(
          x - handleSize / 2,
          y - handleSize / 2,
          handleSize,
          handleSize
        ); // nw
        ctx.fillRect(
          x + width - handleSize / 2,
          y - handleSize / 2,
          handleSize,
          handleSize
        ); // ne
        ctx.fillRect(
          x - handleSize / 2,
          y + height - handleSize / 2,
          handleSize,
          handleSize
        ); // sw
        ctx.fillRect(
          x + width - handleSize / 2,
          y + height - handleSize / 2,
          handleSize,
          handleSize
        ); // se
      }

      // 객체 라벨 표시
      const object = props.objects.find((obj) => obj.id === bbox.objectId);
      if (object) {
        const labelText = object.label;
        const padding = 6;
        const fontSize = 14;
        const fontFamily = "system-ui, -apple-system, sans-serif";

        ctx.font = `bold ${fontSize}px ${fontFamily}`;
        const textMetrics = ctx.measureText(labelText);
        const textWidth = textMetrics.width;
        const textHeight = fontSize;

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
        ctx.quadraticCurveTo(
          labelX + labelWidth,
          labelY,
          labelX + labelWidth,
          labelY + radius
        );
        ctx.lineTo(labelX + labelWidth, labelY + labelHeight - radius);
        ctx.quadraticCurveTo(
          labelX + labelWidth,
          labelY + labelHeight,
          labelX + labelWidth - radius,
          labelY + labelHeight
        );
        ctx.lineTo(labelX + radius, labelY + labelHeight);
        ctx.quadraticCurveTo(
          labelX,
          labelY + labelHeight,
          labelX,
          labelY + labelHeight - radius
        );
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

    // 그리기 중인 BBOX 미리보기 (객체 선택이 필수가 아님)
    if (props.toolMode === "bbox" && isDrawing && startPos && currentMousePos) {
      const startX = startPos.x * canvas.width;
      const startY = startPos.y * canvas.height;
      const endX = currentMousePos.x * canvas.width;
      const endY = currentMousePos.y * canvas.height;

      const x = Math.min(startX, endX);
      const y = Math.min(startY, endY);
      const width = Math.abs(endX - startX);
      const height = Math.abs(endY - startY);

      // 객체가 선택되지 않았으면 기본 색상 사용
      const colorToUse = props.selectedObjectColor || "#6366f1";
      ctx.strokeStyle = colorToUse;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x, y, width, height);
      ctx.fillStyle = `${colorToUse}20`;
      ctx.fillRect(x, y, width, height);
    }

    // 포인트 표시
    if (props.toolMode === "point" && startPos) {
      const canvasX = startPos.x * canvas.width;
      const canvasY = startPos.y * canvas.height;

      ctx.fillStyle = "#6366f1";
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 6, 0, 2 * Math.PI);
      ctx.fill();

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [currentSketchPath, isDrawing, startPos, currentMousePos]);

  // drawCanvas 함수를 ref에 저장
  useEffect(() => {
    drawCanvasRef.current = drawCanvas;
  }, [drawCanvas]);

  // 내부 상태 변경 시 재그리기
  useEffect(() => {
    const rafId = requestAnimationFrame(() => {
      drawCanvasRef.current();
    });
    return () => {
      cancelAnimationFrame(rafId);
    };
  }, [currentSketchPath, isDrawing, startPos, currentMousePos]);

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
    (
      x: number,
      y: number
    ): { bbox: BoundingBox; handle: ResizeHandle } | null => {
      if (!canvasRef.current) return null;

      const canvas = canvasRef.current;
      const canvasX = x * canvas.width;
      const canvasY = y * canvas.height;

      const props = latestPropsRef.current;

      // 역순으로 검사하여 위에 있는 BBOX를 우선 선택
      for (let i = props.bboxes.length - 1; i >= 0; i--) {
        const bbox = props.bboxes[i];
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
    []
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (disabled || !editable) {
        if (onClearSelection) {
          onClearSelection();
        }
        return;
      }

      const pos = getRelativePosition(e.clientX, e.clientY);
      if (!pos) {
        if (onClearSelection) {
          onClearSelection();
        }
        return;
      }

      const props = latestPropsRef.current;

      // 스케치 모드 (객체 선택이 필수가 아님)
      if (props.toolMode === "sketch") {
        setIsDrawing(true);
        setStartPos(pos);
        setInteractionMode("sketch");
        setCurrentSketchPath([pos]);
        return;
      }

      // 지우개 모드
      if (props.toolMode === "eraser") {
        setIsDrawing(true);
        setStartPos(pos);
        setInteractionMode("eraser");
        return;
      }

      // 포인트 모드
      if (props.toolMode === "point") {
        setStartPos(pos);
        if (onInteraction) {
          onInteraction({
            type: "point",
            x: pos.x,
            y: pos.y,
          });
        }
        return;
      }

      // BBOX 모드 또는 선택 모드
      if (props.toolMode === "bbox" || props.toolMode === "select") {
        const result = getBboxAtPosition(pos.x, pos.y);

        if (props.toolMode === "select") {
          if (result) {
            // 기존 BBOX 선택/이동/리사이즈
            setInternalSelectedBboxId(result.bbox.id);
            if (onBboxClick) {
              onBboxClick(result.bbox.id, {
                x: result.bbox.x,
                y: result.bbox.y,
                width: result.bbox.width,
                height: result.bbox.height,
              });
            }

            if (result.handle) {
              // 리사이즈 핸들 클릭 - 리사이즈 모드 시작
              setIsDrawing(true);
              setInteractionMode("resize");
              setResizeHandle(result.handle);
              setStartPos({ x: result.bbox.x, y: result.bbox.y });
            } else {
              // BBOX 내부 클릭 - 이동 모드 시작
              setIsDrawing(true);
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
            // BBOX 외부 클릭 - 선택 해제
            setInternalSelectedBboxId(null);
            if (onClearSelection) {
              onClearSelection();
            }
          }
        } else if (props.toolMode === "bbox") {
          // BBOX 모드: 새 BBOX 그리기 또는 기존 BBOX 선택 (객체 선택이 필수가 아님)
          if (result) {
            // 기존 BBOX 선택
            setInternalSelectedBboxId(result.bbox.id);
            if (onBboxClick) {
              onBboxClick(result.bbox.id, {
                x: result.bbox.x,
                y: result.bbox.y,
                width: result.bbox.width,
                height: result.bbox.height,
              });
            }
          } else {
            // 새 BBOX 그리기 시작
            setIsDrawing(true);
            setStartPos(pos);
            setInteractionMode("draw");
            setInternalSelectedBboxId(null);
          }
        } else {
          // 다른 경우
          setInternalSelectedBboxId(null);
          if (onClearSelection) {
            onClearSelection();
          }
        }
      }
    },
    [
      disabled,
      editable,
      getRelativePosition,
      getBboxAtPosition,
      onBboxClick,
      onClearSelection,
      onInteraction,
    ]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const pos = getRelativePosition(e.clientX, e.clientY);
      if (!pos) return;

      const props = latestPropsRef.current;

      // 스케치 그리기 (객체 선택이 필수가 아님)
      if (interactionMode === "sketch" && isDrawing) {
        setCurrentSketchPath((prev) => [...prev, pos]);
        // 실시간 그리기를 위해 즉시 재그리기
        requestAnimationFrame(() => {
          drawCanvasRef.current();
        });
        return;
      }

      // 지우개
      if (interactionMode === "eraser" && isDrawing) {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const eraserSize = 20;
        const canvasX = pos.x * canvas.width;
        const canvasY = pos.y * canvas.height;

        // 모든 스케치 레이어 확인 - 지우개가 닿은 경로(path)만 삭제
        const currentLayers = latestPropsRef.current.sketchLayers;
        const updatedLayers = currentLayers
          .map((layer) => {
            // 레이어의 각 경로를 확인하여 지우개가 닿은 경로만 필터링
            const remainingPaths = layer.paths.filter((path) => {
              // 경로의 모든 포인트를 확인
              for (const point of path) {
                const pointX = point.x * canvas.width;
                const pointY = point.y * canvas.height;
                const distance = Math.sqrt(
                  Math.pow(pointX - canvasX, 2) + Math.pow(pointY - canvasY, 2)
                );
                // 지우개 영역 안에 포인트가 하나라도 있으면 경로 삭제
                if (distance <= eraserSize) {
                  return false; // 경로 삭제
                }
              }
              return true; // 경로 유지
            });

            // 경로가 남아있으면 레이어 유지, 없으면 레이어 삭제
            return remainingPaths.length > 0
              ? { ...layer, paths: remainingPaths }
              : null;
          })
          .filter((layer): layer is SketchLayer => layer !== null);

        // 낙관적 업데이트: ref에 즉시 반영
        latestPropsRef.current.sketchLayers = updatedLayers;
        requestAnimationFrame(() => {
          drawCanvasRef.current();
        });

        // 부모에 알림
        if (onSketchUpdate) {
          onSketchUpdate(updatedLayers);
        }
        return;
      }

      // BBOX 그리기 미리보기 (객체 선택이 필수가 아님)
      if (interactionMode === "draw" && isDrawing && startPos) {
        setCurrentMousePos(pos);
        requestAnimationFrame(() => {
          drawCanvasRef.current();
        });
      } else if (
        interactionMode === "move" &&
        isDrawing &&
        selectedBboxId &&
        dragOffset &&
        onUpdateBbox
      ) {
        // BBOX 이동
        const canvas = canvasRef.current;
        if (!canvas) return;

        const props = latestPropsRef.current;
        const bbox = props.bboxes.find((b) => b.id === selectedBboxId);
        if (!bbox) return;

        const canvasX = pos.x * canvas.width;
        const canvasY = pos.y * canvas.height;

        const newX = Math.max(
          0,
          Math.min(1, (canvasX - dragOffset.x) / canvas.width)
        );
        const newY = Math.max(
          0,
          Math.min(1, (canvasY - dragOffset.y) / canvas.height)
        );

        onUpdateBbox(selectedBboxId, { x: newX, y: newY });
      } else if (
        interactionMode === "resize" &&
        isDrawing &&
        selectedBboxId &&
        resizeHandle &&
        startPos &&
        onUpdateBbox
      ) {
        // BBOX 리사이즈
        const canvas = canvasRef.current;
        if (!canvas) return;

        const props = latestPropsRef.current;
        const bbox = props.bboxes.find((b) => b.id === selectedBboxId);
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
        if (currentMousePos !== null) {
          setCurrentMousePos(null);
        }
        if (props.toolMode === "select") {
          const result = getBboxAtPosition(pos.x, pos.y);
          if (result && result.handle) {
            const handleCursors: Record<string, string> = {
              nw: "nw-resize",
              ne: "ne-resize",
              sw: "sw-resize",
              se: "se-resize",
            };
            if (canvasRef.current?.parentElement) {
              canvasRef.current.parentElement.style.cursor =
                handleCursors[result.handle] || "move";
            }
          } else if (result) {
            if (canvasRef.current?.parentElement) {
              canvasRef.current.parentElement.style.cursor = "move";
            }
          } else {
            if (canvasRef.current?.parentElement) {
              canvasRef.current.parentElement.style.cursor = "default";
            }
          }
        }
      }
    },
    [
      interactionMode,
      isDrawing,
      startPos,
      selectedBboxId,
      dragOffset,
      resizeHandle,
      onUpdateBbox,
      getRelativePosition,
      getBboxAtPosition,
      onSketchUpdate,
      currentMousePos,
    ]
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      const props = latestPropsRef.current;

      // 스케치 완료 (객체 선택이 필수가 아님)
      if (
        interactionMode === "sketch" &&
        isDrawing &&
        currentSketchPath.length > 1
      ) {
        const currentLayers = latestPropsRef.current.sketchLayers;
        // 객체가 선택되지 않았으면 기본 색상과 ID 사용
        const objectIdToUse = props.selectedObjectId || "new_area";
        const colorToUse = props.selectedObjectColor || "#6366f1";

        const existingLayer = currentLayers.find(
          (layer) => layer.objectId === objectIdToUse
        );

        let updatedLayers: SketchLayer[];
        if (existingLayer) {
          // 기존 레이어에 경로 추가
          updatedLayers = currentLayers.map((layer) =>
            layer.objectId === objectIdToUse
              ? { ...layer, paths: [...layer.paths, currentSketchPath] }
              : layer
          );
        } else {
          // 새 레이어 생성
          updatedLayers = [
            ...currentLayers,
            {
              objectId: objectIdToUse,
              color: colorToUse,
              paths: [currentSketchPath],
            },
          ];
        }

        // 낙관적 업데이트: ref에 즉시 반영하여 깜빡임 방지
        latestPropsRef.current.sketchLayers = updatedLayers;
        requestAnimationFrame(() => {
          drawCanvasRef.current();
        });

        // 스케치 정보 콘솔 출력
        console.log("=== Sketch MouseUp ===");
        console.log("Current Sketch Path:", currentSketchPath);
        console.log("Path Length:", currentSketchPath.length);
        console.log("Selected Object ID:", props.selectedObjectId);
        console.log("Selected Object Color:", props.selectedObjectColor);
        console.log("Existing Layer Found:", !!existingLayer);
        console.log("Updated Layers:", updatedLayers);
        console.log("Updated Layers Count:", updatedLayers.length);
        console.log(
          "Latest Props Sketch Layers:",
          latestPropsRef.current.sketchLayers
        );
        console.log("=====================");

        // 부모 컴포넌트에 알림
        if (onSketchUpdate) {
          onSketchUpdate(updatedLayers);
        }
        if (onInteraction) {
          onInteraction({
            type: "sketch",
            x: 0,
            y: 0,
            sketchData: updatedLayers,
          });
        }

        // 그리기 상태 초기화
        setCurrentSketchPath([]);
        setIsDrawing(false);
        setInteractionMode("none");
      } else if (
        interactionMode === "draw" &&
        isDrawing &&
        startPos &&
        onAddBbox
      ) {
        // BBOX 그리기 완료 (객체 선택이 필수가 아님)
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
          // 객체가 선택되지 않았으면 기본 색상과 ID 사용
          const objectIdToUse = props.selectedObjectId || "new_area";
          const colorToUse = props.selectedObjectColor || "#6366f1";

          onAddBbox({
            objectId: objectIdToUse,
            x: Math.min(startPos.x, pos.x),
            y: Math.min(startPos.y, pos.y),
            width,
            height,
            color: colorToUse,
          });
        }

        setIsDrawing(false);
        setStartPos(null);
        setCurrentMousePos(null);
        setInteractionMode("none");
      } else if (
        interactionMode === "move" ||
        interactionMode === "resize" ||
        interactionMode === "eraser"
      ) {
        // 이동/리사이즈/지우개 완료
        setInteractionMode("none");
        setResizeHandle(null);
        setDragOffset(null);
        setCurrentMousePos(null);
        setIsDrawing(false);
      }
    },
    [
      interactionMode,
      isDrawing,
      startPos,
      onAddBbox,
      getRelativePosition,
      currentSketchPath,
      onSketchUpdate,
      onInteraction,
    ]
  );

  // 키보드 이벤트: Backspace 또는 Delete로 선택된 BBOX 삭제
  useEffect(() => {
    if (!editable || toolMode !== "select") return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }

      if (
        (e.key === "Backspace" || e.key === "Delete") &&
        selectedBboxId &&
        onRemoveBbox
      ) {
        e.preventDefault();
        onRemoveBbox(selectedBboxId);
        setInternalSelectedBboxId(null);
      }

      if (e.key === "Escape" && selectedBboxId) {
        setInternalSelectedBboxId(null);
        if (onClearSelection) {
          onClearSelection();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [editable, toolMode, selectedBboxId, onRemoveBbox, onClearSelection]);

  return (
    <CanvasContainer
      ref={containerRef}
      left={canvasPosition.left}
      top={canvasPosition.top}
      width={canvasPosition.width}
      height={canvasPosition.height}
      editable={editable}
      toolMode={toolMode}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <Canvas ref={canvasRef} />
    </CanvasContainer>
  );
};

export default UnifiedCanvas;
