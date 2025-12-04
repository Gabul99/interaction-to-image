import React, { useState } from "react";
import styled from "styled-components";
import { useImageStore } from "../stores/imageStore";
import { USE_MOCK_MODE } from "../config/api";
import { type ToolMode, type SketchLayer } from "../types";
import ObjectChipList from "./ObjectChipList";
import UnifiedCanvas from "./UnifiedCanvas";
import FloatingToolbox from "./FloatingToolbox";
import ImageViewer from "./ImageViewer";
import { requestObjectList } from "../api/composition";
import { startSession, stepOnce, type LayoutItem } from "../lib/api";
import { API_BASE_URL } from "../config/api";
import { exportSketchToFile } from "../utils/sketchUtils";
import { createUniqueBranchId, extractBackendBranchId } from "../stores/imageStore";

const ModalOverlay = styled.div<{ visible: boolean }>`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
  z-index: 2000;
  display: ${(props) => (props.visible ? "flex" : "none")};
  align-items: center;
  justify-content: center;
  padding: 20px;
`;

const ModalContainer = styled.div`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  width: 100%;
  max-width: 50vw;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const ModalHeader = styled.div`
  padding: 24px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ModalTitle = styled.h2`
  color: #f9fafb;
  font-size: 20px;
  font-weight: 600;
  margin: 0;
`;

const CloseButton = styled.button`
  background: transparent;
  border: none;
  color: #9ca3af;
  font-size: 24px;
  cursor: pointer;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #f9fafb;
  }
`;

const ModalContent = styled.div`
  padding: 24px;
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
`;

const TwoColumnLayout = styled.div`
  display: flex;
  gap: 24px;
  flex: 1;
  min-height: 0;
`;

const LeftColumn = styled.div`
  flex: 0 0 300px;
  display: flex;
  flex-direction: column;
  gap: 24px;
`;

const RightColumn = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 24px;
  min-width: 0;
`;

const Section = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const SectionTitle = styled.h3`
  color: #d1d5db;
  font-size: 14px;
  font-weight: 500;
  margin: 0;
`;

const ImageContainer = styled.div`
  position: relative;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  width: 100%;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  overflow: hidden;

  /* 이미지에 맞게 높이 조정 */
  img {
    max-width: 100%;
    height: auto;
    display: block;
  }
`;

const ActionButtonGroup = styled.div`
  display: flex;
  gap: 12px;
  margin-top: auto;
  padding-top: 24px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

const ActionButton = styled.button<{ variant: "submit" | "cancel" }>`
  flex: 1;
  padding: 12px 20px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;

  ${(props) =>
    props.variant === "submit"
      ? `
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;

    &:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }

    &:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
  `
      : `
    background: rgba(55, 65, 81, 0.5);
    color: #f9fafb;
    border: 2px solid #374151;

    &:hover {
      background: rgba(55, 65, 81, 0.7);
      border-color: #6b7280;
    }
  `}
`;

const LoadingText = styled.div`
  color: #9ca3af;
  font-size: 14px;
  text-align: center;
  padding: 20px;
`;

const PromptInput = styled.input`
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #374151;
  border-radius: 8px;
  font-size: 14px;
  background: rgba(55, 65, 81, 0.5);
  color: #f9fafb;
  outline: none;
  transition: border-color 0.2s ease;

  &:focus {
    border-color: #6366f1;
  }

  &::placeholder {
    color: #9ca3af;
  }
`;

const SendPromptButton = styled.button`
  margin-top: 12px;
  width: 100%;
  padding: 12px;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const OptionGroup = styled.div`
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  flex-wrap: wrap;
`;

const OptionButton = styled.button<{ selected: boolean }>`
  flex: 1;
  min-width: 100px;
  padding: 10px 14px;
  border: 2px solid ${(props) => (props.selected ? "#6366f1" : "#374151")};
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  background: ${(props) =>
    props.selected ? "rgba(99, 102, 241, 0.2)" : "rgba(55, 65, 81, 0.5)"};
  color: ${(props) => (props.selected ? "#6366f1" : "#f9fafb")};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover:not(:disabled) {
    border-color: #6366f1;
    background: rgba(99, 102, 241, 0.1);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const InstructionText = styled.div`
  color: #9ca3af;
  font-size: 12px;
  margin-bottom: 12px;
  padding: 10px;
  background: rgba(99, 102, 241, 0.1);
  border-radius: 6px;
  border: 1px solid rgba(99, 102, 241, 0.2);
`;

interface CompositionModalProps {
  visible: boolean;
  onClose: () => void;
  onComplete?: () => void;
  initialPrompt?: string;
  placeholderNodeId?: string | null;
}

const CompositionModal: React.FC<CompositionModalProps> = ({
  visible,
  onClose,
  onComplete,
  initialPrompt = "",
  placeholderNodeId = null,
}) => {
  const {
    compositionState,
    setObjectList,
    addObject,
    removeObject,
    selectObject,
    addBbox,
    updateBbox,
    removeBbox,
    clearComposition,
    createGraphSession,
    currentGraphSession,
    addPromptNodeToGraph,
    registerParallelSession,
    updatePromptNodePrompt,
    addImageNode,
    addLoadingNode,
    removeLoadingNode,
    selectNode,
  } = useImageStore();

  // Step interval for preview (same as GraphCanvas default)
  const stepInterval = 5;

  const [currentPrompt, setCurrentPrompt] = useState<string>(initialPrompt);
  const [isLoadingObjects, setIsLoadingObjects] = useState(false);
  const [toolMode, setToolMode] = useState<ToolMode>("select");
  const [selectedBboxId, setSelectedBboxId] = useState<string | null>(null);
  const [sketchLayers, setSketchLayers] = useState<SketchLayer[]>([]);
  const [compositionMode, setCompositionMode] = useState<
    "bbox" | "sketch" | null
  >(null);
  const imageRef = React.useRef<HTMLImageElement>(null);
  const placeholderRef = React.useRef<HTMLDivElement>(null);

  // initialPrompt가 변경되면 currentPrompt 업데이트
  React.useEffect(() => {
    if (visible && initialPrompt) {
      setCurrentPrompt(initialPrompt);
    }
  }, [visible, initialPrompt]);

  // 모달이 열릴 때 프롬프트가 있으면 자동으로 객체 리스트 생성
  React.useEffect(() => {
    if (visible && initialPrompt && initialPrompt.trim().length > 0) {
      // 모달이 열리고 프롬프트가 있으면 자동으로 객체 리스트 생성
      handleSendPrompt(initialPrompt);
    }
  }, [visible]); // initialPrompt는 의존성에 포함하지 않음 (한 번만 실행되도록)

  // 모달이 닫힐 때 모든 temporary data 초기화
  React.useEffect(() => {
    if (!visible) {
      // initialPrompt가 있으면 유지, 없으면 초기화
      if (!initialPrompt) {
        setCurrentPrompt("");
      }
      setCompositionMode(null);
      setSketchLayers([]);
      setToolMode("select");
      setSelectedBboxId(null);
      setIsLoadingObjects(false);
      clearComposition();
    }
  }, [visible, clearComposition, initialPrompt]);

  const handleSendPrompt = async (newPrompt: string) => {
    console.log("=".repeat(80));
    console.log(
      "[CompositionModal] ========== 객체 리스트 생성 요청 =========="
    );
    console.log("[CompositionModal] 프롬프트:", newPrompt);
    console.log("=".repeat(80));

    setCurrentPrompt(newPrompt);
    setIsLoadingObjects(true);
    clearComposition();
    setCompositionMode(null);
    setSketchLayers([]);

    try {
      const objects = await requestObjectList(newPrompt);
      console.log("[CompositionModal] 객체 리스트 수신:", objects);
      setObjectList(objects);
      setIsLoadingObjects(false);
    } catch (error) {
      console.error("[CompositionModal] 객체 리스트 요청 실패:", error);
      setIsLoadingObjects(false);
    }
  };

  const handleComplete = async () => {
    console.log("=".repeat(80));
    console.log(
      "[CompositionModal] ========== handleComplete 호출됨 =========="
    );
    console.log("[CompositionModal] currentPrompt:", currentPrompt);

    if (!currentPrompt) {
      console.error("[CompositionModal] 프롬프트가 없습니다!");
      return;
    }

    try {
      // 선택된 모드에 따라 하나만 전송
      let bboxes:
        | Array<{
            objectId: string;
            x: number;
            y: number;
            width: number;
            height: number;
          }>
        | undefined = undefined;
      let sketchImageFile: File | null = null;

      if (compositionMode === "bbox" && compositionState.bboxes.length > 0) {
        bboxes = compositionState.bboxes.map((bbox) => ({
          objectId: bbox.objectId,
          x: bbox.x,
          y: bbox.y,
          width: bbox.width,
          height: bbox.height,
        }));
      } else if (compositionMode === "sketch" && sketchLayers.length > 0) {
        // 스케치 이미지 생성
        try {
          const canvasWidth = 512;
          const canvasHeight = 512;
          sketchImageFile = await exportSketchToFile(
            sketchLayers,
            canvasWidth,
            canvasHeight,
            `sketch_${Date.now()}.png`
          );
          console.log(
            "[CompositionModal] 스케치 이미지 생성 완료:",
            sketchImageFile.name
          );
        } catch (error) {
          console.error("[CompositionModal] 스케치 이미지 생성 실패:", error);
        }
      }

      // 백엔드로 전송할 데이터 로깅
      console.log("=".repeat(80));
      console.log(
        "[CompositionModal] ========== 백엔드로 전송할 데이터 =========="
      );
      console.log("[CompositionModal] 프롬프트:", currentPrompt);
      console.log("[CompositionModal] 객체 리스트:", compositionState.objects);
      console.log("[CompositionModal] 바운딩 박스:", bboxes);
      console.log(
        "[CompositionModal] 스케치 이미지:",
        sketchImageFile ? sketchImageFile.name : "없음"
      );
      console.log("[CompositionModal] API_BASE_URL:", API_BASE_URL);
      console.log("=".repeat(80));

      // REST API: 세션 시작
      const layoutItems: LayoutItem[] | undefined =
        compositionMode === "bbox" && (bboxes?.length || 0) > 0
          ? bboxes!.map((bb) => {
              const obj = compositionState.objects.find(
                (o) => o.id === bb.objectId
              );
              return {
                phrase: obj?.label || "object",
                x0: bb.x,
                y0: bb.y,
                x1: bb.x + bb.width,
                y1: bb.y + bb.height,
              };
            })
          : undefined;

      const useSketchEdge =
        compositionMode === "sketch" && !!sketchImageFile && sketchLayers.length > 0;

      const startResp = await startSession({
        prompt: currentPrompt,
        steps: 50,
        seed: 67,
        model_version: "512",
        gpu_id: 0,
        guidance_scale: 4.5,
        enable_layout:
          compositionMode === "bbox" && (layoutItems?.length || 0) > 0,
        layout_items: layoutItems,
        // 스케치 모드일 때는 edge guidance 활성화 및 스케치 이미지를 edge_files로 전송
        enable_edge: useSketchEdge,
        edge_phrases_text: useSketchEdge ? currentPrompt : undefined,
        edge_files: useSketchEdge && sketchImageFile ? [sketchImageFile] : undefined,
      });

      const sessionId = startResp.session_id;
      const activeBranchId = startResp.active_branch_id;

      let createdPromptNodeId: string | null = null;

      // Check if there's already a graph session (for parallel sessions)
      if (currentGraphSession && currentGraphSession.nodes.length > 0) {
        // Add new prompt node to existing graph session (parallel session)
        console.log("[CompositionModal] Adding parallel session to existing graph");
        
        // Use the placeholderNodeId from props if available (now it's actually a prompt node ID)
        let targetPromptNodeId = placeholderNodeId;
        if (!targetPromptNodeId) {
          const promptNode = currentGraphSession.nodes.find(
            (n) => n.type === "prompt"
          );
          targetPromptNodeId = promptNode?.id || undefined;
        }
        
        const promptNode = targetPromptNodeId
          ? currentGraphSession.nodes.find((n) => n.id === targetPromptNodeId)
          : null;
        
        if (targetPromptNodeId) {
          console.log("[CompositionModal] Found prompt node to update:", targetPromptNodeId);
        }
        
        createdPromptNodeId = addPromptNodeToGraph(
          currentPrompt,
          sessionId,
          activeBranchId,
          compositionMode === "bbox" && compositionState.bboxes.length > 0
            ? compositionState.bboxes
            : undefined,
          compositionMode === "sketch" && sketchLayers.length > 0
            ? sketchLayers
            : undefined,
          promptNode?.position, // Use prompt node position if available
          targetPromptNodeId // Pass prompt node ID to update it
        );
        
        if (createdPromptNodeId) {
          console.log("[CompositionModal] 병렬 세션 프롬프트 노드 생성:", createdPromptNodeId);
          // Register the parallel session
          registerParallelSession(createdPromptNodeId, sessionId, activeBranchId);
        }
      } else {
        // Create new graph session (first session)
        const graphSessionId = createGraphSession(
          currentPrompt,
          sessionId,
          undefined, // rootNodeId
          compositionMode === "bbox" && compositionState.bboxes.length > 0
            ? compositionState.bboxes
            : undefined,
          compositionMode === "sketch" && sketchLayers.length > 0
            ? sketchLayers
            : undefined
        );

        console.log("[CompositionModal] 그래프 세션 생성 완료:", {
          graphSessionId,
          sessionId,
        });
        
        // Register the first session as well
        const promptNode = useImageStore.getState().currentGraphSession?.nodes.find(
          (n) => n.type === "prompt"
        );
        if (promptNode) {
          createdPromptNodeId = promptNode.id;
          registerParallelSession(promptNode.id, sessionId, activeBranchId);
        }
      }

      // 세션 메타 설정
      useImageStore.getState().setBackendSessionMeta(sessionId, activeBranchId);
      console.log("[CompositionModal] 세션 메타 설정 완료:", sessionId);

      // 생성된 prompt node ID 사용
      const finalPromptNodeId = createdPromptNodeId;

      // Modal 닫기 (먼저 닫아서 UI가 반응하도록)
      if (onComplete) {
        onComplete();
      }
      onClose();

      // 첫 이미지를 자동으로 생성하기 위해 Next Step 실행
      // 상태 업데이트를 기다리기 위해 약간의 딜레이 후 실행
      if (finalPromptNodeId) {
        setTimeout(async () => {
          try {
            // 최신 상태 가져오기
            const updatedSession = useImageStore.getState().currentGraphSession;
            if (!updatedSession) return;

            // Prompt node 찾기
            const promptNode = updatedSession.nodes.find((n) => n.id === finalPromptNodeId);
            if (!promptNode) return;

            // Get unique branch ID for the prompt node
            const uniqueBranchId = createUniqueBranchId(sessionId, activeBranchId);
            const backendBranchId = extractBackendBranchId(uniqueBranchId);
            const isMainBranch = backendBranchId === "B0";

            // Calculate position for first image (stepInterval 단계에서)
            const firstPreviewStep = stepInterval;
            const graphSessionId = updatedSession.id;
            
            // 간단한 위치 계산 (GraphCanvas의 calculatePositionWithOffset 대신)
            const rowIndex = isMainBranch 
              ? (promptNode.data?.rowIndex ?? 0)
              : 0; // 간단하게 0으로 설정 (GraphCanvas에서 정확히 계산됨)
            const columnIndex = Math.ceil(firstPreviewStep / stepInterval);
            const pos = {
              x: 100 + columnIndex * 300, // GRID_START_X + columnIndex * GRID_CELL_WIDTH
              y: 50 + rowIndex * 400, // GRID_START_Y + rowIndex * GRID_CELL_HEIGHT
            };

            // Add loading node
            const loadingNodeId = addLoadingNode(graphSessionId, finalPromptNodeId, firstPreviewStep, pos, uniqueBranchId);

            // Run steps until we reach the first preview step
            let lastResp: Awaited<ReturnType<typeof stepOnce>> | null = null;
            const maxIterations = 50;
            let iterations = 0;

            while (iterations < maxIterations) {
              const resp = await stepOnce({
                session_id: sessionId,
                branch_id: backendBranchId,
              });
              lastResp = resp;
              iterations++;

              console.log(`[CompositionModal] Auto Step iteration ${iterations}: backend now at step ${resp.i}, target preview step=${firstPreviewStep}`);

              // Check if we've reached the end
              if (resp.i >= resp.num_steps) {
                console.log(`[CompositionModal] Reached end at step ${resp.i}/${resp.num_steps}`);
                if (loadingNodeId) {
                  removeLoadingNode(graphSessionId, loadingNodeId);
                }
                break;
              }

              // Check if we've reached the first preview step
              if (resp.i >= firstPreviewStep) {
                console.log(`[CompositionModal] Reached first preview step ${resp.i} (target was ${firstPreviewStep})`);
                break;
              }
            }

            // Add preview image
            if (lastResp?.preview_png_base64) {
              const gsAfterStep = useImageStore.getState().currentGraphSession;
              if (!gsAfterStep) return;

              const currentGsId = gsAfterStep.id;
              const promptNodeAfter = gsAfterStep.nodes.find((n) => 
                n.type === "prompt" && n.id === finalPromptNodeId
              );
              const promptNodeIdAfter = promptNodeAfter?.id || null;

              // Find the last node in this branch (should be none, so use prompt node)
              const branchImageNodes = gsAfterStep.nodes.filter((n) => {
                if (n.type !== "image") return false;
                const nodeUniqueBranchId = n.data?.uniqueBranchId as string | undefined;
                return nodeUniqueBranchId === uniqueBranchId;
              });
              const lastBranchNode = branchImageNodes
                .slice()
                .sort((a, b) => (a.data?.step || 0) - (b.data?.step || 0))
                .pop();
              const parentNodeId = lastBranchNode?.id || promptNodeIdAfter || finalPromptNodeId;

              if (parentNodeId) {
                const rowIndexAfter = isMainBranch 
                  ? (promptNodeAfter?.data?.rowIndex ?? 0)
                  : 0;
                const actualStep = lastResp.i;
                const columnIndexAfter = Math.ceil(actualStep / stepInterval);
                const posAfter = {
                  x: 100 + columnIndexAfter * 300,
                  y: 50 + rowIndexAfter * 400,
                };

                console.log(`[CompositionModal] Adding first image node: step=${actualStep}, uniqueBranchId=${uniqueBranchId}`);

                // If loading node step doesn't match the actual step from backend, remove loading node first
                if (loadingNodeId && actualStep !== firstPreviewStep) {
                  console.warn(`[CompositionModal] Step mismatch: loading node at ${firstPreviewStep}, backend returned ${actualStep}. Removing loading node.`);
                  removeLoadingNode(currentGsId, loadingNodeId);
                }

                addImageNode(currentGsId, parentNodeId, lastResp.preview_png_base64, actualStep, posAfter, undefined, uniqueBranchId);

                // Ensure loading node is removed if it still exists
                if (loadingNodeId) {
                  const stateAfterAdd = useImageStore.getState().currentGraphSession;
                  const loadingStillExists = stateAfterAdd?.nodes.some((n) => n.id === loadingNodeId && n.type === "loading");
                  if (loadingStillExists) {
                    console.warn(`[CompositionModal] Loading node ${loadingNodeId} still exists after addImageNode, removing it`);
                    removeLoadingNode(currentGsId, loadingNodeId);
                  }
                }

                // Select the newly created image node
                const newImageNode = useImageStore.getState().currentGraphSession?.nodes.find(
                  (n) => n.type === "image" && n.data?.step === actualStep && (n.data?.uniqueBranchId as string | undefined) === uniqueBranchId
                );
                if (newImageNode) {
                  selectNode(newImageNode.id);
                  console.log("[CompositionModal] 첫 이미지 노드 선택:", newImageNode.id);
                }
              }
            } else if (loadingNodeId) {
              // Remove loading node if no preview was generated
              const gs = useImageStore.getState().currentGraphSession;
              const currentGsId = gs?.id || graphSessionId;
              removeLoadingNode(currentGsId, loadingNodeId);
            }
          } catch (error) {
            console.error("[CompositionModal] 자동 첫 이미지 생성 실패:", error);
          }
        }, 100); // 100ms 딜레이로 상태 업데이트 대기
      }
    } catch (error) {
      console.error("=".repeat(80));
      console.error(
        "[CompositionModal] ========== 이미지 생성 시작 실패 =========="
      );
      console.error("[CompositionModal] 에러:", error);
      if (error instanceof Error) {
        console.error("[CompositionModal] 에러 메시지:", error.message);
        console.error("[CompositionModal] 에러 스택:", error.stack);
      }
      console.error("=".repeat(80));
      // 사용자에게 알림 (선택적)
      alert(
        `이미지 생성 시작 실패: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  };

  const handleCancel = () => {
    // 모든 temporary data 초기화
    clearComposition();
    setCurrentPrompt("");
    setCompositionMode(null);
    setSketchLayers([]);
    setToolMode("select");
    setSelectedBboxId(null);
    setIsLoadingObjects(false);
    onClose();
  };

  if (!visible) return null;

  return (
    <ModalOverlay visible={visible} onClick={handleCancel}>
      <ModalContainer onClick={(e) => e.stopPropagation()}>
        <ModalHeader>
          <ModalTitle>새 이미지 생성</ModalTitle>
          <CloseButton onClick={handleCancel}>×</CloseButton>
        </ModalHeader>
        <ModalContent>
          <Section>
            <SectionTitle>프롬프트</SectionTitle>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (currentPrompt.trim() && !isLoadingObjects) {
                  handleSendPrompt(currentPrompt);
                }
              }}
            >
              <PromptInput
                type="text"
                placeholder="이미지 생성을 위한 프롬프트를 입력하세요..."
                value={currentPrompt}
                onChange={(e) => {
                  const newPrompt = e.target.value;
                  setCurrentPrompt(newPrompt);
                  // Prompt node의 프롬프트도 실시간으로 업데이트
                  if (placeholderNodeId && currentGraphSession) {
                    updatePromptNodePrompt(currentGraphSession.id, placeholderNodeId, newPrompt);
                  }
                }}
                disabled={isLoadingObjects}
              />
              <SendPromptButton
                type="submit"
                disabled={isLoadingObjects || !currentPrompt.trim()}
              >
                객체 리스트 생성
              </SendPromptButton>
            </form>
          </Section>

          {isLoadingObjects && (
            <LoadingText>객체 리스트를 생성하는 중...</LoadingText>
          )}

          {compositionState.objects.length > 0 && !isLoadingObjects && (
            <TwoColumnLayout>
              <LeftColumn>
                <Section>
                  <SectionTitle>객체 리스트</SectionTitle>
                  <ObjectChipList
                    objects={compositionState.objects}
                    selectedObjectId={compositionState.selectedObjectId}
                    onSelectObject={selectObject}
                    onAddObject={addObject}
                    onRemoveObject={removeObject}
                  />
                </Section>
              </LeftColumn>
              <RightColumn>
                <Section>
                  <SectionTitle>구도 설정</SectionTitle>

                  {/* 구도 방식 선택 */}
                  {!compositionMode && (
                    <>
                      <SectionTitle>구도 방식 선택</SectionTitle>
                      <OptionGroup>
                        <OptionButton
                          selected={false}
                          onClick={() => {
                            setCompositionMode("bbox");
                            setSketchLayers([]);
                            setToolMode("select");
                          }}
                        >
                          박스 (BBOX)
                        </OptionButton>
                        <OptionButton
                          selected={false}
                          onClick={() => {
                            setCompositionMode("sketch");
                            // 모든 BBOX 제거
                            compositionState.bboxes.forEach((bbox) => {
                              removeBbox(bbox.id);
                            });
                            setSelectedBboxId(null);
                            setToolMode("select");
                          }}
                        >
                          스케치
                        </OptionButton>
                      </OptionGroup>
                      <InstructionText>
                        구도 설정 방식을 선택하세요. 박스 또는 스케치 중 하나만
                        선택할 수 있습니다.
                      </InstructionText>
                    </>
                  )}

                  {/* 선택된 모드에 따른 UI */}
                  {compositionMode && (
                    <>
                      <OptionGroup>
                        <OptionButton
                          selected={compositionMode === "bbox"}
                          onClick={() => {
                            setCompositionMode("bbox");
                            setSketchLayers([]);
                            setToolMode("select");
                          }}
                        >
                          박스 (BBOX)
                        </OptionButton>
                        <OptionButton
                          selected={compositionMode === "sketch"}
                          onClick={() => {
                            setCompositionMode("sketch");
                            // 모든 BBOX 제거
                            compositionState.bboxes.forEach((bbox) => {
                              removeBbox(bbox.id);
                            });
                            setSelectedBboxId(null);
                            setToolMode("select");
                          }}
                        >
                          스케치
                        </OptionButton>
                      </OptionGroup>
                    </>
                  )}

                  <ImageContainer>
                    <ImageViewer
                      imageUrl={undefined}
                      onImageLoad={() => console.log("이미지 로드 완료")}
                      imageRef={imageRef}
                      placeholderRef={placeholderRef}
                    />
                    <UnifiedCanvas
                      bboxes={compositionState.bboxes}
                      selectedBboxId={selectedBboxId}
                      onBboxClick={(bboxId) => {
                        setSelectedBboxId(bboxId);
                      }}
                      onAddBbox={(bbox) => {
                        // BBOX 모드가 아니면 무시
                        if (compositionMode !== "bbox") return;
                        addBbox(bbox);
                      }}
                      onUpdateBbox={updateBbox}
                      onRemoveBbox={(bboxId) => {
                        removeBbox(bboxId);
                        if (selectedBboxId === bboxId) {
                          setSelectedBboxId(null);
                        }
                      }}
                      onClearSelection={() => {
                        setSelectedBboxId(null);
                      }}
                      objects={compositionState.objects}
                      selectedObjectId={compositionState.selectedObjectId}
                      selectedObjectColor={
                        compositionState.objects.find(
                          (obj) => obj.id === compositionState.selectedObjectId
                        )?.color || null
                      }
                      toolMode={toolMode}
                      editable={true}
                      sketchLayers={
                        compositionMode === "sketch" ? sketchLayers : []
                      }
                      onSketchUpdate={(layers) => {
                        // 스케치 모드가 아니면 무시
                        if (compositionMode !== "sketch") return;
                        setSketchLayers(layers);
                      }}
                      imageRef={imageRef}
                      placeholderRef={placeholderRef}
                    />
                    {compositionMode && (
                      <FloatingToolbox
                        toolMode={toolMode}
                        onToolChange={setToolMode}
                        selectedObjectColor={
                          compositionState.objects.find(
                            (obj) =>
                              obj.id === compositionState.selectedObjectId
                          )?.color || null
                        }
                        enabledTools={
                          compositionMode === "bbox"
                            ? ["select", "bbox"]
                            : ["select", "sketch", "eraser"]
                        }
                      />
                    )}
                  </ImageContainer>
                </Section>
              </RightColumn>
            </TwoColumnLayout>
          )}

          <ActionButtonGroup>
            <ActionButton variant="cancel" onClick={handleCancel}>
              취소
            </ActionButton>
            <ActionButton
              variant="submit"
              onClick={handleComplete}
              disabled={!currentPrompt || isLoadingObjects}
            >
              {compositionMode === "bbox" && compositionState.bboxes.length > 0
                ? "구도 설정 완료 및 이미지 생성 시작"
                : compositionMode === "sketch" && sketchLayers.length > 0
                ? "구도 설정 완료 및 이미지 생성 시작"
                : "구도 없이 이미지 생성 시작"}
            </ActionButton>
          </ActionButtonGroup>
        </ModalContent>
      </ModalContainer>
    </ModalOverlay>
  );
};

export default CompositionModal;
