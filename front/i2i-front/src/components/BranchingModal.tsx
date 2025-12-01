import React, { useState, useRef } from "react";
import styled from "styled-components";
import {
  type FeedbackArea,
  type FeedbackType,
  type FeedbackRecord,
  type ToolMode,
  type BoundingBox,
} from "../types";
import { useImageStore } from "../stores/imageStore";
import { forkAtStep, applyGuidance } from "../lib/api";
import ImageViewer from "./ImageViewer";
import UnifiedCanvas from "./UnifiedCanvas";
import FloatingToolbox from "./FloatingToolbox";

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
  max-width: 1600px;
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
  flex-shrink: 0;
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
  min-height: 0;
`;

const TwoColumnLayout = styled.div`
  display: flex;
  gap: 24px;
  flex: 1;
  min-height: 0;
`;

const ImageColumn = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 24px;
  overflow-y: auto;
  min-width: 0; /* flex item이 shrink될 수 있도록 */
`;

const RightColumn = styled.div`
  flex: 0 0 400px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  overflow-y: auto;
`;

const SectionTitle = styled.h3`
  color: #d1d5db;
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 12px;
  margin-top: 0;

  &:first-child {
    margin-top: 0;
  }
`;

const AddFeedbackButton = styled.button`
  width: 100%;
  padding: 12px 20px;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  border: none;
  border-radius: 8px;
  color: white;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-bottom: 24px;

  &:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
  }

  &:disabled {
    opacity: 0.6;
  }
`;

const FeedbackList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 24px;
`;

const FeedbackItem = styled.div`
  padding: 16px;
  background: rgba(55, 65, 81, 0.5);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  position: relative;
`;

const FeedbackItemHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
`;

const FeedbackItemInfo = styled.div`
  flex: 1;
`;

const FeedbackAreaBadge = styled.span<{ area: FeedbackArea }>`
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
  margin-right: 8px;
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
  border: 1px solid
    ${(props) => {
      if (props.area === "full") return "rgba(99, 102, 241, 0.3)";
      if (props.area === "bbox") return "rgba(139, 92, 246, 0.3)";
      return "rgba(236, 72, 153, 0.3)";
    }};
`;

const DeleteButton = styled.button`
  padding: 4px 8px;
  background: rgba(239, 68, 68, 0.2);
  border: 1px solid rgba(239, 68, 68, 0.5);
  border-radius: 4px;
  color: #ef4444;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(239, 68, 68, 0.3);
    border-color: #ef4444;
  }
`;

const FeedbackContent = styled.div`
  color: #d1d5db;
  font-size: 14px;
  margin-bottom: 8px;
  white-space: pre-wrap;
  word-break: break-word;
`;

const AddFeedbackForm = styled.div`
  padding: 16px;
  background: rgba(99, 102, 241, 0.1);
  border-radius: 8px;
  border: 1px solid rgba(99, 102, 241, 0.2);
  margin-bottom: 24px;
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

const TextArea = styled.textarea`
  width: 100%;
  min-height: 100px;
  padding: 12px;
  background: rgba(55, 65, 81, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  color: #f9fafb;
  font-size: 14px;
  font-family: inherit;
  resize: vertical;
  margin-bottom: 12px;

  &:focus {
    outline: none;
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  }

  &::placeholder {
    color: #6b7280;
  }
`;

const FileInput = styled.input`
  display: none;
`;

const FileUploadButton = styled.button`
  width: 100%;
  padding: 10px;
  background: rgba(55, 65, 81, 0.5);
  border: 1px dashed rgba(255, 255, 255, 0.2);
  border-radius: 6px;
  color: #d1d5db;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-bottom: 12px;

  &:hover {
    background: rgba(55, 65, 81, 0.7);
    border-color: rgba(255, 255, 255, 0.3);
  }
`;

const ImagePreview = styled.div`
  position: relative;
  margin-bottom: 12px;
`;

const PreviewImage = styled.img`
  width: 100%;
  max-height: 200px;
  object-fit: contain;
  border-radius: 6px;
  background: rgba(0, 0, 0, 0.3);
`;

const RemoveImageButton = styled.button`
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 6px 12px;
  background: rgba(239, 68, 68, 0.9);
  border: none;
  border-radius: 4px;
  color: white;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(239, 68, 68, 1);
  }
`;

const FormButtonGroup = styled.div`
  display: flex;
  gap: 8px;
`;

const FormButton = styled.button<{ variant: "submit" | "cancel" }>`
  flex: 1;
  padding: 10px 16px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
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
    border: 1px solid #374151;

    &:hover {
      background: rgba(55, 65, 81, 0.7);
      border-color: #6b7280;
    }
  `}
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

const InstructionText = styled.div`
  color: #9ca3af;
  font-size: 12px;
  margin-bottom: 12px;
  padding: 10px;
  background: rgba(99, 102, 241, 0.1);
  border-radius: 6px;
  border: 1px solid rgba(99, 102, 241, 0.2);
`;

const ImageSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
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

interface BranchingModalProps {
  visible: boolean;
  nodeId: string | null;
  onClose: () => void;
  onBranchCreated?: (branchId: string, websocketUrl?: string) => void;
  compositionBboxes?: Array<{
    id: string;
    objectId: string;
    x: number;
    y: number;
    width: number;
    height: number;
    color: string;
  }>;
  // Composition 데이터 (bboxes only)
  compositionData?: {
    bboxes: BoundingBox[];
  } | null;
}

const BranchingModal: React.FC<BranchingModalProps> = ({
  visible,
  nodeId,
  onClose,
  onBranchCreated,
  compositionBboxes = [],
  compositionData = null,
}) => {
  // BranchingModal이 열릴 때 composition data 로그 출력
  React.useEffect(() => {
    if (visible && nodeId) {
      console.log("=".repeat(80));
      console.log("[BranchingModal] BranchingModal 열림");
      console.log("[BranchingModal] nodeId:", nodeId);
      console.log("[BranchingModal] 받은 compositionData prop:", {
        hasCompositionData: !!compositionData,
        compositionData: compositionData
          ? {
              bboxesCount: compositionData.bboxes?.length || 0,
              bboxes: compositionData.bboxes,
            }
          : null,
      });
      console.log("[BranchingModal] 받은 compositionBboxes prop:", {
        count: compositionBboxes.length,
        bboxes: compositionBboxes,
      });
      console.log("=".repeat(80));
    }
  }, [visible, nodeId, compositionData, compositionBboxes]);

  // 모달이 닫힐 때 모든 temporary data 초기화
  React.useEffect(() => {
    if (!visible) {
      setIsAddingFeedback(false);
      setText("");
      setImageFile(null);
      setImagePreviewUrl(null);
      setSelectedType("text");
      setSelectedAreaType("full");
      setSelectedPartialTool(null);
      setSelectedArea("full");
      setSelectedBboxIdForFeedback(null);
      setBranchingBboxes([]);
      setSelectedBboxId(null);
      setToolMode("select");
    }
  }, [visible]);

  const {
    currentFeedbackList,
    addFeedbackToCurrentList,
    removeFeedbackFromCurrentList,
    clearCurrentFeedbackList,
    currentGraphSession,
  } = useImageStore();

  const [isAddingFeedback, setIsAddingFeedback] = useState(false);
  // 영역 선택: "full" (전체 이미지) 또는 "partial" (일부)
  const [selectedAreaType, setSelectedAreaType] = useState<"full" | "partial">(
    "full"
  );
  // 일부 선택 시 사용할 도구: "bbox"만 가능
  const [selectedPartialTool, setSelectedPartialTool] = useState<"bbox" | null>(
    null
  );
  // 실제 피드백 area (서버 전송용)
  const [selectedArea, setSelectedArea] = useState<FeedbackArea>("full");
  const [selectedType, setSelectedType] = useState<FeedbackType>("text");
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [selectedBboxIdForFeedback, setSelectedBboxIdForFeedback] = useState<
    string | null
  >(null);

  // 도구 모드
  const [toolMode, setToolMode] = useState<ToolMode>("select");

  // BBOX 관리 (피드백을 위한 새로운 BBOX만 저장)
  const [branchingBboxes, setBranchingBboxes] = useState<BoundingBox[]>([]);
  const [selectedBboxId, setSelectedBboxId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const placeholderRef = useRef<HTMLDivElement>(null);

  // 노드에서 이미지 URL 가져오기
  const nodeImageUrl = React.useMemo(() => {
    if (!nodeId || !currentGraphSession) return null;
    const node = currentGraphSession.nodes.find((n) => n.id === nodeId);
    return node?.data?.imageUrl || null;
  }, [nodeId, currentGraphSession]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreviewUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFileUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveImage = () => {
    setImageFile(null);
    setImagePreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleBboxClick = (bboxId: string) => {
    // 기존 피드백의 BBOX는 클릭해도 무시
    const isExistingFeedbackBbox = existingFeedbackBboxes.some(
      (bbox) => bbox.id === bboxId
    );
    if (isExistingFeedbackBbox) {
      return;
    }

    setSelectedBboxId(bboxId);
    setSelectedBboxIdForFeedback(bboxId);
    setSelectedPartialTool("bbox");
    setSelectedArea("bbox");
  };

  const handleSubmitFeedback = async () => {
    const hasText = text.trim().length > 0;
    const hasImage = !!imageFile;

    if (!hasText && !hasImage) {
      return;
    }

    // 선택된 BBOX 찾기
    const selectedBbox = selectedBboxIdForFeedback
      ? branchingBboxes.find((b) => b.id === selectedBboxIdForFeedback)
      : null;

    const feedback: FeedbackRecord = {
      id: `feedback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      area: selectedArea,
      type: hasImage ? "image" : "text",
      text: hasText ? text : undefined,
      imageUrl: hasImage ? imagePreviewUrl || undefined : undefined,
      bbox:
        selectedArea === "bbox" && selectedBbox
          ? {
              x: selectedBbox.x,
              y: selectedBbox.y,
              width: selectedBbox.width,
              height: selectedBbox.height,
            }
          : undefined,
      bboxId:
        selectedArea === "bbox"
          ? selectedBboxIdForFeedback || undefined
          : undefined,
      timestamp: Date.now(),
    };

    addFeedbackToCurrentList(feedback);

    // 폼 초기화
    setText("");
    setImageFile(null);
    setImagePreviewUrl(null);
    setSelectedType("text");
    setSelectedAreaType("full");
    setSelectedPartialTool(null);
    setSelectedArea("full");
    setSelectedBboxIdForFeedback(null);
    setIsAddingFeedback(false);
  };

  const handleCancelAdd = () => {
    setText("");
    setImageFile(null);
    setImagePreviewUrl(null);
    setSelectedType("text");
    setSelectedAreaType("full");
    setSelectedPartialTool(null);
    setSelectedArea("full");
    setSelectedBboxIdForFeedback(null);
    setBranchingBboxes([]);
    setSelectedBboxId(null);
    setToolMode("select");
    setIsAddingFeedback(false);
  };

  const handleDeleteFeedback = (feedbackId: string) => {
    removeFeedbackFromCurrentList(feedbackId);
  };

  const handleCreateBranch = async () => {
    // Requirement change: 실제 새로운 브랜치를 생성하고, 선택 노드 위에 병렬 노드를 표시
    if (!nodeId || !currentGraphSession) return;
    try {
      const store = useImageStore.getState();
      const sessionId = store.backendSessionId || currentGraphSession.id;
      
      // Resolve step index and backend branch ID from selected node
      const selectedNode = currentGraphSession.nodes.find((n) => n.id === nodeId);
      const stepIdx = selectedNode?.data?.step || 0;
      
      // Get backend branch ID from node data (set when node was created)
      // Priority: 1. Node's backendBranchId, 2. Edge data, 3. Active branch, 4. Default "B0"
      let incomingBranchId: string;
      if (selectedNode?.data?.backendBranchId) {
        incomingBranchId = selectedNode.data.backendBranchId as string;
      } else {
        const incoming = currentGraphSession.edges.filter((e) => e.target === nodeId);
        incomingBranchId =
          (incoming.find((e) => e.type === "branch")?.data?.branchId as string | undefined) ||
          (store.backendActiveBranchId || "B0");
    }

      console.log(`[BranchingModal] Creating branch from node ${nodeId}, backend branch ${incomingBranchId}, step ${stepIdx}`);
      
      // Call backend to fork at this step
      const resp = await forkAtStep({
        session_id: sessionId,
        branch_id: incomingBranchId,
        step_index: stepIdx,
      });
      const newBranchId = resp.new_branch_id || resp.active_branch_id;

      // Apply guidance from currentFeedbackList to the new branch
      for (const feedback of currentFeedbackList) {
        if (feedback.type === "text" && feedback.text) {
          // Text guidance
          const textRegion = feedback.bbox
            ? {
                x0: feedback.bbox.x,
                y0: feedback.bbox.y,
                x1: feedback.bbox.x + feedback.bbox.width,
                y1: feedback.bbox.y + feedback.bbox.height,
              }
            : undefined;
          await applyGuidance({
            session_id: sessionId,
            branch_id: newBranchId,
            intervene_choice: "Text Guidance",
            text_input: feedback.text,
            text_scale: 2.0,
            text_region: textRegion,
          });
        } else if (feedback.type === "image" && feedback.imageUrl) {
          // Style guidance - need to convert imageUrl to File
          // For now, if we have the original file stored, use it
          // Otherwise, we need to fetch the image and create a File
          const styleRegion = feedback.bbox
            ? {
                x0: feedback.bbox.x,
                y0: feedback.bbox.y,
                x1: feedback.bbox.x + feedback.bbox.width,
                y1: feedback.bbox.y + feedback.bbox.height,
              }
            : undefined;
          
          // Try to fetch the image and create a File object
          try {
            const response = await fetch(feedback.imageUrl);
            const blob = await response.blob();
            const file = new File([blob], "style_reference.png", { type: blob.type });
            await applyGuidance({
              session_id: sessionId,
              branch_id: newBranchId,
              intervene_choice: "Style Guidance",
              style_scale: 5.0,
              style_region: styleRegion,
              style_file: file,
            });
          } catch (imgError) {
            console.error("[BranchingModal] Failed to apply style guidance:", imgError);
          }
        }
      }

      // Register new branch in graph
      useImageStore.getState().createBranchInGraph(sessionId, newBranchId, nodeId);
      // Create parallel node above: duplicate selected image
      const imageUrl = selectedNode?.data?.imageUrl || "";
      // Position slightly above the selected node
      const position = selectedNode
        ? { x: selectedNode.position.x, y: selectedNode.position.y - 220 }
        : { x: 0, y: 0 };
      useImageStore
        .getState()
        .addImageNodeToBranch(sessionId, newBranchId, imageUrl, stepIdx, position);
      // Set backend active branch to the new one
      useImageStore.getState().setBackendSessionMeta(sessionId, newBranchId);
      clearCurrentFeedbackList();
      onClose();
    } catch (error) {
      console.error("[BranchingModal] 브랜치 생성 실패:", error);
      alert("브랜치 생성에 실패했습니다. 다시 시도해주세요.");
    }
  };

  const handleStartAddingFeedback = () => {
    setIsAddingFeedback(true);

    // 이미 BBOX 피드백이 있으면 영역 지정 모드로 시작하고 바로 BBOX 그리기 모드로
    if (hasBboxFeedback) {
      setSelectedAreaType("partial");
      setSelectedPartialTool("bbox");
      setSelectedArea("bbox");
      setToolMode("bbox");
    } else if (hasFullFeedback) {
      // 전체 이미지 피드백이 있으면 영역 지정 모드로 시작 (BBOX는 나중에 선택)
      setSelectedAreaType("partial");
      setSelectedPartialTool(null);
      setSelectedArea("bbox");
      setToolMode("select");
    } else {
      // 기본값은 전체 이미지
      setSelectedAreaType("full");
      setSelectedPartialTool(null);
      setSelectedArea("full");
      setToolMode("select");
    }

    // 초기화
    setText("");
    setImageFile(null);
    setImagePreviewUrl(null);
    setSelectedType("text");
    setBranchingBboxes([]);
    setSelectedBboxId(null);
    setSelectedBboxIdForFeedback(null);
  };

  const handleCancel = () => {
    // 모든 temporary data 초기화
    clearCurrentFeedbackList();
    setIsAddingFeedback(false);
    setText("");
    setImageFile(null);
    setImagePreviewUrl(null);
    setSelectedType("text");
    setSelectedAreaType("full");
    setSelectedPartialTool(null);
    setSelectedArea("full");
    setSelectedBboxIdForFeedback(null);
    setBranchingBboxes([]);
    setSelectedBboxId(null);
    setToolMode("select");
    onClose();
  };

  const canSubmit =
    (selectedAreaType === "full" ||
      (selectedAreaType === "partial" &&
        selectedPartialTool === "bbox" &&
        selectedBboxIdForFeedback !== null)) &&
    (selectedType === "text" ? text.trim().length > 0 : imageFile !== null);

  const getAreaLabel = (area: FeedbackArea) => {
    if (area === "full") return "전체 이미지";
    if (area === "bbox") return "BBOX";
    return "";
  };

  // 전체 이미지 피드백이 있는지 확인
  const hasFullFeedback = currentFeedbackList.some(
    (feedback) => feedback.area === "full"
  );

  // 영역 지정 피드백이 있는지 확인
  const hasBboxFeedback = currentFeedbackList.some(
    (feedback) => feedback.area === "bbox"
  );

  // 이미 추가된 피드백의 BBOX들 추출
  const existingFeedbackBboxes = React.useMemo(() => {
    return currentFeedbackList
      .filter(
        (feedback) =>
          feedback.area === "bbox" && feedback.bbox && feedback.bboxId
      )
      .map((feedback) => ({
        id: feedback.bboxId!,
        objectId: feedback.bboxId!,
        x: feedback.bbox!.x,
        y: feedback.bbox!.y,
        width: feedback.bbox!.width,
        height: feedback.bbox!.height,
        color: "#8b5cf6", // 피드백 BBOX 색상
      }));
  }, [currentFeedbackList]);

  // 표시할 모든 BBOX (기존 피드백 BBOX + 현재 작성 중인 BBOX)
  const allBboxes = React.useMemo(() => {
    return [...existingFeedbackBboxes, ...branchingBboxes];
  }, [existingFeedbackBboxes, branchingBboxes]);

  if (!visible) return null;

  return (
    <ModalOverlay visible={visible} onClick={handleCancel}>
      <ModalContainer onClick={(e) => e.stopPropagation()}>
        <ModalHeader>
          <ModalTitle>브랜치 생성</ModalTitle>
          <CloseButton onClick={handleCancel}>×</CloseButton>
        </ModalHeader>
        <ModalContent>
          {nodeImageUrl ? (
            <TwoColumnLayout>
              <ImageColumn>
                <ImageSection>
                  <SectionTitle>이미지</SectionTitle>
                  <ImageContainer>
                    <ImageViewer
                      imageUrl={nodeImageUrl}
                      onImageLoad={() => console.log("이미지 로드 완료")}
                      imageRef={imageRef}
                      placeholderRef={placeholderRef}
                    />
                    {/* 통합 Canvas */}
                    <UnifiedCanvas
                      bboxes={allBboxes}
                      selectedBboxId={selectedBboxId}
                      onBboxClick={handleBboxClick}
                      onAddBbox={(bbox) => {
                        // 영역 지정 모드일 때만 BBOX 추가
                        if (
                          selectedAreaType === "partial" &&
                          selectedPartialTool
                        ) {
                          const newBbox: BoundingBox = {
                            id: `bbox_${Date.now()}_${Math.random()
                              .toString(36)
                              .substr(2, 9)}`,
                            ...bbox,
                            objectId: `bbox_${Date.now()}`,
                            color: "#6366f1", // 기본 색상
                          };
                          setBranchingBboxes([...branchingBboxes, newBbox]);
                          setSelectedBboxId(newBbox.id);
                          setSelectedBboxIdForFeedback(newBbox.id);
                          setSelectedPartialTool("bbox");
                          setSelectedArea("bbox");
                        }
                      }}
                      onUpdateBbox={(bboxId, updates) => {
                        // 기존 피드백의 BBOX는 업데이트 불가
                        const isExistingFeedbackBbox =
                          existingFeedbackBboxes.some(
                            (bbox) => bbox.id === bboxId
                          );
                        if (isExistingFeedbackBbox) {
                          return;
                        }

                        setBranchingBboxes(
                          branchingBboxes.map((bbox) =>
                            bbox.id === bboxId ? { ...bbox, ...updates } : bbox
                          )
                        );
                      }}
                      onRemoveBbox={(bboxId) => {
                        // 기존 피드백의 BBOX는 삭제 불가
                        const isExistingFeedbackBbox =
                          existingFeedbackBboxes.some(
                            (bbox) => bbox.id === bboxId
                          );
                        if (isExistingFeedbackBbox) {
                          return;
                        }

                        setBranchingBboxes(
                          branchingBboxes.filter((bbox) => bbox.id !== bboxId)
                        );
                        if (selectedBboxId === bboxId) {
                          setSelectedBboxId(null);
                          setSelectedBboxIdForFeedback(null);
                        }
                      }}
                      toolMode={toolMode}
                      editable={isAddingFeedback}
                      disabled={
                        !isAddingFeedback || selectedAreaType === "full"
                      }
                      onClearSelection={() => {
                        setSelectedBboxId(null);
                        setSelectedBboxIdForFeedback(null);
                      }}
                      imageRef={imageRef}
                      placeholderRef={placeholderRef}
                    />
                    {/* Floating Toolbox - 일부 선택 시에만 표시 */}
                    {isAddingFeedback &&
                      selectedAreaType === "partial" &&
                      selectedPartialTool && (
                        <FloatingToolbox
                          toolMode={toolMode}
                          onToolChange={setToolMode}
                          disabled={false}
                          enabledTools={["select", "bbox"]}
                          requireObject={false}
                        />
                      )}
                  </ImageContainer>
                </ImageSection>
              </ImageColumn>
              <RightColumn>
                <AddFeedbackButton
                  onClick={handleStartAddingFeedback}
                  disabled={hasFullFeedback}
                  title={
                    hasFullFeedback
                      ? "전체 이미지 피드백이 하나 있습니다. 삭제 후 추가할 수 있습니다."
                      : undefined
                  }
                >
                  + 피드백 추가하기
                </AddFeedbackButton>

                {isAddingFeedback && (
                  <AddFeedbackForm>
                    {/* 첫 번째 선택: 전체 이미지 vs 일부 */}
                    <SectionTitle>영역 선택</SectionTitle>
                    <OptionGroup>
                      <OptionButton
                        selected={selectedAreaType === "full"}
                        onClick={() => {
                          setSelectedAreaType("full");
                          setSelectedPartialTool(null);
                          setSelectedArea("full");
                          setSelectedBboxIdForFeedback(null);
                          // 전체 이미지 선택 시 BBOX 초기화
                          setBranchingBboxes([]);
                          setSelectedBboxId(null);
                          setToolMode("select");
                        }}
                        disabled={hasBboxFeedback || hasFullFeedback}
                        title={
                          hasBboxFeedback
                            ? "이미 영역 지정 피드백이 있어서 전체 이미지 피드백을 추가할 수 없습니다"
                            : hasFullFeedback
                            ? "이미 전체 이미지 피드백이 있습니다. 하나만 추가할 수 있습니다"
                            : undefined
                        }
                      >
                        전체 이미지
                      </OptionButton>
                      <OptionButton
                        selected={selectedAreaType === "partial"}
                        onClick={() => {
                          setSelectedAreaType("partial");
                          setSelectedPartialTool(null);
                          // 일부 선택 시 초기화
                          setBranchingBboxes([]);
                          setSelectedBboxId(null);
                          setSelectedBboxIdForFeedback(null);
                          setToolMode("select");
                        }}
                        disabled={hasFullFeedback}
                        title={
                          hasFullFeedback
                            ? "이미 전체 이미지 피드백이 있어서 영역 지정 피드백을 추가할 수 없습니다"
                            : undefined
                        }
                      >
                        영역 지정
                      </OptionButton>
                    </OptionGroup>

                    {/* 일부 선택 시: BBOX 그리기 */}
                    {selectedAreaType === "partial" && !selectedPartialTool && (
                      <>
                        <InstructionText>
                          이미지 위에서 BBOX를 그려주세요.
                        </InstructionText>
                        <OptionButton
                          selected={false}
                          onClick={() => {
                            setSelectedPartialTool("bbox");
                            setSelectedArea("bbox");
                            setToolMode("bbox");
                          }}
                        >
                          BBOX 그리기 시작
                        </OptionButton>
                      </>
                    )}

                    {/* 도구 선택 후: 피드백 입력 영역 표시 */}
                    {(selectedAreaType === "full" ||
                      (selectedAreaType === "partial" &&
                        selectedPartialTool === "bbox" &&
                        selectedBboxIdForFeedback !== null)) && (
                      <>
                        {selectedAreaType === "partial" && (
                          <InstructionText>
                            {selectedPartialTool === "bbox" &&
                              selectedBboxIdForFeedback &&
                              "BBOX 선택됨"}
                          </InstructionText>
                        )}

                        <SectionTitle>피드백 방식</SectionTitle>
                        <OptionGroup>
                          <OptionButton
                            selected={selectedType === "text"}
                            onClick={() => setSelectedType("text")}
                          >
                            텍스트
                          </OptionButton>
                          <OptionButton
                            selected={selectedType === "image"}
                            onClick={() => setSelectedType("image")}
                          >
                            이미지
                          </OptionButton>
                        </OptionGroup>

                        {selectedType === "text" && (
                          <TextArea
                            placeholder="피드백을 입력하세요..."
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                          />
                        )}

                        {selectedType === "image" && (
                          <>
                            <FileInput
                              ref={fileInputRef}
                              type="file"
                              accept="image/*"
                              onChange={handleFileSelect}
                            />
                            {!imageFile ? (
                              <FileUploadButton onClick={handleFileUploadClick}>
                                이미지 선택
                              </FileUploadButton>
                            ) : (
                              <ImagePreview>
                                <PreviewImage
                                  src={imagePreviewUrl || ""}
                                  alt="Preview"
                                />
                                <RemoveImageButton onClick={handleRemoveImage}>
                                  제거
                                </RemoveImageButton>
                              </ImagePreview>
                            )}
                          </>
                        )}

                        <FormButtonGroup>
                          <FormButton
                            variant="cancel"
                            onClick={handleCancelAdd}
                          >
                            취소
                          </FormButton>
                          <FormButton
                            variant="submit"
                            onClick={handleSubmitFeedback}
                            disabled={!canSubmit}
                          >
                            저장
                          </FormButton>
                        </FormButtonGroup>
                      </>
                    )}
                  </AddFeedbackForm>
                )}

                {currentFeedbackList.length > 0 && (
                  <>
                    <SectionTitle>
                      피드백 목록 ({currentFeedbackList.length})
                    </SectionTitle>
                    <FeedbackList>
                      {currentFeedbackList.map((feedback) => (
                        <FeedbackItem key={feedback.id}>
                          <FeedbackItemHeader>
                            <FeedbackItemInfo>
                              <FeedbackAreaBadge area={feedback.area}>
                                {getAreaLabel(feedback.area)}
                              </FeedbackAreaBadge>
                            </FeedbackItemInfo>
                            <DeleteButton
                              onClick={() => handleDeleteFeedback(feedback.id)}
                            >
                              삭제
                            </DeleteButton>
                          </FeedbackItemHeader>
                          {feedback.text && (
                            <FeedbackContent>{feedback.text}</FeedbackContent>
                          )}
                          {feedback.imageUrl && (
                            <div style={{ fontSize: "12px", color: "#9ca3af" }}>
                              [참조 이미지 피드백]
                            </div>
                          )}
                        </FeedbackItem>
                      ))}
                    </FeedbackList>
                  </>
                )}

                {currentFeedbackList.length === 0 && !isAddingFeedback && (
                  <InstructionText>
                    피드백을 추가하여 브랜치를 생성하세요.
                  </InstructionText>
                )}

                <ActionButtonGroup>
                  <ActionButton variant="cancel" onClick={handleCancel}>
                    취소
                  </ActionButton>
                  <ActionButton
                    variant="submit"
                    onClick={handleCreateBranch}
                    disabled={currentFeedbackList.length === 0}
                  >
                    브랜치 생성
                  </ActionButton>
                </ActionButtonGroup>
              </RightColumn>
            </TwoColumnLayout>
          ) : (
            <>
              <AddFeedbackButton
                onClick={handleStartAddingFeedback}
                disabled={hasFullFeedback}
                title={
                  hasFullFeedback
                    ? "전체 이미지 피드백이 하나 있습니다. 삭제 후 추가할 수 있습니다."
                    : undefined
                }
              >
                + 피드백 추가하기
              </AddFeedbackButton>
              {currentFeedbackList.length === 0 && !isAddingFeedback && (
                <InstructionText>
                  피드백을 추가하여 브랜치를 생성하세요.
                </InstructionText>
              )}
              <ActionButtonGroup>
                <ActionButton variant="cancel" onClick={handleCancel}>
                  취소
                </ActionButton>
                <ActionButton
                  variant="submit"
                  onClick={handleCreateBranch}
                  disabled={currentFeedbackList.length === 0}
                >
                  브랜치 생성
                </ActionButton>
              </ActionButtonGroup>
            </>
          )}
        </ModalContent>
      </ModalContainer>
    </ModalOverlay>
  );
};

export default BranchingModal;
