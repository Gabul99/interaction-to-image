import React from "react";
import { Handle, Position } from "reactflow";
import styled from "styled-components";

const NodeContainer = styled.div<{ selected: boolean }>`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border: 2px solid
    ${(props) => (props.selected ? "#6366f1" : "rgba(255, 255, 255, 0.2)")};
  border-radius: 12px;
  padding: 16px 20px;
  min-width: 260px;
  max-width: 360px;
  box-shadow: ${(props) =>
    props.selected
      ? "0 8px 32px rgba(99, 102, 241, 0.4)"
      : "0 4px 16px rgba(0, 0, 0, 0.2)"};
  transition: all 0.2s ease;
`;

const NodeLabel = styled.div`
  color: #9ca3af;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
`;

const PromptInput = styled.textarea`
  width: 100%;
  min-height: 60px;
  max-height: 120px;
  resize: vertical;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid #4b5563;
  background: rgba(31, 41, 55, 0.9);
  color: #f9fafb;
  font-size: 13px;
  line-height: 1.5;
  outline: none;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;

  &::placeholder {
    color: #6b7280;
  }

  &:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.6);
  }
`;

const ActionRow = styled.div`
  margin-top: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const CompositionButton = styled.button<{ hasComposition?: boolean }>`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 999px;
  border: 1px solid
    ${(props) =>
      props.hasComposition ? "rgba(56, 189, 248, 0.9)" : "rgba(148, 163, 184, 0.6)"};
  background: ${(props) =>
    props.hasComposition
      ? "linear-gradient(135deg, rgba(56, 189, 248, 0.3) 0%, rgba(129, 140, 248, 0.5) 100%)"
      : "rgba(31, 41, 55, 0.8)"};
  color: ${(props) => (props.hasComposition ? "#f9fafb" : "#e5e7eb")};
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;

  &:hover {
    border-color: ${(props) =>
      props.hasComposition ? "rgba(56, 189, 248, 1)" : "#6366f1"};
    background: ${(props) =>
      props.hasComposition
        ? "linear-gradient(135deg, rgba(56, 189, 248, 0.4) 0%, rgba(129, 140, 248, 0.6) 100%)"
        : "rgba(55, 65, 81, 0.9)"};
  }
`;

const GenerateButton = styled.button`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 18px;
  border-radius: 999px;
  border: 1px solid rgba(99, 102, 241, 0.6);
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
  color: #e5e7eb;
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;

  &:hover:not(:disabled) {
    border-color: #6366f1;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

interface PromptNodeData {
  prompt?: string;
  sessionId?: string;
  onChangePrompt?: (value: string) => void;
  onFocusPrompt?: () => void;
  onOpenComposition?: () => void;
  onGenerate?: () => void;
  hasGeneratedImages?: boolean; // 이미 생성된 이미지가 있는지 여부
  // Composition data attached to this prompt (from bbox or sketch)
  compositionData?: {
    bboxes?: unknown[];
    sketchLayers?: unknown[];
  };
}

interface PromptNodeProps {
  data: PromptNodeData;
  selected?: boolean;
}

const PromptNode: React.FC<PromptNodeProps> = ({ data, selected }) => {
  const handleCompositionClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (data?.onOpenComposition) {
      data.onOpenComposition();
    }
  };

  // 사용자가 BBOX 또는 스케치 입력을 설정한 경우 (compositionData 존재 여부로 판단)
  const hasComposition =
    !!data.compositionData &&
    (((data.compositionData.bboxes || []).length > 0) ||
      ((data.compositionData.sketchLayers || []).length > 0));

  const handleGenerateClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (data?.onGenerate) {
      data.onGenerate();
    }
  };

  return (
    <>
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <NodeContainer selected={selected || false}>
        <NodeLabel>Prompt</NodeLabel>
        <PromptInput
          value={data.prompt || ""}
          placeholder="Describe the image you want to generate..."
          onChange={(e) => {
            if (data.onChangePrompt) {
              data.onChangePrompt(e.target.value);
            }
          }}
          onFocus={() => {
            if (data.onFocusPrompt) {
              data.onFocusPrompt();
            }
          }}
          onClick={(e) => {
            e.stopPropagation();
            if (data.onFocusPrompt) {
              data.onFocusPrompt();
            }
          }}
        />
        <ActionRow>
          <CompositionButton
            hasComposition={hasComposition}
            onClick={handleCompositionClick}
          >
            🎨 Set Layout
          </CompositionButton>
          <GenerateButton 
            onClick={handleGenerateClick}
            disabled={!data.prompt || data.prompt.trim().length === 0}
          >
            {data.hasGeneratedImages ? "Restart" : "Start"}
          </GenerateButton>
        </ActionRow>
      </NodeContainer>
      <Handle type="source" position={Position.Right} style={{ background: "#6366f1" }} />
    </>
  );
};

export default PromptNode;

