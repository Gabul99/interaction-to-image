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

const ImageRow = styled.div`
  margin-top: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const UploadLabel = styled.label`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.6);
  background: rgba(31, 41, 55, 0.8);
  color: #e5e7eb;
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;

  &:hover {
    border-color: #6366f1;
    background: rgba(55, 65, 81, 0.9);
  }
`;

const HiddenFileInput = styled.input`
  display: none;
`;

const ImagePreview = styled.img`
  width: 40px;
  height: 40px;
  border-radius: 6px;
  object-fit: cover;
  border: 1px solid rgba(148, 163, 184, 0.8);
`;

export interface SimplePromptNodeData {
  prompt: string;
  onChangePrompt: (value: string) => void;
  inputImagePreviewUrl?: string | null;
  inputImageDataUrl?: string | null;
  inputImageSourceNodeId?: string | null;
  onUploadImage?: (file: File | null) => void;
}

export interface SimplePromptNodeProps {
  id: string;
  data: SimplePromptNodeData;
  selected?: boolean;
}

const SimplePromptNode: React.FC<SimplePromptNodeProps> = ({
  data,
  selected,
}) => {
  return (
    <>
      {/* Visible target handle on the left for incoming connections (image ➝ prompt) */}
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: "#6366f1" }}
      />
      <NodeContainer selected={selected ?? false}>
        <NodeLabel>Prompt</NodeLabel>
        <PromptInput
          value={data.prompt}
          placeholder="Describe the image you want to generate..."
          onChange={(e) => data.onChangePrompt(e.target.value)}
        />
        <ImageRow>
          <UploadLabel>
            📎 Input image
            <HiddenFileInput
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (data.onUploadImage) {
                  data.onUploadImage(file ?? null);
                }
              }}
            />
          </UploadLabel>
          {data.inputImagePreviewUrl && (
            <ImagePreview src={data.inputImagePreviewUrl} alt="Input" />
          )}
        </ImageRow>
      </NodeContainer>
      {/* Visible source handle on the right: generated images connect here */}
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: "#6366f1" }}
      />
    </>
  );
};

export default SimplePromptNode;


