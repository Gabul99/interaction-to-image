import React from "react";
import { Handle, Position } from "reactflow";
import styled from "styled-components";

const NodeContainer = styled.div<{ selected: boolean }>`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border: 2px solid ${(props) => (props.selected ? "#6366f1" : "rgba(255, 255, 255, 0.2)")};
  border-radius: 12px;
  padding: 8px;
  min-width: 200px;
  max-width: 300px;
  box-shadow: ${(props) =>
    props.selected
      ? "0 8px 32px rgba(99, 102, 241, 0.4)"
      : "0 4px 16px rgba(0, 0, 0, 0.2)"};
  transition: all 0.2s ease;
`;

const ImageWrapper = styled.div`
  position: relative;
  width: 100%;
  aspect-ratio: 1;
  border-radius: 8px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.3);
  min-height: 200px;
`;

const PlaceholderImage = styled.div`
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
  font-size: 12px;
  position: absolute;
  top: 0;
  left: 0;
`;

const Image = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
  position: relative;
  z-index: 1;
`;

const StepBadge = styled.div`
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  font-size: 11px;
  font-weight: 600;
  padding: 4px 8px;
  border-radius: 4px;
`;

const NodeLabel = styled.div`
  color: #9ca3af;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-top: 8px;
  text-align: center;
`;

const BranchingButton = styled.button<{ visible: boolean }>`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  border: none;
  border-radius: 8px;
  padding: 10px 20px;
  color: white;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  opacity: ${(props) => (props.visible ? 1 : 0)};
  pointer-events: ${(props) => (props.visible ? "auto" : "none")};
  transition: all 0.2s ease;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
  z-index: 10;

  &:hover {
    transform: translate(-50%, -50%) translateY(-2px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5);
  }

  &:active {
    transform: translate(-50%, -50%);
  }
`;

interface ImageNodeData {
  imageUrl?: string;
  step?: number;
  sessionId?: string;
  onBranchClick?: () => void;
}

interface ImageNodeProps {
  data: ImageNodeData;
  selected?: boolean;
  id: string;
}

const ImageNode: React.FC<ImageNodeProps> = ({ data, selected, id }) => {
  const [imageLoaded, setImageLoaded] = React.useState(false);
  const handleBranchClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (data?.onBranchClick) {
      data.onBranchClick();
    }
  };

  React.useEffect(() => {
    if (data?.imageUrl) {
      console.log(`[ImageNode] 이미지 URL 업데이트: nodeId=${id}, imageUrl=${data.imageUrl.substring(0, 100)}, step=${data.step}`);
      setImageLoaded(false);
    } else {
      console.log(`[ImageNode] 이미지 URL 없음: nodeId=${id}, step=${data?.step}`);
    }
  }, [data?.imageUrl, id, data?.step]);

  return (
    <>
      <Handle type="target" position={Position.Left} style={{ background: "#6366f1" }} />
      <NodeContainer selected={selected || false}>
        <ImageWrapper>
          {/* Placeholder - 항상 표시 (이미지 로드 전까지) */}
          {!imageLoaded && (
            <PlaceholderImage>
              {data?.step !== undefined ? `Step ${data.step}` : "로딩 중..."}
            </PlaceholderImage>
          )}
          {/* 실제 이미지 */}
          {data?.imageUrl && (
            <Image
              src={data.imageUrl}
              alt={`Step ${data.step || ""}`}
              onLoad={() => {
                console.log(`[ImageNode] 이미지 로드 완료: nodeId=${id}, step=${data.step}`);
                setImageLoaded(true);
              }}
              onError={(e) => {
                console.error(`[ImageNode] 이미지 로드 실패: nodeId=${id}, step=${data.step}, url=${data.imageUrl}`);
                console.error(e);
              }}
              style={{ opacity: imageLoaded ? 1 : 0 }}
            />
          )}
          {data?.step !== undefined && imageLoaded && (
            <StepBadge>Step {data.step}</StepBadge>
          )}
          <BranchingButton visible={selected || false} onClick={handleBranchClick}>
            Branching
          </BranchingButton>
        </ImageWrapper>
        <NodeLabel>이미지</NodeLabel>
      </NodeContainer>
      <Handle type="source" position={Position.Right} style={{ background: "#6366f1" }} />
    </>
  );
};

export default ImageNode;

