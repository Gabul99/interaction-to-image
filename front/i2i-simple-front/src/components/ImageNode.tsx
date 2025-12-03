import React from "react";
import { Handle, Position } from "reactflow";
import styled from "styled-components";

// Simple image node without captions or branching UI
const NodeWrapper = styled.div`
  position: relative;
`;

const NodeContainer = styled.div<{ selected: boolean }>`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border: 2px solid
    ${(props) => (props.selected ? "#6366f1" : "rgba(255, 255, 255, 0.2)")};
  border-radius: 12px;
  padding: 8px;
  min-width: 180px;
  max-width: 220px;
  box-shadow: ${(props) =>
    props.selected
      ? "0 8px 32px rgba(99, 102, 241, 0.4)"
      : "0 4px 16px rgba(0, 0, 0, 0.2)"};
  transition: all 0.2s ease;
  position: relative;
`;

const ImageWrapper = styled.div`
  position: relative;
  width: 100%;
  aspect-ratio: 1;
  border-radius: 8px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.3);
  min-height: 150px;
`;

const PlaceholderImage = styled.div`
  width: 100%;
  height: 100%;
  background: linear-gradient(
    135deg,
    rgba(99, 102, 241, 0.1) 0%,
    rgba(139, 92, 246, 0.1) 100%
  );
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

interface ImageNodeData {
  imageUrl?: string;
  step?: number;
}

interface ImageNodeProps {
  data: ImageNodeData;
  selected?: boolean;
  id: string;
}

const ImageNode: React.FC<ImageNodeProps> = ({ data, selected, id }) => {
  const [imageLoaded, setImageLoaded] = React.useState(false);

  React.useEffect(() => {
    if (data?.imageUrl) {
      setImageLoaded(false);
    }
  }, [data?.imageUrl, id]);

  return (
    <NodeWrapper>
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: "#6366f1", top: "50%" }}
      />

      <NodeContainer selected={selected || false}>
        <ImageWrapper>
          {!imageLoaded && <PlaceholderImage />}
          {data?.imageUrl && (
            <Image
              src={data.imageUrl}
              alt="Generated"
              onLoad={() => {
                setImageLoaded(true);
              }}
              onError={(e) => {
                console.error(
                  `[ImageNode] 이미지 로드 실패: nodeId=${id}, url=${data.imageUrl}`
                );
                console.error(e);
              }}
              style={{ opacity: imageLoaded ? 1 : 0 }}
            />
          )}
        </ImageWrapper>
      </NodeContainer>

      <Handle
        type="source"
        position={Position.Right}
        style={{ background: "#6366f1", top: "50%" }}
      />
    </NodeWrapper>
  );
};

export default ImageNode;
