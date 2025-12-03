import React from "react";
import { Handle, Position } from "reactflow";
import styled from "styled-components";
import { useImageStore } from "../stores/imageStore";
import { FaBookmark, FaRegBookmark } from "react-icons/fa";

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

const BookmarkButton = styled.button<{ $isBookmarked: boolean }>`
  position: absolute;
  bottom: 8px;
  right: 8px;
  width: auto;
  height: auto;
  padding: 4px;
  border: none;
  background: transparent;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  z-index: 20;

  svg {
    width: 24px;
    height: 24px;
    color: ${props => props.$isBookmarked 
      ? '#fbbf24' 
      : 'rgba(255, 255, 255, 0.6)'};
    transition: all 0.2s ease;
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.5));
  }

  &:hover {
    svg {
      color: ${props => props.$isBookmarked 
        ? '#f59e0b' 
        : 'rgba(255, 255, 255, 0.9)'};
      transform: scale(1.15);
    }
  }

  &:active {
    svg {
      transform: scale(0.95);
    }
  }
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

const SimpleImageNode: React.FC<ImageNodeProps> = ({ data, selected, id }) => {
  const [imageLoaded, setImageLoaded] = React.useState(false);
  const [isHovered, setIsHovered] = React.useState(false);
  const { toggleBookmark, isBookmarked } = useImageStore();

  React.useEffect(() => {
    if (data?.imageUrl) {
      setImageLoaded(false);
    }
  }, [data?.imageUrl, id]);

  const bookmarked = isBookmarked(id);
  
  const handleBookmarkClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    toggleBookmark(id);
  };

  return (
    <NodeWrapper
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
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
                  `[SimpleImageNode] 이미지 로드 실패: nodeId=${id}, url=${data.imageUrl}`
                );
                console.error(e);
              }}
              style={{ opacity: imageLoaded ? 1 : 0 }}
            />
          )}
          
          {/* 북마크 버튼 - hover 시 표시 */}
          {isHovered && (
            <BookmarkButton
              $isBookmarked={bookmarked}
              onClick={handleBookmarkClick}
              title={bookmarked ? "북마크 해제" : "북마크 추가"}
            >
              {bookmarked ? <FaBookmark /> : <FaRegBookmark />}
            </BookmarkButton>
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

export default SimpleImageNode;

