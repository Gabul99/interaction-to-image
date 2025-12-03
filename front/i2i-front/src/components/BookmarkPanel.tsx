import React from "react";
import styled from "styled-components";
import { useImageStore } from "../stores/imageStore";
import { FaBookmark } from "react-icons/fa";

const Panel = styled.div`
  position: absolute;
  left: 0;
  top: 60px;
  bottom: 80px;
  width: 200px;
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 0 12px 12px 0;
  z-index: 1100;
  overflow-y: auto;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;
`;

const PanelHeader = styled.div`
  padding: 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  gap: 8px;
`;

const PanelTitle = styled.h3`
  color: #f9fafb;
  font-size: 14px;
  font-weight: 600;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 6px;
`;

const BookmarkList = styled.div`
  flex: 1;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const BookmarkItem = styled.div`
  position: relative;
  width: 100%;
  aspect-ratio: 1;
  border-radius: 8px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.3);
  cursor: pointer;
  transition: all 0.2s ease;
  border: 2px solid transparent;

  &:hover {
    border-color: rgba(234, 179, 8, 0.5);
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(234, 179, 8, 0.3);
  }
`;

const BookmarkButton = styled.button`
  position: absolute;
  bottom: 6px;
  right: 6px;
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
  z-index: 10;
  opacity: 0;
  pointer-events: none;

  svg {
    width: 20px;
    height: 20px;
    color: #fbbf24;
    transition: all 0.2s ease;
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.5));
  }

  &:hover {
    svg {
      color: #f59e0b;
      transform: scale(1.15);
    }
  }

  &:active {
    svg {
      transform: scale(0.95);
    }
  }
`;

const BookmarkItemWrapper = styled.div`
  position: relative;
  width: 100%;

  &:hover ${BookmarkButton} {
    opacity: 1;
    pointer-events: auto;
  }
`;

const BookmarkImage = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
`;

const BookmarkPlaceholder = styled.div`
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
  font-size: 12px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
`;

const EmptyState = styled.div`
  padding: 24px;
  text-align: center;
  color: #9ca3af;
  font-size: 12px;
  line-height: 1.6;
`;

const StepBadge = styled.div`
  position: absolute;
  top: 4px;
  left: 4px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  font-size: 10px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 4px;
`;

interface BookmarkPanelProps {
  onNodeClick?: (nodeId: string) => void;
  // SimpleGraphCanvas를 위한 nodes prop (선택적)
  nodes?: Array<{ id: string; type: string; data?: { imageUrl?: string; step?: number } }>;
}

const BookmarkPanel: React.FC<BookmarkPanelProps> = ({ onNodeClick, nodes: externalNodes }) => {
  const { bookmarkedNodeIds, currentGraphSession, toggleBookmark } = useImageStore();

  // 북마크된 노드들의 정보 가져오기
  const bookmarkedNodes = React.useMemo(() => {
    // externalNodes가 제공되면 그것을 사용 (SimpleGraphCanvas용)
    // 그렇지 않으면 currentGraphSession 사용 (GraphCanvas용)
    const sourceNodes = externalNodes || currentGraphSession?.nodes || [];
    
    return bookmarkedNodeIds
      .map((nodeId) => {
        const node = sourceNodes.find((n) => n.id === nodeId);
        return node && node.type === "image" ? node : null;
      })
      .filter((node): node is NonNullable<typeof node> => node !== null)
      .sort((a, b) => {
        // Step 순서대로 정렬 (최신 것이 위에)
        const stepA = a.data?.step ?? 0;
        const stepB = b.data?.step ?? 0;
        return stepB - stepA;
      });
  }, [bookmarkedNodeIds, currentGraphSession, externalNodes]);

  const handleItemClick = (nodeId: string, e: React.MouseEvent) => {
    // 북마크 버튼 클릭이면 토글만 하고 이동하지 않음
    const target = e.target as HTMLElement;
    if (target.closest('button')) {
      e.stopPropagation();
      toggleBookmark(nodeId);
      return;
    }
    
    // 이미지 클릭이면 노드로 이동
    if (onNodeClick) {
      onNodeClick(nodeId);
    }
  };

  const handleBookmarkClick = (nodeId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    toggleBookmark(nodeId);
  };

  return (
    <Panel>
      <PanelHeader>
        <PanelTitle>
          🔖 북마크
          {bookmarkedNodes.length > 0 && (
            <span style={{ color: "#9ca3af", fontSize: "12px", fontWeight: "normal" }}>
              ({bookmarkedNodes.length})
            </span>
          )}
        </PanelTitle>
      </PanelHeader>
      <BookmarkList>
        {bookmarkedNodes.length === 0 ? (
          <EmptyState>
            북마크한 이미지가 없습니다.
            <br />
            <span style={{ fontSize: "11px", opacity: 0.7 }}>
              이미지에 마우스를 올려 북마크를 추가하세요.
            </span>
          </EmptyState>
        ) : (
          bookmarkedNodes.map((node) => (
            <BookmarkItemWrapper key={node.id}>
              <BookmarkItem
                onClick={(e) => handleItemClick(node.id, e)}
                title={`Step ${node.data?.step ?? "?"} - 클릭하여 이동`}
              >
                {node.data?.imageUrl ? (
                  <>
                    <BookmarkImage
                      src={node.data.imageUrl}
                      alt={`Step ${node.data.step ?? ""}`}
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                      }}
                    />
                    {node.data.step !== undefined && (
                      <StepBadge>Step {node.data.step}</StepBadge>
                    )}
                  </>
                ) : (
                  <BookmarkPlaceholder>
                    Step {node.data?.step ?? "?"}
                  </BookmarkPlaceholder>
                )}
                <BookmarkButton
                  onClick={(e) => handleBookmarkClick(node.id, e)}
                  title="북마크 해제"
                >
                  <FaBookmark />
                </BookmarkButton>
              </BookmarkItem>
            </BookmarkItemWrapper>
          ))
        )}
      </BookmarkList>
    </Panel>
  );
};

export default BookmarkPanel;

