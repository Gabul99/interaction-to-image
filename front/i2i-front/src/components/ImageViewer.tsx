import React from "react";
import styled from "styled-components";

const ImageViewerContainer = styled.div`
  position: relative;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 60px;
  height: fit-content;
  width: 100%;
`;

const ImageWrapper = styled.div`
  position: relative;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
  background: #f8fafc;
  border: 2px solid #e2e8f0;
`;

const Image = styled.img`
  max-width: 100%;
  max-height: 70vh;
  width: auto;
  height: auto;
  display: block;
`;

const PlaceholderImage = styled.div.withConfig({
  shouldForwardProp: () => true,
})`
  width: 512px;
  height: 512px;
  background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
  font-size: 18px;
  font-weight: 500;
  border-radius: 12px;
  border: 2px solid #374151;
`;

const LoadingSpinner = styled.div`
  width: 40px;
  height: 40px;
  border: 3px solid #374151;
  border-top: 3px solid #6366f1;
  border-radius: 50%;
  animation: spin 1s linear infinite;

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
`;

interface ImageViewerProps {
  imageUrl?: string;
  isLoading?: boolean;
  onImageLoad?: () => void;
  imageRef?: React.RefObject<HTMLImageElement | null>;
  placeholderRef?: React.RefObject<HTMLDivElement | null>;
}

const ImageViewer: React.FC<ImageViewerProps> = ({
  imageUrl,
  isLoading = false,
  onImageLoad,
  imageRef,
  placeholderRef,
}) => {
  console.log("imageUrl", imageUrl);

  // 이미지 URL이 있으면 항상 이미지를 표시 (중간 과정을 보여주기 위해)
  // 각 스텝 이미지가 도착하면 즉시 표시되어야 함
  return (
    <ImageViewerContainer>
      <ImageWrapper>
        {imageUrl ? (
          <Image
            ref={imageRef}
            src={imageUrl}
            alt="Generated image"
            onLoad={onImageLoad}
          />
        ) : (
          <PlaceholderImage ref={placeholderRef}>
            {isLoading ? (
              <>
                <LoadingSpinner />
                <div style={{ marginTop: "16px" }}>이미지 생성 중...</div>
              </>
            ) : (
              "객체를 선택하고 이 곳에서 구도를 설정하세요"
            )}
          </PlaceholderImage>
        )}
      </ImageWrapper>
    </ImageViewerContainer>
  );
};

export default ImageViewer;
