import React from "react";
import styled from "styled-components";

const ImageViewerContainer = styled.div`
  position: relative;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding-top: 60px;
  min-height: 60vh;
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

const PlaceholderImage = styled.div`
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

const LoadingOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(26, 26, 46, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
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
}

const ImageViewer: React.FC<ImageViewerProps> = ({
  imageUrl,
  isLoading = false,
  onImageLoad,
  imageRef,
}) => {
  console.log("imageUrl", imageUrl);
  return (
    <ImageViewerContainer>
      <ImageWrapper>
        {isLoading ? (
          <LoadingOverlay>
            <LoadingSpinner />
          </LoadingOverlay>
        ) : imageUrl ? (
          <Image
            ref={imageRef}
            src={imageUrl}
            alt="Generated image"
            onLoad={onImageLoad}
          />
        ) : (
          <PlaceholderImage>
            프롬프트를 입력하고 이미지 생성을 시작하세요
          </PlaceholderImage>
        )}
      </ImageWrapper>
    </ImageViewerContainer>
  );
};

export default ImageViewer;
