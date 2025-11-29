import React from "react";
import styled from "styled-components";

const Button = styled.button`
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  border: none;
  color: white;
  font-size: 32px;
  font-weight: 300;
  cursor: pointer;
  box-shadow: 0 8px 32px rgba(99, 102, 241, 0.4);
  transition: all 0.2s ease;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  line-height: 1;

  &:hover {
    transform: translate(-50%, -50%) scale(1.1);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.5);
  }

  &:active {
    transform: translate(-50%, -50%) scale(0.95);
  }
`;

interface AddNodeButtonProps {
  onClick: () => void;
  visible?: boolean;
}

const AddNodeButton: React.FC<AddNodeButtonProps> = ({
  onClick,
  visible = true,
}) => {
  if (!visible) return null;

  return <Button onClick={onClick}>+</Button>;
};

export default AddNodeButton;

