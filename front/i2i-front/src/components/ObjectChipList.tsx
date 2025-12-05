import React, { useState } from "react";
import styled from "styled-components";
import { type ObjectChip } from "../types";

const ChipListContainer = styled.div`
  position: relative;
  width: 280px;
  height: fit-content;
  max-height: calc(100vh - 80px);
  background: rgba(26, 26, 46, 0.9);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  z-index: 1000;
  display: flex;
  flex-direction: column;
  padding: 20px;
  gap: 12px;
  overflow-y: auto;
  margin-right: 20px;
`;

const Title = styled.h3`
  color: #f9fafb;
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 8px 0;
`;

const ChipContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const Chip = styled.button<{ selected: boolean; color: string }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  background: ${(props) =>
    props.selected
      ? `rgba(${parseInt(props.color.slice(1, 3), 16)}, ${parseInt(
          props.color.slice(3, 5),
          16
        )}, ${parseInt(props.color.slice(5, 7), 16)}, 0.3)`
      : "rgba(55, 65, 81, 0.5)"};
  border: 2px solid ${(props) => (props.selected ? props.color : "#374151")};
  border-radius: 8px;
  color: #f9fafb;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;

  &:hover {
    border-color: ${(props) => props.color};
    background: ${(props) =>
      props.selected
        ? `rgba(${parseInt(props.color.slice(1, 3), 16)}, ${parseInt(
            props.color.slice(3, 5),
            16
          )}, ${parseInt(props.color.slice(5, 7), 16)}, 0.4)`
        : "rgba(55, 65, 81, 0.7)"};
  }
`;

const ChipLabel = styled.span`
  flex: 1;
  text-align: left;
`;

const ChipColorIndicator = styled.div<{ color: string }>`
  width: 16px;
  height: 16px;
  border-radius: 4px;
  background: ${(props) => props.color};
  margin-right: 8px;
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

const DeleteButton = styled.button`
  padding: 4px 8px;
  background: rgba(239, 68, 68, 0.2);
  border: 1px solid rgba(239, 68, 68, 0.5);
  border-radius: 4px;
  color: #ef4444;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(239, 68, 68, 0.3);
    border-color: #ef4444;
  }
`;

const AddObjectContainer = styled.div`
  display: flex;
  width: 100%;
  gap: 8px;
  margin-top: 8px;
`;

const AddObjectInput = styled.input`
  flex: 1;
  padding: 8px 12px;
  border: 2px solid #374151;
  border-radius: 8px;
  font-size: 14px;
  background: rgba(55, 65, 81, 0.5);
  color: #f9fafb;
  outline: none;
  transition: border-color 0.2s ease;

  &:focus {
    border-color: #6366f1;
  }

  &::placeholder {
    color: #9ca3af;
  }
`;

const AddButton = styled.button`
  padding: 8px 16px;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  flex-shrink: 0;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

interface ObjectChipListProps {
  objects: ObjectChip[];
  selectedObjectId: string | null;
  onSelectObject: (objectId: string | null) => void;
  onAddObject: (label: string) => void;
  onRemoveObject: (objectId: string) => void;
}

const ObjectChipList: React.FC<ObjectChipListProps> = ({
  objects,
  selectedObjectId,
  onSelectObject,
  onAddObject,
  onRemoveObject,
}) => {
  const [newObjectLabel, setNewObjectLabel] = useState("");

  const handleAddObject = () => {
    if (newObjectLabel.trim()) {
      onAddObject(newObjectLabel.trim());
      setNewObjectLabel("");
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleAddObject();
    }
  };

  if (objects.length === 0) {
    return null;
  }

  return (
    <ChipListContainer>
      <Title>객체 목록</Title>
      <ChipContainer>
        {objects.map((obj) => (
          <Chip
            key={obj.id}
            selected={selectedObjectId === obj.id}
            color={obj.color}
            onClick={() =>
              onSelectObject(selectedObjectId === obj.id ? null : obj.id)
            }
          >
            <ChipColorIndicator color={obj.color} />
            <ChipLabel>{obj.label}</ChipLabel>
            <DeleteButton
              onClick={(e) => {
                e.stopPropagation();
                onRemoveObject(obj.id);
              }}
            >
              삭제
            </DeleteButton>
          </Chip>
        ))}
      </ChipContainer>
      <AddObjectContainer>
        <AddObjectInput
          type="text"
          placeholder="Add an object..."
          value={newObjectLabel}
          onChange={(e) => setNewObjectLabel(e.target.value)}
          onKeyPress={handleKeyPress}
        />
        <AddButton onClick={handleAddObject} disabled={!newObjectLabel.trim()}>
          추가
        </AddButton>
      </AddObjectContainer>
    </ChipListContainer>
  );
};

export default ObjectChipList;
