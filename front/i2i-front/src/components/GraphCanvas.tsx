import React, { useCallback, useMemo, useState, useRef, useEffect } from "react";
import ReactFlow, {
  type Node,
  type Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type NodeTypes,
  type EdgeTypes,
  ReactFlowProvider,
} from "reactflow";
import "reactflow/dist/style.css";
import styled, { createGlobalStyle } from "styled-components";
import { useImageStore } from "../stores/imageStore";
import { USE_MOCK_MODE } from "../config/api";
import { connectImageStream } from "../api/websocket";
import PromptNode from "./PromptNode";
import ImageNode from "./ImageNode";
import BranchingModal from "./BranchingModal";
import FeedbackEdge from "./FeedbackEdge";
import { stepOnce, mergeBranches, backtrackTo } from "../lib/api";

const nodeTypes: NodeTypes = {
  prompt: PromptNode,
  image: ImageNode,
};

const edgeTypes: EdgeTypes = {
  branch: FeedbackEdge,
  default: FeedbackEdge,
};

// Grid layout constants
const GRID_CELL_WIDTH = 60; // Horizontal spacing between nodes (reduced)
const GRID_CELL_HEIGHT = 280; // Vertical spacing between rows (consistent)
const GRID_START_X = 100; // Starting X position
const GRID_START_Y = 50; // Starting Y position for main branch (row 0)

// Branch colors - distinct colors for each branch
const BRANCH_COLORS = [
  "#6366f1", // Indigo (main branch B0)
  "#ec4899", // Pink
  "#f59e0b", // Amber
  "#10b981", // Emerald
  "#8b5cf6", // Violet
  "#ef4444", // Red
  "#06b6d4", // Cyan
  "#84cc16", // Lime
  "#f97316", // Orange
  "#14b8a6", // Teal
  "#a855f7", // Purple
  "#3b82f6", // Blue
];

const EmptyStateContainer = styled.div`
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
`;

const EmptyStateNode = styled.div`
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  padding: 24px 32px;
  min-width: 280px;
  max-width: 400px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  transition: all 0.2s ease;
  text-align: center;
`;

const EmptyStateLabel = styled.div`
  color: #9ca3af;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 12px;
`;

const EmptyStateText = styled.div`
  color: #f9fafb;
  font-size: 14px;
  font-weight: 500;
  word-wrap: break-word;
  line-height: 1.6;
  margin-bottom: 20px;
`;

const EmptyStateButton = styled.button`
  width: 100%;
  padding: 14px 24px;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  border: none;
  border-radius: 8px;
  color: white;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5);
  }

  &:active {
    transform: translateY(0);
  }
`;

const PlusIcon = styled.span`
  font-size: 20px;
  font-weight: 300;
  line-height: 1;
`;

// Global style for pulse animation
const GlobalPulseStyle = createGlobalStyle`
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.85;
      transform: scale(1.02);
    }
  }
`;

const MergeConfirmModal = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
`;

const MergeConfirmContent = styled.div`
  background: rgba(26, 26, 46, 0.98);
  backdrop-filter: blur(10px);
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-radius: 16px;
  padding: 24px 32px;
  max-width: 400px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
`;

const MergeTitle = styled.h3`
  color: #f9fafb;
  font-size: 18px;
  font-weight: 600;
  margin: 0 0 12px 0;
`;

const MergeDescription = styled.p`
  color: #9ca3af;
  font-size: 14px;
  line-height: 1.6;
  margin: 0 0 20px 0;
`;

const MergeButtonRow = styled.div`
  display: flex;
  gap: 12px;
  justify-content: flex-end;
`;

const MergeButton = styled.button<{ primary?: boolean }>`
  padding: 10px 20px;
  border-radius: 8px;
  border: none;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;

  ${(props) =>
    props.primary
      ? `
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 16px rgba(16, 185, 129, 0.5);
    }
  `
      : `
    background: rgba(255, 255, 255, 0.1);
    color: #9ca3af;
    &:hover {
      background: rgba(255, 255, 255, 0.15);
      color: #f9fafb;
    }
  `}
`;

// Settings Panel - top left corner
const SettingsPanel = styled.div`
  position: absolute;
  top: 16px;
  left: 16px;
  z-index: 1200;
  background: rgba(26, 26, 46, 0.95);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 12px;
  padding: 12px 16px;
  min-width: 180px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
`;

const SettingsTitle = styled.div`
  color: #9ca3af;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 10px;
`;

const SettingsRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
`;

const SettingsLabel = styled.label`
  color: #e5e7eb;
  font-size: 13px;
  font-weight: 500;
`;

const SettingsInput = styled.input`
  width: 60px;
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.08);
  color: #f9fafb;
  font-size: 13px;
  font-weight: 600;
  text-align: center;
  outline: none;
  transition: all 0.2s ease;

  &:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
  }

  &::-webkit-inner-spin-button,
  &::-webkit-outer-spin-button {
    opacity: 1;
  }
`;

interface GraphCanvasProps {
  className?: string;
  onAddNodeClick?: () => void;
}

const GraphCanvas: React.FC<GraphCanvasProps> = ({
  className,
  onAddNodeClick,
}) => {
  const {
    currentGraphSession,
    selectedNodeId,
    selectNode,
    backendSessionId,
  } = useImageStore();
  const [branchingModalVisible, setBranchingModalVisible] = useState(false);
  const [branchingNodeId, setBranchingNodeId] = useState<string | null>(null);
  const [isStepping, setIsStepping] = useState(false);
  const [isRunningToEnd, setIsRunningToEnd] = useState(false);
  
  // Step interval for preview visualization (default: 5 = show every 5th step)
  const [stepInterval, setStepInterval] = useState(5);
  
  // Pause control for Run to End - using ref so it can be checked synchronously in the loop
  const isPausedRef = useRef(false);
  const [isPaused, setIsPaused] = useState(false);
  
  // Merge state
  const [mergeConfirmVisible, setMergeConfirmVisible] = useState(false);
  const [mergeSourceNode, setMergeSourceNode] = useState<Node | null>(null);
  const [mergeTargetNode, setMergeTargetNode] = useState<Node | null>(null);
  const [isMerging, setIsMerging] = useState(false);
  const [potentialMergeTargetId, setPotentialMergeTargetId] = useState<string | null>(null);
  
  // Store original positions for snap-back
  const originalPositionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  
  // Backtracking state
  const [isBacktracking, setIsBacktracking] = useState(false);

  // 현재 선택된 노드의 composition 데이터 가져오기
  const compositionData = useMemo(() => {
    if (!branchingNodeId || !currentGraphSession) {
      console.log(
        "[GraphCanvas] Composition 데이터: 없음 (branchingNodeId 또는 currentGraphSession 없음)"
      );
      return null;
    }

    // 루트 노드(prompt 노드)에서 composition 데이터 찾기
    const rootNode = currentGraphSession.nodes.find((n) => n.type === "prompt");
    const compositionData = rootNode?.data?.compositionData || null;

    console.log("[GraphCanvas] Composition 데이터 저장 위치:", {
      branchingNodeId,
      rootNodeId: rootNode?.id,
      rootNodeType: rootNode?.type,
      hasCompositionData: !!compositionData,
      compositionData: compositionData
        ? {
            bboxesCount: compositionData.bboxes?.length || 0,
            bboxes: compositionData.bboxes,
          }
        : null,
    });

    return compositionData;
  }, [branchingNodeId, currentGraphSession]);

  // Get branch color by index - consistent colors for branches
  const getBranchColor = useCallback((branchId?: string): string => {
    if (!branchId) return BRANCH_COLORS[0]; // Default to indigo
    
    // Extract branch number from ID (e.g., "B0" -> 0, "B1" -> 1)
    const match = branchId.match(/B(\d+)/);
    if (match) {
      const index = parseInt(match[1], 10);
      return BRANCH_COLORS[index % BRANCH_COLORS.length];
    }
    
    // Fallback: hash-based color
    let hash = 0;
    for (let i = 0; i < branchId.length; i++) {
      hash = branchId.charCodeAt(i) + ((hash << 5) - hash);
    }
    return BRANCH_COLORS[Math.abs(hash) % BRANCH_COLORS.length];
  }, []);

  // 선택된 노드에서 root까지의 경로 찾기 (역순: 선택된 노드 -> root)
  const getPathToRoot = useCallback(
    (nodeId: string): string[] => {
      if (!currentGraphSession) return [];

      const path: string[] = [];
      const visited = new Set<string>();
      let currentNodeId: string | null = nodeId;

      while (currentNodeId && !visited.has(currentNodeId)) {
        visited.add(currentNodeId);
        path.push(currentNodeId);

        // 현재 노드로 들어오는 edge 찾기 (부모 찾기)
        const incomingEdge = currentGraphSession.edges.find(
          (e) => e.target === currentNodeId
        );

        if (incomingEdge) {
          currentNodeId = incomingEdge.source;
        } else {
          // root 노드에 도달
          break;
        }
      }

      return path; // [선택된노드, ..., root] 순서
    },
    [currentGraphSession]
  );

  // Helper to get branch ID from a node
  const getNodeBranchId = useCallback(
    (nodeId: string): string | null => {
      if (!currentGraphSession) return null;
      const node = currentGraphSession.nodes.find((n) => n.id === nodeId);
      if (node?.data?.backendBranchId) {
        return node.data.backendBranchId as string;
      }
      // Fallback to edge data
      const incoming = currentGraphSession.edges.find((e) => e.target === nodeId);
      if (incoming?.data?.branchId) {
        return incoming.data.branchId as string;
      }
      // Default to B0 for main branch
      return "B0";
    },
    [currentGraphSession]
  );

  // Get all nodes in the same branch as the selected node (both ancestors and descendants)
  const getFullBranchNodes = useCallback(
    (nodeId: string): string[] => {
      if (!currentGraphSession) return [];

      const selectedBranchId = getNodeBranchId(nodeId);
      if (!selectedBranchId) return [nodeId];

      const branchNodes: string[] = [];
      const visited = new Set<string>();

      // First, find all nodes that belong to this branch
      // For main branch (B0), we need to trace the main path
      // For other branches, we find nodes with matching branchId

      if (selectedBranchId === "B0") {
        // For main branch, trace from root to the end of main branch
        // Start from the prompt node and follow edges that are not branch edges
        const promptNode = currentGraphSession.nodes.find((n) => n.type === "prompt");
        if (promptNode) {
          branchNodes.push(promptNode.id);
          visited.add(promptNode.id);

          // Follow main branch edges (non-branch type edges or edges without branchId)
          let currentNodeId: string | null = promptNode.id;
          while (currentNodeId) {
            const outgoingEdges = currentGraphSession.edges.filter(
              (e) => e.source === currentNodeId
            );
            
            // Find the main branch edge (type !== 'branch' or no branchId)
            const mainEdge = outgoingEdges.find(
              (e) => e.type !== "branch" || !e.data?.branchId
            );
            
            if (mainEdge && !visited.has(mainEdge.target)) {
              branchNodes.push(mainEdge.target);
              visited.add(mainEdge.target);
              currentNodeId = mainEdge.target;
            } else {
              break;
            }
          }
        }
      } else {
        // For non-main branches, find all nodes with matching branchId
        // Find the branch source node (where this branch started)
        const branch = currentGraphSession.branches.find((b) => b.id === selectedBranchId);
        const branchSourceNodeId = branch?.sourceNodeId;

        // Add all nodes in this branch
        for (const node of currentGraphSession.nodes) {
          const nodeBranchId = getNodeBranchId(node.id);
          if (nodeBranchId === selectedBranchId) {
            branchNodes.push(node.id);
            visited.add(node.id);
          }
        }

        // Also include the path from branch source to root (ancestor nodes in main branch)
        if (branchSourceNodeId) {
          const ancestorPath = getPathToRoot(branchSourceNodeId);
          for (const ancestorId of ancestorPath) {
            if (!visited.has(ancestorId)) {
              branchNodes.push(ancestorId);
              visited.add(ancestorId);
            }
          }
        }
      }

      return branchNodes;
    },
    [currentGraphSession, getNodeBranchId, getPathToRoot]
  );

  // Get all edges in the full branch
  const getFullBranchEdges = useCallback(
    (branchNodeIds: Set<string>): Set<string> => {
      if (!currentGraphSession) return new Set();

      const edgeIds = new Set<string>();
      
      for (const edge of currentGraphSession.edges) {
        // Include edge if both source and target are in the branch
        if (branchNodeIds.has(edge.source) && branchNodeIds.has(edge.target)) {
          edgeIds.add(edge.id);
        }
      }

      return edgeIds;
    },
    [currentGraphSession]
  );

  // 선택된 노드의 전체 브랜치 계산 (ancestors + descendants)
  const selectedBranchNodeIds = useMemo(() => {
    if (!selectedNodeId || !currentGraphSession) return new Set<string>();
    const branchNodes = getFullBranchNodes(selectedNodeId);
    return new Set(branchNodes);
  }, [selectedNodeId, currentGraphSession, getFullBranchNodes]);

  const selectedBranchEdgeIds = useMemo(() => {
    return getFullBranchEdges(selectedBranchNodeIds);
  }, [selectedBranchNodeIds, getFullBranchEdges]);

  // Get the branch color for the selected node's branch
  const selectedBranchColor = useMemo(() => {
    if (!selectedNodeId || !currentGraphSession) return null;
    const branchId = getNodeBranchId(selectedNodeId);
    return branchId ? getBranchColor(branchId) : null;
  }, [selectedNodeId, currentGraphSession, getNodeBranchId, getBranchColor]);

  // Get the rightmost node in the selected branch for the arrow indicator
  const rightmostBranchNodeId = useMemo(() => {
    if (!selectedNodeId || !currentGraphSession) return null;
    
    const branchId = getNodeBranchId(selectedNodeId);
    if (!branchId) return null;
    
    // Find all nodes in this branch
    let branchNodesList: string[] = [];
    
    if (branchId === "B0") {
      // Main branch - get all main branch nodes
      branchNodesList = Array.from(selectedBranchNodeIds);
    } else {
      // Non-main branch - get nodes with this branch ID
      for (const node of currentGraphSession.nodes) {
        if (getNodeBranchId(node.id) === branchId) {
          branchNodesList.push(node.id);
        }
      }
    }
    
    // Find the node with the highest X position (rightmost)
    let rightmostNode: { id: string; x: number } | null = null;
    for (const nodeId of branchNodesList) {
      const node = currentGraphSession.nodes.find((n) => n.id === nodeId);
      if (node && node.type === "image") {
        if (!rightmostNode || node.position.x > rightmostNode.x) {
          rightmostNode = { id: node.id, x: node.position.x };
        }
      }
    }
    
    return rightmostNode?.id || null;
  }, [selectedNodeId, currentGraphSession, selectedBranchNodeIds, getNodeBranchId]);

  // Check if the selected branch is completed (reached num_steps)
  // This is determined by checking if the rightmost node's step equals num_steps
  const isBranchCompleted = useMemo(() => {
    if (!selectedNodeId || !currentGraphSession || !rightmostBranchNodeId) return false;
    
    const rightmostNode = currentGraphSession.nodes.find((n) => n.id === rightmostBranchNodeId);
    if (!rightmostNode || rightmostNode.type !== "image") return false;
    
    const step = rightmostNode.data?.step;
    // We consider the branch completed if the step is at or beyond num_steps
    // Note: num_steps is typically 50, and steps are 0-indexed, so step 50 means done
    // We'll check if step >= 49 (assuming 50 total steps, 0-49)
    // But since we don't have num_steps in frontend, we check if status was "done"
    // For now, we'll assume a branch is complete if step >= 49 (for 50 steps)
    // This should be improved by tracking completion status from backend
    if (step !== undefined && step >= 49) {
      return true;
    }
    return false;
  }, [selectedNodeId, currentGraphSession, rightmostBranchNodeId]);

  // Calculate grid position for a node based on step and branch
  const calculateGridPosition = useCallback((step: number, branchIndex: number) => {
    return {
      x: GRID_START_X + step * GRID_CELL_WIDTH,
      y: GRID_START_Y + branchIndex * GRID_CELL_HEIGHT,
    };
  }, []);

  // Get branch row index (main branch = 0, others = 1, 2, 3...)
  const getBranchRowIndex = useCallback((branchId: string): number => {
    if (!currentGraphSession) return 0;
    if (branchId === "B0") return 0;
    
    // Find branch index (excluding main branch)
    const nonMainBranches = currentGraphSession.branches.filter((b) => b.id !== "B0");
    const index = nonMainBranches.findIndex((b) => b.id === branchId);
    return index >= 0 ? index + 1 : 1; // +1 because main branch is row 0
  }, [currentGraphSession]);

  // React-flow의 nodes와 edges를 store의 데이터와 동기화
  const nodes: Node[] = useMemo(() => {
    if (!currentGraphSession) return [];
    
    // Store original positions for snap-back
    const newOriginalPositions = new Map<string, { x: number; y: number }>();
    
    return currentGraphSession.nodes.map((node) => {
      const isInBranch = selectedBranchNodeIds.has(node.id);
      const isSelected = selectedNodeId === node.id;
      const isMergeTarget = node.id === potentialMergeTargetId;
      const isRightmost = node.id === rightmostBranchNodeId;
      
      // Calculate fixed grid position based on step and branch
      let fixedPosition = node.position;
      if (node.type === "image") {
        const step = node.data?.step ?? 0;
        const branchId = getNodeBranchId(node.id) || "B0";
        const rowIndex = getBranchRowIndex(branchId);
        fixedPosition = calculateGridPosition(step, rowIndex);
      } else if (node.type === "prompt") {
        // Prompt node at column -1 (before step 0)
        fixedPosition = { x: GRID_START_X - GRID_CELL_WIDTH, y: GRID_START_Y };
      }
      
      // Store for snap-back
      newOriginalPositions.set(node.id, fixedPosition);
      
      // Determine node style based on state
      let nodeStyle: React.CSSProperties | undefined;
      if (isMergeTarget) {
        nodeStyle = {
          boxShadow: "0 0 20px 5px rgba(16, 185, 129, 0.6)",
          border: "3px solid #10b981",
          borderRadius: "12px",
        };
      } else if (isSelected && selectedBranchColor) {
        nodeStyle = {
          boxShadow: `0 0 20px 5px ${selectedBranchColor}80`,
          border: `3px solid ${selectedBranchColor}`,
          borderRadius: "12px",
        };
      } else if (isInBranch && selectedBranchColor) {
        nodeStyle = {
          boxShadow: `0 0 12px 3px ${selectedBranchColor}50`,
          border: `2px solid ${selectedBranchColor}`,
          borderRadius: "12px",
        };
      }

      return {
        id: node.id,
        type: node.type,
        position: fixedPosition,
        data: {
          ...node.data,
          onBranchClick:
            node.type === "image"
              ? () => {
                  setBranchingNodeId(node.id);
                  setBranchingModalVisible(true);
                }
              : undefined,
          isMergeTarget,
          isInBranch,
          isRightmost,
          branchColor: isRightmost ? selectedBranchColor : undefined,
        },
        selected: isSelected,
        style: nodeStyle,
        draggable: true, // Allow dragging for merge detection
      };
    });
  }, [
    currentGraphSession,
    selectedNodeId,
    potentialMergeTargetId,
    selectedBranchNodeIds,
    selectedBranchColor,
    rightmostBranchNodeId,
    getNodeBranchId,
    getBranchRowIndex,
    calculateGridPosition,
  ]);

  // Update original positions ref when nodes change
  React.useEffect(() => {
    const newPositions = new Map<string, { x: number; y: number }>();
    nodes.forEach((node) => {
      newPositions.set(node.id, node.position);
    });
    originalPositionsRef.current = newPositions;
  }, [nodes]);

  const edges: Edge[] = useMemo(() => {
    if (!currentGraphSession) return [];
    
    const edgeList: Edge[] = currentGraphSession.edges.map((edge) => {
      const branchId = edge.data?.branchId || "B0";
      const branchColor = getBranchColor(branchId);
      const isInBranch = selectedBranchEdgeIds.has(edge.id);
      const isMergeEdge = edge.data?.isMergeEdge === true;

      // Merge edges get a special green color and dashed style
      const mergeColor = "#10b981"; // Emerald green for merge edges

      // Use selected branch color for branch highlighting
      const branchHighlightColor = selectedBranchColor || branchColor;

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edge.type || "default",
        data: edge.data,
        animated: isInBranch,
        style: {
          stroke: isInBranch
            ? branchHighlightColor
            : isMergeEdge
            ? mergeColor
            : branchColor,
          strokeWidth: isInBranch
            ? 5 // Thick for selected branch
            : isMergeEdge
            ? 4
            : 3, // Thicker default edges
          strokeDasharray: isMergeEdge ? "5,5" : undefined,
        },
      };
    });

    return edgeList;
  }, [currentGraphSession, getBranchColor, selectedBranchEdgeIds, selectedBranchColor, rightmostBranchNodeId]);

  const [reactFlowNodes, setNodes, onNodesChange] = useNodesState(nodes);
  const [reactFlowEdges, setEdges, onEdgesChange] = useEdgesState(edges);

  // nodes와 edges가 변경되면 React-flow 상태 업데이트
  React.useEffect(() => {
    setNodes(nodes);
  }, [nodes, setNodes]);

  React.useEffect(() => {
    setEdges(edges);
  }, [edges, setEdges]);

  // Find potential merge target: another node at the same step but different branch
  // Uses reactFlowNodes for accurate position after drag
  const findMergeTarget = useCallback(
    (draggedNode: Node): Node | null => {
      if (!currentGraphSession) return null;
      if (draggedNode.type !== "image") return null;

      const draggedStep = draggedNode.data?.step;
      const draggedBranchId = getNodeBranchId(draggedNode.id);

      console.log(`[GraphCanvas] findMergeTarget: dragged node ${draggedNode.id}, step=${draggedStep}, branch=${draggedBranchId}, pos=(${draggedNode.position.x.toFixed(0)}, ${draggedNode.position.y.toFixed(0)})`);

      if (draggedStep === undefined || draggedBranchId === null) {
        console.log(`[GraphCanvas] findMergeTarget: skipping - no step or branch`);
        return null;
      }

      // Find nodes from different branches within proximity
      // Now allows merging from different steps!
      // Use reactFlowNodes for accurate positions (they get updated during drag)
      const MERGE_PROXIMITY = 200; // pixels - increased for easier merging

      for (const node of reactFlowNodes) {
        if (node.id === draggedNode.id) continue;
        if (node.type !== "image") continue;

        const nodeStep = node.data?.step;
        const nodeBranchId = getNodeBranchId(node.id);

        console.log(`[GraphCanvas] findMergeTarget: checking ${node.id}, step=${nodeStep}, branch=${nodeBranchId}, pos=(${node.position.x.toFixed(0)}, ${node.position.y.toFixed(0)})`);

        // Must be different branch (but can be any step now)
        if (nodeBranchId === draggedBranchId) {
          console.log(`[GraphCanvas] findMergeTarget: same branch`);
          continue;
        }

        // Check proximity
        const dx = Math.abs(draggedNode.position.x - node.position.x);
        const dy = Math.abs(draggedNode.position.y - node.position.y);
        const distance = Math.sqrt(dx * dx + dy * dy);

        console.log(`[GraphCanvas] findMergeTarget: distance=${distance.toFixed(0)}, threshold=${MERGE_PROXIMITY}`);

        if (distance < MERGE_PROXIMITY) {
          console.log(
            `[GraphCanvas] Merge candidate found: ${draggedNode.id} (branch ${draggedBranchId}, step ${draggedStep}) -> ${node.id} (branch ${nodeBranchId}, step ${nodeStep})`
          );
          return node;
        }
      }

      console.log(`[GraphCanvas] findMergeTarget: no merge target found`);
      return null;
    },
    [currentGraphSession, getNodeBranchId, reactFlowNodes]
  );

  // 노드 드래그 종료 시 - snap back to original position or show merge dialog
  const onNodeDragStop = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      // Clear potential merge target highlight
      setPotentialMergeTargetId(null);
      
      if (!currentGraphSession) return;

      // Check for merge target
      const mergeTarget = findMergeTarget(node);
      if (mergeTarget) {
        // Show merge confirmation
        setMergeSourceNode(node);
        setMergeTargetNode(mergeTarget);
        setMergeConfirmVisible(true);
      }

      // Always snap back to original position (grid-aligned)
      const originalPos = originalPositionsRef.current.get(node.id);
      if (originalPos) {
        setNodes((nds) =>
          nds.map((n) =>
            n.id === node.id ? { ...n, position: originalPos } : n
          )
        );
      }
    },
    [currentGraphSession, findMergeTarget, setNodes]
  );

  // Handle merge confirmation
  const handleMergeConfirm = useCallback(async () => {
    if (!mergeSourceNode || !mergeTargetNode || !currentGraphSession) return;

    const sessionId = backendSessionId || currentGraphSession.id;
    const sourceBranchId = getNodeBranchId(mergeSourceNode.id);
    const targetBranchId = getNodeBranchId(mergeTargetNode.id);
    const sourceStep = mergeSourceNode.data?.step;
    const targetStep = mergeTargetNode.data?.step;

    if (!sourceBranchId || !targetBranchId || sourceStep === undefined || targetStep === undefined) {
      console.error("[GraphCanvas] Cannot merge: missing branch ID or step");
      setMergeConfirmVisible(false);
      return;
    }

    console.log(
      `[GraphCanvas] Merging branches: ${sourceBranchId}@${sourceStep} + ${targetBranchId}@${targetStep}`
    );

    setIsMerging(true);
    try {
      const result = await mergeBranches({
        session_id: sessionId,
        branch_id_1: sourceBranchId,
        branch_id_2: targetBranchId,
        step_index_1: sourceStep,
        step_index_2: targetStep,
        merge_weight: 0.5,
      });

      console.log("[GraphCanvas] Merge result:", result);

      if (result.new_branch_id) {
        // Update store with new branch info
        const { setBackendSessionMeta, createMergedBranchWithNode } =
          useImageStore.getState();
        
        // Use the graph session ID for graph operations
        const graphSessionId = currentGraphSession.id;
        
        // Get the actual merge start step from the response
        const mergeStartStep = result.merge_steps?.start_step ?? Math.max(sourceStep, targetStep);
        
        console.log(`[GraphCanvas] Creating merged branch: ${result.new_branch_id} in session ${graphSessionId}`);
        console.log(`[GraphCanvas] Source nodes: ${mergeSourceNode.id}@${sourceStep}, ${mergeTargetNode.id}@${targetStep}`);
        console.log(`[GraphCanvas] Merged branch starts at step ${mergeStartStep}`);
        
        // Update active branch in backend session meta
        setBackendSessionMeta(sessionId, result.new_branch_id);

        // Calculate position for merged branch node using grid layout
        const newBranchRowIndex = currentGraphSession.branches.length; // New branch gets next row
        const mergeNodePosition = calculateGridPosition(mergeStartStep, newBranchRowIndex);

        // Use a placeholder image or the target node's image for the initial merged node
        // The actual merged preview will be generated on the next step
        const initialImageUrl = mergeTargetNode.data?.imageUrl || mergeSourceNode.data?.imageUrl || "";
        
        console.log(`[GraphCanvas] Adding merged branch node at step ${mergeStartStep}, image length: ${initialImageUrl?.length || 0}`);
        
        // Create the merged branch and its initial node atomically
        // Connect to BOTH source nodes to visually show the merge
        const newNodeId = createMergedBranchWithNode(
          graphSessionId,
          result.new_branch_id,
          mergeSourceNode.id, // First source node
          mergeTargetNode.id, // Second source node
          initialImageUrl,
          mergeStartStep,
          mergeNodePosition
        );

        console.log(`[GraphCanvas] Created merged branch node: ${newNodeId} at position:`, mergeNodePosition);

        // Verify node was created
        const finalState = useImageStore.getState();
        const newNode = finalState.currentGraphSession?.nodes.find(n => n.id === newNodeId);
        console.log(`[GraphCanvas] New node created:`, newNode);
        console.log(`[GraphCanvas] Total nodes:`, finalState.currentGraphSession?.nodes.length);
        console.log(`[GraphCanvas] Total edges:`, finalState.currentGraphSession?.edges.length);
        console.log(`[GraphCanvas] Total branches:`, finalState.currentGraphSession?.branches.length);

        if (!newNodeId) {
          console.warn(`[GraphCanvas] Merge request processed but node creation failed`);
        }
      }
    } catch (error) {
      console.error("[GraphCanvas] Merge failed:", error);
      alert("브랜치 병합에 실패했습니다.");
    } finally {
      setIsMerging(false);
      setMergeConfirmVisible(false);
      setMergeSourceNode(null);
      setMergeTargetNode(null);
    }
  }, [
    mergeSourceNode,
    mergeTargetNode,
    currentGraphSession,
    backendSessionId,
    getNodeBranchId,
    calculateGridPosition,
  ]);

  // Handle merge cancellation
  const handleMergeCancel = useCallback(() => {
    // Restore node to original position (or just close modal)
    setMergeConfirmVisible(false);
    setMergeSourceNode(null);
    setMergeTargetNode(null);
  }, []);

  // Handle node drag to show visual feedback for potential merge
  const onNodeDrag = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      const mergeTarget = findMergeTarget(node);
      setPotentialMergeTargetId(mergeTarget?.id || null);
    },
    [findMergeTarget]
  );

  // Clear potential merge target when drag starts
  const onNodeDragStart = useCallback(() => {
    setPotentialMergeTargetId(null);
  }, []);

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  const onConnect = useCallback(() => {
    // 자동 연결은 비활성화 (수동으로 엣지를 만들지 않음)
  }, []);

  const handleBranchCreated = useCallback(
    (branchId: string, websocketUrl?: string) => {
      console.log("브랜치 생성됨:", branchId, websocketUrl);
      setBranchingModalVisible(false);
      setBranchingNodeId(null);

      if (!currentGraphSession) return;

      // Mock 모드 체크
      if (USE_MOCK_MODE) {
        console.log("[GraphCanvas] Mock 모드: 브랜치 시뮬레이션 시작");
        const { simulateBranchImageStream } = useImageStore.getState();
        simulateBranchImageStream(currentGraphSession.id, branchId);
        return;
      }

      // 백엔드에서 websocketUrl을 받은 경우 실제 WebSocket 연결
      if (websocketUrl) {
        const { addImageNodeToBranch } = useImageStore.getState();

        connectImageStream(
          currentGraphSession.id,
          websocketUrl,
          (imageStep) => {
            // 브랜치의 이미지 스텝 수신
            console.log(
              "[GraphCanvas] 브랜치 이미지 스텝 수신:",
              imageStep.step
            );

            // 브랜치에 이미지 노드 추가
            const branch = currentGraphSession.branches.find(
              (b) => b.id === branchId
            );
            if (branch) {
              // position은 addImageNodeToBranch에서 자동 계산되므로 전달하지 않음
              addImageNodeToBranch(
                currentGraphSession.id,
                branchId,
                imageStep.url,
                imageStep.step || 0
              );
            }
          },
          (error) => {
            console.error("[GraphCanvas] 브랜치 WebSocket 에러:", error);
          },
          () => {
            console.log("[GraphCanvas] 브랜치 이미지 생성 완료");
          }
        );
      } else {
        // websocketUrl이 없는 경우 시뮬레이션으로 처리
        const { simulateBranchImageStream } = useImageStore.getState();
        simulateBranchImageStream(currentGraphSession.id, branchId);
      }
    },
    [currentGraphSession]
  );

  // Helper to get target branch ID from selected node
  // IMPORTANT: This hook must be before any early returns to follow Rules of Hooks
  const getTargetBranchFromSelectedNode = useCallback(() => {
    const gs = useImageStore.getState().currentGraphSession;
    const activeBranch = useImageStore.getState().backendActiveBranchId;
    let targetBranchId = activeBranch || "B0";
    
    if (selectedNodeId && gs) {
      const selectedNode = gs.nodes.find((n) => n.id === selectedNodeId);
      if (selectedNode?.data?.backendBranchId) {
        targetBranchId = selectedNode.data.backendBranchId as string;
      } else {
        const incoming = gs.edges.filter((e) => e.target === selectedNodeId);
        const incomingBranch = incoming.find((e) => e.type === "branch");
        if (incomingBranch?.data?.branchId) {
          targetBranchId = incomingBranch.data.branchId as string;
        }
      }
    }
    return targetBranchId;
  }, [selectedNodeId]);

  // Handle Next Step click - runs stepInterval steps and shows the final preview
  // IMPORTANT: This hook must be before any early returns to follow Rules of Hooks
  const handleNextStep = useCallback(async () => {
    try {
      if (!currentGraphSession || !selectedNodeId) return;
      const { backendSessionId, addImageNode, addImageNodeToBranch } =
        useImageStore.getState();
      const sessionId = backendSessionId || currentGraphSession.id;
      const targetBranchId = getTargetBranchFromSelectedNode();
      
      console.log(`[GraphCanvas] Next Step: selectedNodeId=${selectedNodeId}, targetBranchId=${targetBranchId}, stepInterval=${stepInterval}`);
      setIsStepping(true);
      
      // Run stepInterval steps, only showing the last preview
      let lastResp: Awaited<ReturnType<typeof stepOnce>> | null = null;
      
      for (let i = 0; i < stepInterval; i++) {
        const resp = await stepOnce({
          session_id: sessionId,
          branch_id: targetBranchId,
        });
        lastResp = resp;
        
        // Check if we've reached the end
        if (resp.i >= resp.num_steps) {
          console.log(`[GraphCanvas] Reached end at step ${resp.i}/${resp.num_steps}`);
          break;
        }
      }
      
      // Only add preview for the last step
      if (lastResp?.preview_png_base64) {
        const gs = useImageStore.getState().currentGraphSession;
        if (targetBranchId === "B0") {
          const promptNode = gs?.nodes.find((n) => n.type === "prompt");
          const mainImageNodes = (gs?.nodes || []).filter((n) => {
            if (n.type !== "image") return false;
            const inEdge = (gs?.edges || []).find((e) => e.target === n.id);
            return !inEdge || inEdge.type !== "branch";
          });
          const lastMain = mainImageNodes
            .slice()
            .sort((a, b) => (a.data?.step || 0) - (b.data?.step || 0))
            .pop();
          const parentNodeId = lastMain?.id || promptNode?.id || null;
          if (parentNodeId) {
            // Position will be calculated by grid layout
            const pos = calculateGridPosition(lastResp.i, 0);
            addImageNode(sessionId, parentNodeId, lastResp.preview_png_base64, lastResp.i, pos);
          }
        } else {
          addImageNodeToBranch(sessionId, targetBranchId, lastResp.preview_png_base64, lastResp.i);
        }
      }
    } catch (e) {
      console.error("[GraphCanvas] Next Step failed:", e);
    } finally {
      setIsStepping(false);
    }
  }, [currentGraphSession, selectedNodeId, getTargetBranchFromSelectedNode, stepInterval, calculateGridPosition]);

  // Handle Run to End click - runs step by step showing each preview based on stepInterval
  // IMPORTANT: This hook must be before any early returns to follow Rules of Hooks
  const handleRunToEnd = useCallback(async () => {
    try {
      if (!currentGraphSession || !selectedNodeId) return;
      const { backendSessionId } =
        useImageStore.getState();
      const sessionId = backendSessionId || currentGraphSession.id;
      const targetBranchId = getTargetBranchFromSelectedNode();
      
      console.log(`[GraphCanvas] Run to End: selectedNodeId=${selectedNodeId}, targetBranchId=${targetBranchId}, stepInterval=${stepInterval}`);
      setIsRunningToEnd(true);
      isPausedRef.current = false;
      setIsPaused(false);
      
      // Run step by step until completion, showing preview based on stepInterval
      let stepCount = 0;
      const maxSteps = 100; // Safety limit
      
      while (stepCount < maxSteps) {
        // Check if paused
        if (isPausedRef.current) {
          console.log(`[GraphCanvas] Run to End paused at step ${stepCount}`);
          break;
        }
        
        // Get fresh state for each step
        const { addImageNodeToBranch, addImageNode, currentGraphSession: gs } =
          useImageStore.getState();
        
        const resp = await stepOnce({
          session_id: sessionId,
          branch_id: targetBranchId,
        });
        
        console.log(`[GraphCanvas] Run to End step ${resp.i}/${resp.num_steps}`);
        
        stepCount++;
        
        // Only add preview to graph based on stepInterval
        // Show if: step is divisible by interval, or it's the last step
        const isLastStep = resp.status === "done" || resp.i >= resp.num_steps;
        const shouldShowPreview = (resp.i % stepInterval === 0) || isLastStep;
        
        if (resp.preview_png_base64 && shouldShowPreview) {
          console.log(`[GraphCanvas] Adding preview for step ${resp.i} (interval: ${stepInterval})`);
          if (targetBranchId === "B0") {
            const promptNode = gs?.nodes.find((n) => n.type === "prompt");
            const mainImageNodes = (gs?.nodes || []).filter((n) => {
              if (n.type !== "image") return false;
              const inEdge = (gs?.edges || []).find((e) => e.target === n.id);
              return !inEdge || inEdge.type !== "branch";
            });
            const lastMain = mainImageNodes
              .slice()
              .sort((a, b) => (a.data?.step || 0) - (b.data?.step || 0))
              .pop();
            const parentNodeId = lastMain?.id || promptNode?.id || null;
            if (parentNodeId) {
              // Position will be calculated by grid layout
              const pos = calculateGridPosition(resp.i, 0);
              addImageNode(sessionId, parentNodeId, resp.preview_png_base64, resp.i, pos);
            }
          } else {
            addImageNodeToBranch(sessionId, targetBranchId, resp.preview_png_base64, resp.i);
          }
        }
        
        // Check if we've reached the end
        if (isLastStep) {
          console.log(`[GraphCanvas] Run to End completed at step ${resp.i}/${resp.num_steps}`);
          break;
        }
      }
      
      console.log(`[GraphCanvas] Run to End finished after ${stepCount} steps`);
    } catch (e) {
      console.error("[GraphCanvas] Run to End failed:", e);
      alert("Run to End 실패: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setIsRunningToEnd(false);
      isPausedRef.current = false;
      setIsPaused(false);
    }
  }, [currentGraphSession, selectedNodeId, getTargetBranchFromSelectedNode, stepInterval, calculateGridPosition]);

  // Handle Pause button click
  const handlePause = useCallback(() => {
    isPausedRef.current = true;
    setIsPaused(true);
    console.log("[GraphCanvas] Pause requested");
  }, []);

  // Handle backtracking - remove selected node and all descendants
  const handleBacktrack = useCallback(async () => {
    if (!currentGraphSession || !selectedNodeId) return;
    if (isBacktracking || isStepping || isRunningToEnd) return;
    
    // Find the selected node
    const selectedNode = currentGraphSession.nodes.find((n) => n.id === selectedNodeId);
    if (!selectedNode || selectedNode.type !== "image") {
      console.log("[GraphCanvas] Cannot backtrack: selected node is not an image node");
      return;
    }
    
    const step = selectedNode.data?.step;
    if (step === undefined || step === 0) {
      console.log("[GraphCanvas] Cannot backtrack: already at step 0 or no step info");
      return;
    }
    
    const { backendSessionId, removeNodeAndDescendants } = useImageStore.getState();
    const sessionId = backendSessionId || currentGraphSession.id;
    const targetBranchId = getTargetBranchFromSelectedNode();
    
    console.log(`[GraphCanvas] Backtracking: node=${selectedNodeId}, step=${step}, branch=${targetBranchId}`);
    setIsBacktracking(true);
    
    try {
      // Call backend to backtrack
      const result = await backtrackTo({
        session_id: sessionId,
        branch_id: targetBranchId,
        step_index: step - 1, // Go back one step before the selected node
      });
      
      console.log("[GraphCanvas] Backtrack result:", result);
      
      // Remove the node and its descendants from the frontend
      removeNodeAndDescendants(currentGraphSession.id, selectedNodeId);
      
      console.log("[GraphCanvas] Backtrack completed");
    } catch (error) {
      console.error("[GraphCanvas] Backtrack failed:", error);
    } finally {
      setIsBacktracking(false);
    }
  }, [currentGraphSession, selectedNodeId, isBacktracking, isStepping, isRunningToEnd, getTargetBranchFromSelectedNode]);

  // Keyboard event handler for backspace
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Check if backspace is pressed and we have a selected node
      if (event.key === "Backspace" && selectedNodeId && currentGraphSession) {
        // Don't trigger if user is typing in an input
        const target = event.target as HTMLElement;
        if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) {
          return;
        }
        
        event.preventDefault();
        handleBacktrack();
      }
    };
    
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedNodeId, currentGraphSession, handleBacktrack]);

  // Early return for empty state - MUST be after all hooks
  if (!currentGraphSession) {
    return (
      <EmptyStateContainer>
        <EmptyStateNode>
          <EmptyStateLabel>안내</EmptyStateLabel>
          <EmptyStateText>
            그래프가 없습니다.
            <br />
            아래 버튼을 눌러 새 이미지 생성을 시작하세요.
          </EmptyStateText>
          <EmptyStateButton onClick={onAddNodeClick}>
            <PlusIcon>+</PlusIcon>새 이미지 생성 시작
          </EmptyStateButton>
        </EmptyStateNode>
      </EmptyStateContainer>
    );
  }

  return (
    <ReactFlowProvider>
      <GlobalPulseStyle />
      <div style={{ width: "100%", height: "100%" }} className={className}>
        {/* Settings Panel - top left corner */}
        <SettingsPanel>
          <SettingsTitle>⚙️ Settings</SettingsTitle>
          <SettingsRow>
            <SettingsLabel htmlFor="stepInterval">Preview Interval</SettingsLabel>
            <SettingsInput
              id="stepInterval"
              type="number"
              min={1}
              max={20}
              value={stepInterval}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                if (!isNaN(val) && val >= 1 && val <= 20) {
                  setStepInterval(val);
                }
              }}
              disabled={isRunningToEnd}
              title="Show preview every N steps (1 = every step, 2 = every 2nd step, etc.)"
            />
          </SettingsRow>
        </SettingsPanel>

        {/* Control buttons at bottom center of canvas */}
        <div
          style={{
            position: "absolute",
            left: "50%",
            bottom: "24px",
            transform: "translateX(-50%)",
            zIndex: 1200,
            display: "flex",
            gap: "12px",
          }}
        >
          {/* Next Step Button */}
          <button
            onClick={handleNextStep}
            disabled={!currentGraphSession || !selectedNodeId || isStepping || isRunningToEnd || isBranchCompleted}
            style={{
              padding: "10px 16px",
              borderRadius: 8,
              border: "none",
              fontWeight: 700,
              cursor: !currentGraphSession || !selectedNodeId || isStepping || isRunningToEnd || isBranchCompleted ? "not-allowed" : "pointer",
              background: !currentGraphSession || !selectedNodeId || isStepping || isRunningToEnd || isBranchCompleted
                ? "linear-gradient(135deg, #4b5563 0%, #6b7280 100%)"
                : "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
              color: "#fff",
              opacity: !currentGraphSession || !selectedNodeId || isStepping || isRunningToEnd || isBranchCompleted ? 0.6 : 1,
            }}
            title={
              !selectedNodeId 
                ? "노드를 선택하세요" 
                : isBranchCompleted 
                ? "이 브랜치는 완료되었습니다" 
                : "선택된 브랜치에서 다음 스텝을 수행합니다"
            }
          >
            {isStepping ? "Processing..." : !selectedNodeId ? "Select a node" : isBranchCompleted ? "Completed ✓" : "Next Step"}
          </button>
          
          {/* Run to End Button */}
          <button
            onClick={handleRunToEnd}
            disabled={!currentGraphSession || !selectedNodeId || isStepping || isRunningToEnd || isBranchCompleted}
            style={{
              padding: "10px 16px",
              borderRadius: 8,
              border: "none",
              fontWeight: 700,
              cursor: !currentGraphSession || !selectedNodeId || isStepping || isRunningToEnd || isBranchCompleted ? "not-allowed" : "pointer",
              background: !currentGraphSession || !selectedNodeId || isStepping || isRunningToEnd || isBranchCompleted
                ? "linear-gradient(135deg, #4b5563 0%, #6b7280 100%)"
                : "linear-gradient(135deg, #10b981 0%, #059669 100%)",
              color: "#fff",
              opacity: !currentGraphSession || !selectedNodeId || isStepping || isRunningToEnd || isBranchCompleted ? 0.6 : 1,
            }}
            title={
              !selectedNodeId 
                ? "노드를 선택하세요" 
                : isBranchCompleted 
                ? "이 브랜치는 완료되었습니다" 
                : "선택된 브랜치를 끝까지 자동으로 진행합니다"
            }
          >
            {isRunningToEnd ? "Running..." : isBranchCompleted ? "Completed ✓" : "Run to End"}
          </button>

          {/* Pause Button - only visible when running */}
          {isRunningToEnd && (
            <button
              onClick={handlePause}
              disabled={isPaused}
              style={{
                padding: "10px 16px",
                borderRadius: 8,
                border: "none",
                fontWeight: 700,
                cursor: isPaused ? "not-allowed" : "pointer",
                background: isPaused
                  ? "linear-gradient(135deg, #4b5563 0%, #6b7280 100%)"
                  : "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
                color: "#fff",
                opacity: isPaused ? 0.6 : 1,
                animation: isPaused ? "none" : "pulse 1.5s infinite",
              }}
              title="현재 진행 중인 생성을 일시 정지합니다"
            >
              {isPaused ? "Pausing..." : "⏸ Pause"}
            </button>
          )}
        </div>
        <ReactFlow
          nodes={reactFlowNodes}
          edges={reactFlowEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onNodeDragStart={onNodeDragStart}
          onNodeDrag={onNodeDrag}
          onNodeDragStop={onNodeDragStop}
          onPaneClick={onPaneClick}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          attributionPosition="bottom-left"
        >
          <Background />
          <Controls />
          <MiniMap />
        </ReactFlow>
      </div>
      <BranchingModal
        visible={branchingModalVisible}
        nodeId={branchingNodeId}
        onClose={() => {
          setBranchingModalVisible(false);
          setBranchingNodeId(null);
        }}
        onBranchCreated={handleBranchCreated}
        compositionData={compositionData}
      />

      {/* Merge Confirmation Modal */}
      {mergeConfirmVisible && mergeSourceNode && mergeTargetNode && (
        <MergeConfirmModal onClick={handleMergeCancel}>
          <MergeConfirmContent onClick={(e) => e.stopPropagation()}>
            <MergeTitle>🔀 브랜치 병합</MergeTitle>
            <MergeDescription>
              두 브랜치를 병합하시겠습니까?
              <br />
              <br />
              <strong>소스:</strong> {getNodeBranchId(mergeSourceNode.id)} (Step{" "}
              {mergeSourceNode.data?.step})
              <br />
              <strong>타겟:</strong> {getNodeBranchId(mergeTargetNode.id)} (Step{" "}
              {mergeTargetNode.data?.step})
              <br />
              <br />
              {mergeSourceNode.data?.step !== mergeTargetNode.data?.step ? (
                <>
                  <span style={{ color: "#fbbf24" }}>⚠️ 서로 다른 스텝의 latent를 병합합니다.</span>
                  <br />
                  병합된 브랜치는 스텝 {Math.max(mergeSourceNode.data?.step ?? 0, mergeTargetNode.data?.step ?? 0)}부터 시작합니다.
                  <br />
                  <br />
                </>
              ) : null}
              병합된 브랜치는 두 브랜치의 가이던스 설정을 모두 유지하며,
              Extended Attention을 사용하여 두 latent를 결합합니다.
            </MergeDescription>
            <MergeButtonRow>
              <MergeButton onClick={handleMergeCancel} disabled={isMerging}>
                취소
              </MergeButton>
              <MergeButton
                primary
                onClick={handleMergeConfirm}
                disabled={isMerging}
              >
                {isMerging ? "병합 중..." : "병합"}
              </MergeButton>
            </MergeButtonRow>
          </MergeConfirmContent>
        </MergeConfirmModal>
      )}
    </ReactFlowProvider>
  );
};

export default GraphCanvas;
