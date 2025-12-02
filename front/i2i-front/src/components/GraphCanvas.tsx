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
import { useImageStore, getBranchRowIndex, extractBackendBranchId, createUniqueBranchId } from "../stores/imageStore";
import { USE_MOCK_MODE } from "../config/api";
import { connectImageStream } from "../api/websocket";
import { type GraphNode } from "../types";
import PromptNode from "./PromptNode";
import ImageNode from "./ImageNode";
import PlaceholderNode from "./PlaceholderNode";
import LoadingNode from "./LoadingNode";
import BranchingModal from "./BranchingModal";
import FeedbackEdge from "./FeedbackEdge";
import { stepOnce, mergeBranches, backtrackTo } from "../lib/api";

const nodeTypes: NodeTypes = {
  prompt: PromptNode,
  image: ImageNode,
  placeholder: PlaceholderNode,
  loading: LoadingNode,
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
    initializeEmptyGraphSession,
    addPlaceholderNode,
    addEdge,
    updateNodePosition,
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
  const [mergePlaceholderNodeId, setMergePlaceholderNodeId] = useState<string | null>(null); // Placeholder node ID 저장
  const [isMerging, setIsMerging] = useState(false);
  const [potentialMergeTargetId, setPotentialMergeTargetId] = useState<string | null>(null);
  
  // Store original positions for snap-back
  const originalPositionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  
  // Track which nodes have been manually moved (prompt nodes and their children)
  // These nodes should use their stored position instead of grid calculation
  const movedNodesRef = useRef<Set<string>>(new Set());
  
  // Track prompt offset from original grid position - used to offset new nodes added to moved prompts
  // Key: prompt node ID, Value: { deltaX, deltaY } offset from original grid position
  const promptOffsetRef = useRef<Map<string, { deltaX: number; deltaY: number }>>(new Map());
  
  // Track prompt node dragging for moving child nodes together
  const draggingPromptRef = useRef<{
    nodeId: string;
    startPosition: { x: number; y: number };
    childNodeIds: string[];
    childStartPositions: Map<string, { x: number; y: number }>;
  } | null>(null);

  // 초기 렌더링 시 빈 GraphSession 생성
  useEffect(() => {
    if (!currentGraphSession) {
      initializeEmptyGraphSession();
    }
  }, []); // 최초 한 번만 실행
  
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

  // Helper to get unique branch ID from a node
  // Returns the uniqueBranchId (e.g., "sess_abc123_B0") for proper filtering across parallel sessions
  const getNodeBranchId = useCallback(
    (nodeId: string): string | null => {
      if (!currentGraphSession) return null;
      const node = currentGraphSession.nodes.find((n) => n.id === nodeId);
      
      // Priority: uniqueBranchId > edge branchId > backendBranchId
      if (node?.data?.uniqueBranchId) {
        return node.data.uniqueBranchId as string;
      }
      
      // Fallback to edge data (which now stores unique branch ID)
      const incoming = currentGraphSession.edges.find((e) => e.target === nodeId);
      if (incoming?.data?.branchId) {
        return incoming.data.branchId as string;
      }
      
      // Legacy fallback: construct unique ID from backendBranchId and backendSessionId
      if (node?.data?.backendBranchId && node?.data?.backendSessionId) {
        return createUniqueBranchId(node.data.backendSessionId as string, node.data.backendBranchId as string);
      }
      
      // Final fallback to backendBranchId (for backwards compatibility)
      if (node?.data?.backendBranchId) {
        return node.data.backendBranchId as string;
      }
      
      // Default to null (unknown branch)
      return null;
    },
    [currentGraphSession]
  );

  // Get the full trajectory of a node: all ancestors (parents) and all descendants (children)
  // This highlights the complete path from root to all leaf nodes through the selected node
  const getFullTrajectory = useCallback(
    (nodeId: string): string[] => {
      if (!currentGraphSession) return [];

      const trajectoryNodes: string[] = [];
      const visited = new Set<string>();

      // 1. Get all ancestors (trace back to root)
      const ancestors: string[] = [];
      let currentId: string | null = nodeId;
      while (currentId) {
        if (visited.has(currentId)) break;
        ancestors.unshift(currentId); // Add to front to maintain order from root
        visited.add(currentId);
        
        // Find parent via incoming edge
        const incomingEdge = currentGraphSession.edges.find((e) => e.target === currentId);
        currentId = incomingEdge ? incomingEdge.source : null;
      }
      
      // Add ancestors to trajectory
      trajectoryNodes.push(...ancestors);

      // 2. Get all descendants (BFS from selected node)
      const queue: string[] = [nodeId];
      const descendantVisited = new Set<string>([nodeId]); // nodeId already in ancestors
      
      while (queue.length > 0) {
        const current = queue.shift()!;
        
        // Find all children via outgoing edges
        const outgoingEdges = currentGraphSession.edges.filter((e) => e.source === current);
        
        for (const edge of outgoingEdges) {
          if (!descendantVisited.has(edge.target) && !visited.has(edge.target)) {
            descendantVisited.add(edge.target);
            visited.add(edge.target);
            trajectoryNodes.push(edge.target);
            queue.push(edge.target);
          }
        }
      }

      return trajectoryNodes;
    },
    [currentGraphSession]
  );

  // Legacy function for branch-based selection (kept for compatibility)
  const getFullBranchNodes = useCallback(
    (nodeId: string): string[] => {
      // Now delegates to getFullTrajectory for consistent behavior
      return getFullTrajectory(nodeId);
    },
    [getFullTrajectory]
  );

  // Get all descendants (children) of a node - used for moving prompt node with its children
  const getAllDescendants = useCallback(
    (nodeId: string): string[] => {
      if (!currentGraphSession) return [];

      const descendants: string[] = [];
      const visited = new Set<string>([nodeId]);
      const queue: string[] = [nodeId];
      
      while (queue.length > 0) {
        const current = queue.shift()!;
        
        // Find all children via outgoing edges
        const outgoingEdges = currentGraphSession.edges.filter((e) => e.source === current);
        
        for (const edge of outgoingEdges) {
          if (!visited.has(edge.target)) {
            visited.add(edge.target);
            descendants.push(edge.target);
            queue.push(edge.target);
          }
        }
      }

      return descendants;
    },
    [currentGraphSession]
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

  // 선택된 노드의 전체 trajectory 계산 (모든 ancestors + 모든 descendants)
  // This highlights the complete path from root through the selected node to all its children
  const selectedBranchNodeIds = useMemo(() => {
    if (!selectedNodeId || !currentGraphSession) return new Set<string>();
    const trajectoryNodes = getFullBranchNodes(selectedNodeId);
    return new Set(trajectoryNodes);
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

  // Calculate position for a new node, applying prompt offset if the prompt has been moved
  const calculatePositionWithOffset = useCallback(
    (step: number, rowIndex: number, promptNodeId: string | null): { x: number; y: number } => {
      const basePosition = calculateGridPosition(step, rowIndex);
      
      if (promptNodeId) {
        const offset = promptOffsetRef.current.get(promptNodeId);
        if (offset) {
          return {
            x: basePosition.x + offset.deltaX,
            y: basePosition.y + offset.deltaY,
          };
        }
      }
      
      return basePosition;
    },
    [calculateGridPosition]
  );

  // Get branch row index using unified function from imageStore
  const getBranchRowIndexLocal = useCallback((branchId: string): number => {
    if (!currentGraphSession) return 0;
    return getBranchRowIndex(branchId, currentGraphSession.branches, currentGraphSession.nodes);
  }, [currentGraphSession]);

  // React-flow의 nodes와 edges를 store의 데이터와 동기화
  const nodes: Node[] = useMemo(() => {
    if (!currentGraphSession) return [];
    
    // Store original positions for snap-back
    const newOriginalPositions = new Map<string, { x: number; y: number }>();
    
    // Debug: Log branch positions
    console.log("[GraphCanvas] === Branch Position Debug ===");
    console.log(`[GraphCanvas] Total branches: ${currentGraphSession.branches.length}`);
    console.log(`[GraphCanvas] Branch IDs in order:`, currentGraphSession.branches.map(b => b.id).join(", "));
    
    // Get sorted non-main branches for debugging
    const getBranchNumber = (id: string): number => {
      const match = id.match(/^B(\d+)$/);
      return match ? parseInt(match[1], 10) : 0;
    };
    const nonMainBranches = currentGraphSession.branches
      .filter((b) => b.id !== "B0")
      .sort((a, b) => getBranchNumber(a.id) - getBranchNumber(b.id));
    console.log(`[GraphCanvas] Sorted non-main branches:`, nonMainBranches.map(b => b.id).join(", "));
    
    currentGraphSession.branches.forEach((branch) => {
      const rowIndex = getBranchRowIndexLocal(branch.id);
      const yPos = GRID_START_Y + rowIndex * GRID_CELL_HEIGHT;
      console.log(`[GraphCanvas] Branch ${branch.id}: rowIndex=${rowIndex}, y=${yPos}, nodes=${branch.nodes.length}`);
    });
    
    // Helper to find the prompt node for a given node by tracing back through edges
    const findPromptNodeId = (nodeId: string): string | null => {
      let currentId: string | null = nodeId;
      const visited = new Set<string>();
      
      while (currentId && !visited.has(currentId)) {
        visited.add(currentId);
        const currentNode = currentGraphSession.nodes.find((n) => n.id === currentId);
        if (currentNode?.type === "prompt") {
          return currentId;
        }
        // Find parent via incoming edge
        const incomingEdge = currentGraphSession.edges.find((e) => e.target === currentId);
        currentId = incomingEdge ? incomingEdge.source : null;
      }
      
      return null;
    };
    
    const nodeList = currentGraphSession.nodes.map((node) => {
      const isInBranch = selectedBranchNodeIds.has(node.id);
      const isSelected = selectedNodeId === node.id;
      const isMergeTarget = node.id === potentialMergeTargetId;
      const isRightmost = node.id === rightmostBranchNodeId;
      
      // Find the prompt node for this node (to check if it has been moved)
      const promptNodeId = node.type === "prompt" ? node.id : findPromptNodeId(node.id);
      const promptOffset = promptNodeId ? promptOffsetRef.current.get(promptNodeId) : null;
      const hasPromptBeenMoved = !!promptOffset;
      
      // Calculate position - check if node has been manually moved first
      let fixedPosition = node.position;
      const hasBeenMoved = movedNodesRef.current.has(node.id);
      
      if (hasBeenMoved) {
        // Use the stored position from the store (which was updated when dragged)
        fixedPosition = node.position;
      } else if (node.type === "image") {
        const step = node.data?.step ?? 0;
        const branchId = getNodeBranchId(node.id) || "B0";
        const rowIndex = getBranchRowIndexLocal(branchId);
        let basePosition = calculateGridPosition(step, rowIndex);
        // Apply prompt offset if the prompt has been moved
        if (hasPromptBeenMoved && promptOffset) {
          basePosition = {
            x: basePosition.x + promptOffset.deltaX,
            y: basePosition.y + promptOffset.deltaY,
          };
        }
        fixedPosition = basePosition;
      } else if (node.type === "prompt") {
        // Prompt node at column -1 (before step 0)
        // Use rowIndex from node data for parallel prompt nodes
        const promptRowIndex = node.data?.rowIndex ?? 0;
        let basePosition = { 
          x: GRID_START_X - GRID_CELL_WIDTH, 
          y: GRID_START_Y + promptRowIndex * GRID_CELL_HEIGHT 
        };
        // Apply prompt offset if this prompt has been moved
        if (hasPromptBeenMoved && promptOffset) {
          basePosition = {
            x: basePosition.x + promptOffset.deltaX,
            y: basePosition.y + promptOffset.deltaY,
          };
        }
        fixedPosition = basePosition;
      } else if (node.type === "loading") {
        // Loading nodes use the same positioning as image nodes
        const step = node.data?.step ?? 0;
        const branchId = getNodeBranchId(node.id) || "B0";
        const rowIndex = getBranchRowIndexLocal(branchId);
        let basePosition = calculateGridPosition(step, rowIndex);
        // Apply prompt offset if the prompt has been moved
        if (hasPromptBeenMoved && promptOffset) {
          basePosition = {
            x: basePosition.x + promptOffset.deltaX,
            y: basePosition.y + promptOffset.deltaY,
          };
        }
        fixedPosition = basePosition;
      }
      
      // Store for snap-back (only for non-moved nodes that don't have a moved prompt)
      if (!hasBeenMoved && !hasPromptBeenMoved) {
        newOriginalPositions.set(node.id, fixedPosition);
      }
      
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
    
    // Debug: Log node positions
    console.log("[GraphCanvas] === Node Position Debug ===");
    nodeList.forEach((node) => {
      if (node.type === "image") {
        const branchId = getNodeBranchId(node.id) || "B0";
        const rowIndex = getBranchRowIndexLocal(branchId);
        const expectedY = GRID_START_Y + rowIndex * GRID_CELL_HEIGHT;
        console.log(
          `[GraphCanvas] Node ${node.id}: step=${node.data?.step}, branch=${branchId}, ` +
          `rowIndex=${rowIndex}, position=(${node.position.x.toFixed(0)}, ${node.position.y.toFixed(0)}), ` +
          `expectedY=${expectedY.toFixed(0)}, diff=${Math.abs(node.position.y - expectedY).toFixed(0)}`
        );
      }
    });
    
    return nodeList;
  }, [
    currentGraphSession,
    selectedNodeId,
    potentialMergeTargetId,
    selectedBranchNodeIds,
    selectedBranchColor,
    rightmostBranchNodeId,
    getNodeBranchId,
    getBranchRowIndexLocal,
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
      const isBranchEdge = edge.type === "branch";

      // Get feedback from branch if edge doesn't have it and it's a branch edge
      let edgeFeedback = edge.data?.feedback;
      if (!edgeFeedback && isBranchEdge && branchId !== "B0") {
        const branch = currentGraphSession.branches.find((b) => b.id === branchId);
        if (branch && branch.feedback && branch.feedback.length > 0) {
          edgeFeedback = branch.feedback;
        }
      }

      // Merge edges get a special green color and dashed style
      const mergeColor = "#10b981"; // Emerald green for merge edges

      // Use selected branch color for branch highlighting
      const branchHighlightColor = selectedBranchColor || branchColor;

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edge.type || "default",
        data: {
          ...edge.data,
          feedback: edgeFeedback, // Include feedback in edge data
        },
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

      // Placeholder node는 자유롭게 움직일 수 있도록 위치 업데이트
      const graphNode = currentGraphSession.nodes.find((n) => n.id === node.id);
      if (graphNode?.type === "placeholder") {
        // 새로운 위치 저장
        updateNodePosition(currentGraphSession.id, node.id, node.position);
        originalPositionsRef.current.set(node.id, node.position);
        return;
      }

      // Prompt node - save new position for prompt and all children
      if (graphNode?.type === "prompt" && draggingPromptRef.current?.nodeId === node.id) {
        const { startPosition, childNodeIds, childStartPositions } = draggingPromptRef.current;
        
        // Calculate delta from start position
        const deltaX = node.position.x - startPosition.x;
        const deltaY = node.position.y - startPosition.y;
        
        // Get or accumulate the total offset for this prompt
        const existingOffset = promptOffsetRef.current.get(node.id) || { deltaX: 0, deltaY: 0 };
        const totalOffset = {
          deltaX: existingOffset.deltaX + deltaX,
          deltaY: existingOffset.deltaY + deltaY,
        };
        promptOffsetRef.current.set(node.id, totalOffset);
        
        // Update prompt node position in store and mark as moved
        updateNodePosition(currentGraphSession.id, node.id, node.position);
        originalPositionsRef.current.set(node.id, node.position);
        movedNodesRef.current.add(node.id);
        
        // Update all child node positions in store and mark as moved
        for (const childId of childNodeIds) {
          const childStart = childStartPositions.get(childId);
          if (childStart) {
            const newChildPos = {
              x: childStart.x + deltaX,
              y: childStart.y + deltaY,
            };
            updateNodePosition(currentGraphSession.id, childId, newChildPos);
            originalPositionsRef.current.set(childId, newChildPos);
            movedNodesRef.current.add(childId);
          }
        }
        
        console.log(`[GraphCanvas] Prompt node ${node.id} moved by (${deltaX}, ${deltaY}), total offset: (${totalOffset.deltaX}, ${totalOffset.deltaY}), updated ${childNodeIds.length} children`);
        
        // Clear the dragging ref
        draggingPromptRef.current = null;
        return;
      }

      // Check for merge target (이미지 노드인 경우에만)
      const mergeTarget = findMergeTarget(node);
      if (mergeTarget) {
        // Show merge confirmation
        setMergeSourceNode(node);
        setMergeTargetNode(mergeTarget);
        setMergeConfirmVisible(true);
      }

      // 이미지 노드는 항상 원래 위치로 되돌림 (grid-aligned)
      const originalPos = originalPositionsRef.current.get(node.id);
      if (originalPos) {
        setNodes((nds) =>
          nds.map((n) =>
            n.id === node.id ? { ...n, position: originalPos } : n
          )
        );
      }
      
      // Clear dragging ref if it was set (safety cleanup)
      draggingPromptRef.current = null;
    },
    [currentGraphSession, findMergeTarget, setNodes, updateNodePosition]
  );

  // Handle merge confirmation
  const handleMergeConfirm = useCallback(async () => {
    if (!mergeSourceNode || !mergeTargetNode || !currentGraphSession) return;

    // Get the correct backend session ID for the source node (both nodes should be from the same session for merge)
    const { getBackendSessionForNode } = useImageStore.getState();
    const sourceNodeSession = getBackendSessionForNode(mergeSourceNode.id);
    const targetNodeSession = getBackendSessionForNode(mergeTargetNode.id);
    
    // Verify both nodes are from the same session (merging across sessions is not supported)
    if (sourceNodeSession?.sessionId !== targetNodeSession?.sessionId) {
      console.error("[GraphCanvas] Cannot merge: nodes are from different sessions");
      alert("Cannot merge nodes from different sessions");
      setMergeConfirmVisible(false);
      return;
    }
    
    const sessionId = sourceNodeSession?.sessionId || backendSessionId || currentGraphSession.id;
    
    // Get unique branch IDs for frontend operations
    const sourceUniqueBranchId = getNodeBranchId(mergeSourceNode.id);
    const targetUniqueBranchId = getNodeBranchId(mergeTargetNode.id);
    
    // Extract backend branch IDs for API call
    const sourceBackendBranchId = sourceUniqueBranchId ? extractBackendBranchId(sourceUniqueBranchId) : null;
    const targetBackendBranchId = targetUniqueBranchId ? extractBackendBranchId(targetUniqueBranchId) : null;
    
    const sourceStep = mergeSourceNode.data?.step;
    const targetStep = mergeTargetNode.data?.step;

    if (!sourceBackendBranchId || !targetBackendBranchId || sourceStep === undefined || targetStep === undefined) {
      console.error("[GraphCanvas] Cannot merge: missing branch ID or step");
      setMergeConfirmVisible(false);
      return;
    }

    console.log(
      `[GraphCanvas] Merging branches: session=${sessionId}, unique=${sourceUniqueBranchId}@${sourceStep} + ${targetUniqueBranchId}@${targetStep}, backend=${sourceBackendBranchId} + ${targetBackendBranchId}`
    );

    setIsMerging(true);
    
    // Use the graph session ID for graph operations
    const graphSessionId = currentGraphSession.id;
    
    // Calculate merge start step (before API call, for loading node)
    const mergeStartStep = Math.max(sourceStep, targetStep);
    
    // Calculate row index for the new branch
    const nonMainBranches = currentGraphSession.branches.filter((b) => b.id !== "B0");
    const newBranchRowIndex = nonMainBranches.length + 1; // +1 because main branch is row 0
    
    // Calculate position for loading/merged node
    let mergeNodePosition: { x: number; y: number };
    if (mergePlaceholderNodeId) {
      const placeholderNode = currentGraphSession.nodes.find(
        (n) => n.id === mergePlaceholderNodeId
      );
      if (placeholderNode) {
        mergeNodePosition = placeholderNode.position;
        console.log(`[GraphCanvas] Using placeholder node position:`, mergeNodePosition);
      } else {
        mergeNodePosition = calculateGridPosition(mergeStartStep, newBranchRowIndex);
      }
    } else {
      mergeNodePosition = calculateGridPosition(mergeStartStep, newBranchRowIndex);
    }
    
    // Add loading node BEFORE API call to show progress
    // We'll create a temporary branch ID for the loading node
    const tempBranchId = `merge_pending_${Date.now()}`;
    const { addLoadingNode, removeLoadingNode } = useImageStore.getState();
    
    // Find PARENT nodes of the source nodes (for edge connection)
    const sourceEdge1 = currentGraphSession.edges.find((e) => e.target === mergeSourceNode.id);
    const parentNodeId1 = sourceEdge1 ? sourceEdge1.source : mergeSourceNode.id;
    
    const loadingNodeId = addLoadingNode(
      graphSessionId,
      parentNodeId1, // Connect to parent of first source
      mergeStartStep,
      mergeNodePosition,
      tempBranchId
    );
    console.log(`[GraphCanvas] Added loading node for merge: ${loadingNodeId}`);
    
    try {
      // Use backend branch IDs for API call
      const result = await mergeBranches({
        session_id: sessionId,
        branch_id_1: sourceBackendBranchId,
        branch_id_2: targetBackendBranchId,
        step_index_1: sourceStep,
        step_index_2: targetStep,
        merge_weight: 0.5,
      });

      console.log("[GraphCanvas] Merge result:", result);
      
      // Remove loading node before creating the actual merged node
      removeLoadingNode(graphSessionId, loadingNodeId);

      if (result.new_branch_id) {
        // Update store with new branch info
        const { setBackendSessionMeta, createMergedBranchWithNode } =
          useImageStore.getState();
        
        // Get the actual merge start step from the response (may differ from our estimate)
        const actualMergeStartStep = result.merge_steps?.start_step ?? mergeStartStep;
        
        // Recalculate position if step changed
        let finalMergePosition = mergeNodePosition;
        if (actualMergeStartStep !== mergeStartStep && !mergePlaceholderNodeId) {
          finalMergePosition = calculateGridPosition(actualMergeStartStep, newBranchRowIndex);
        }
        
        console.log(`[GraphCanvas] Creating merged branch: ${result.new_branch_id} in session ${graphSessionId}`);
        console.log(`[GraphCanvas] Source nodes: ${mergeSourceNode.id}@${sourceStep}, ${mergeTargetNode.id}@${targetStep}`);
        console.log(`[GraphCanvas] Merged branch starts at step ${actualMergeStartStep}`);
        
        // Update active branch in backend session meta
        setBackendSessionMeta(sessionId, result.new_branch_id);

        // Use a placeholder image or the target node's image for the initial merged node
        // The actual merged preview will be generated on the next step
        const initialImageUrl = mergeTargetNode.data?.imageUrl || mergeSourceNode.data?.imageUrl || "";
        
        console.log(`[GraphCanvas] Adding merged branch node at step ${actualMergeStartStep}, image length: ${initialImageUrl?.length || 0}`);
        
        // Create the merged branch and its initial node atomically
        // Connect to BOTH source nodes to visually show the merge
        // Placeholder node가 있으면 그 node를 변환
        const newNodeId = createMergedBranchWithNode(
          graphSessionId,
          result.new_branch_id, // Backend branch ID (e.g., "B3")
          sessionId, // Backend session ID
          mergeSourceNode.id, // First source node
          mergeTargetNode.id, // Second source node
          initialImageUrl,
          actualMergeStartStep,
          finalMergePosition,
          mergePlaceholderNodeId || undefined // Placeholder node ID 전달
        );

        console.log(`[GraphCanvas] Created merged branch node: ${newNodeId} at position:`, finalMergePosition);

        // Remove edges that were connected to placeholder node (user-drawn edges)
        // These are the temporary edges created when user connected two image nodes to placeholder
        // Only remove edges that have source nodes as the merge source nodes (user-drawn edges)
        // NOT the edges from parent nodes (which are the proper merge edges)
        if (mergePlaceholderNodeId) {
          const finalState = useImageStore.getState();
          const updatedSession = finalState.currentGraphSession;
          if (updatedSession) {
            // Find edges that target the placeholder node AND have source as one of the merge source nodes
            // These are the user-drawn edges, not the proper merge edges from parent nodes
            const edgesToRemove = updatedSession.edges.filter(
              (e) => e.target === mergePlaceholderNodeId &&
                     (e.source === mergeSourceNode.id || e.source === mergeTargetNode.id)
            );
            
            if (edgesToRemove.length > 0) {
              console.log(`[GraphCanvas] Removing ${edgesToRemove.length} user-drawn edges to placeholder node`);
              const { currentGraphSession } = useImageStore.getState();
              if (currentGraphSession) {
                useImageStore.setState({
                  currentGraphSession: {
                    ...currentGraphSession,
                    edges: currentGraphSession.edges.filter(
                      (e) => !edgesToRemove.some((er) => er.id === e.id)
                    ),
                  },
                });
              }
            }
          }
        }

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
      // Remove loading node on error
      if (loadingNodeId) {
        const { removeLoadingNode: removeLoadingNodeCleanup } = useImageStore.getState();
        removeLoadingNodeCleanup(graphSessionId, loadingNodeId);
        console.log(`[GraphCanvas] Removed loading node on error: ${loadingNodeId}`);
      }
      alert("브랜치 병합에 실패했습니다.");
    } finally {
      setIsMerging(false);
      setMergeConfirmVisible(false);
      setMergeSourceNode(null);
      setMergeTargetNode(null);
      setMergePlaceholderNodeId(null);
    }
  }, [
    mergeSourceNode,
    mergeTargetNode,
    mergePlaceholderNodeId,
    currentGraphSession,
    backendSessionId,
    getNodeBranchId,
    calculateGridPosition,
    getBranchRowIndexLocal,
  ]);

  // Handle merge cancellation
  const handleMergeCancel = useCallback(() => {
    // Restore node to original position (or just close modal)
    setMergeConfirmVisible(false);
    setMergeSourceNode(null);
    setMergeTargetNode(null);
    setMergePlaceholderNodeId(null);
  }, []);

  // Handle node drag to show visual feedback for potential merge and move prompt children
  const onNodeDrag = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      // Check for merge target (for image nodes)
      const mergeTarget = findMergeTarget(node);
      setPotentialMergeTargetId(mergeTarget?.id || null);
      
      // If dragging a prompt node, move all children together
      if (node.type === "prompt" && draggingPromptRef.current?.nodeId === node.id) {
        const { startPosition, childNodeIds, childStartPositions } = draggingPromptRef.current;
        
        // Calculate delta from start position
        const deltaX = node.position.x - startPosition.x;
        const deltaY = node.position.y - startPosition.y;
        
        // Update all child node positions
        setNodes((nds) =>
          nds.map((n) => {
            if (childNodeIds.includes(n.id)) {
              const childStart = childStartPositions.get(n.id);
              if (childStart) {
                return {
                  ...n,
                  position: {
                    x: childStart.x + deltaX,
                    y: childStart.y + deltaY,
                  },
                };
              }
            }
            return n;
          })
        );
      }
    },
    [findMergeTarget, setNodes]
  );

  // Clear potential merge target when drag starts and track prompt node dragging
  const onNodeDragStart = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      setPotentialMergeTargetId(null);
      
      // Check if dragging a prompt node - if so, track it for moving children together
      if (node.type === "prompt" && currentGraphSession) {
        const childNodeIds = getAllDescendants(node.id);
        const childStartPositions = new Map<string, { x: number; y: number }>();
        
        // Get the current offset for this prompt (if any)
        const currentOffset = promptOffsetRef.current.get(node.id) || { deltaX: 0, deltaY: 0 };
        
        // Store starting positions of all child nodes
        // Calculate positions with the current offset applied
        for (const childId of childNodeIds) {
          const childNode = currentGraphSession.nodes.find(n => n.id === childId);
          if (!childNode) continue;
          
          // Calculate the base grid position for this child
          let basePosition: { x: number; y: number };
          if (childNode.type === "image" || childNode.type === "loading") {
            const step = childNode.data?.step ?? 0;
            const branchId = getNodeBranchId(childId) || "B0";
            const rowIndex = getBranchRowIndexLocal(branchId);
            basePosition = calculateGridPosition(step, rowIndex);
          } else {
            // For other node types, use their stored position
            basePosition = childNode.position;
          }
          
          // Apply the current offset
          const positionWithOffset = {
            x: basePosition.x + currentOffset.deltaX,
            y: basePosition.y + currentOffset.deltaY,
          };
          
          childStartPositions.set(childId, positionWithOffset);
        }
        
        draggingPromptRef.current = {
          nodeId: node.id,
          startPosition: { ...node.position },
          childNodeIds,
          childStartPositions,
        };
        
        console.log(`[GraphCanvas] Started dragging prompt node ${node.id} with ${childNodeIds.length} children`);
      }
    },
    [getAllDescendants, currentGraphSession, getNodeBranchId, getBranchRowIndexLocal, calculateGridPosition]
  );

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  const onConnect = useCallback(
    (connection: { source: string | null; target: string | null }) => {
      if (!currentGraphSession || !connection.source || !connection.target) {
        return;
      }

      const sourceNode = currentGraphSession.nodes.find((n) => n.id === connection.source);
      const targetNode = currentGraphSession.nodes.find((n) => n.id === connection.target);

      if (!sourceNode || !targetNode) {
        return;
      }

      // Placeholder node로의 연결인지 확인
      if (targetNode.type === "placeholder" && sourceNode.type === "image") {
        // Edge 추가
        addEdge(currentGraphSession.id, connection.source, connection.target, {
          isMergeEdge: true,
        });

        // Edge 추가 후 업데이트된 세션에서 확인
        setTimeout(() => {
          const updatedSession = useImageStore.getState().currentGraphSession;
          if (!updatedSession) return;

          // Placeholder node로 연결된 edge 개수 확인 (새로 추가한 edge 포함)
          const edgesToPlaceholder = updatedSession.edges.filter(
            (e) => e.target === connection.target
          );
          
          // 2개의 이미지 node가 연결되었을 때만 merge 실행
          if (edgesToPlaceholder.length >= 2) {
            // 모든 source node 찾기
            const sourceNodes = edgesToPlaceholder
              .map((e) => updatedSession.nodes.find((n) => n.id === e.source))
              .filter((n): n is GraphNode => n !== undefined && n.type === "image");

            // 2개의 이미지 node가 연결되었을 때만 merge 실행
            if (sourceNodes.length === 2) {
              const [sourceNode1, sourceNode2] = sourceNodes;
              setMergeSourceNode(sourceNode1 as Node);
              setMergeTargetNode(sourceNode2 as Node);
              setMergePlaceholderNodeId(connection.target); // Placeholder node ID 저장
              setMergeConfirmVisible(true);
            }
          }
        }, 0);
      }
    },
    [currentGraphSession, addEdge]
  );

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

  // Helper to get target unique branch ID from selected node
  // Returns the uniqueBranchId for proper filtering across parallel sessions
  // IMPORTANT: This hook must be before any early returns to follow Rules of Hooks
  const getTargetBranchFromSelectedNode = useCallback(() => {
    const gs = useImageStore.getState().currentGraphSession;
    
    if (selectedNodeId && gs) {
      const selectedNode = gs.nodes.find((n) => n.id === selectedNodeId);
      
      // Priority: uniqueBranchId > edge branchId > construct from backend IDs
      if (selectedNode?.data?.uniqueBranchId) {
        return selectedNode.data.uniqueBranchId as string;
      }
      
      const incoming = gs.edges.filter((e) => e.target === selectedNodeId);
      const incomingBranch = incoming.find((e) => e.type === "branch");
      if (incomingBranch?.data?.branchId) {
        return incomingBranch.data.branchId as string;
      }
      
      // Construct unique ID from backend IDs if available
      if (selectedNode?.data?.backendBranchId && selectedNode?.data?.backendSessionId) {
        return createUniqueBranchId(selectedNode.data.backendSessionId as string, selectedNode.data.backendBranchId as string);
      }
      
      // Fallback to backend branch ID
      if (selectedNode?.data?.backendBranchId) {
        return selectedNode.data.backendBranchId as string;
      }
    }
    
    // Default fallback - try to get from active branch (this is backend branch ID)
    const activeBranch = useImageStore.getState().backendActiveBranchId;
    return activeBranch || "B0";
  }, [selectedNodeId]);

  // Handle Next Step click - runs stepInterval steps and shows the final preview
  // IMPORTANT: This hook must be before any early returns to follow Rules of Hooks
  const handleNextStep = useCallback(async () => {
    try {
      if (!currentGraphSession || !selectedNodeId) return;
      const { backendSessionId, addImageNode, addLoadingNode, removeLoadingNode, getBackendSessionForNode } =
        useImageStore.getState();
      
      // Get the correct backend session for this node (may be from a parallel session)
      const nodeSession = getBackendSessionForNode(selectedNodeId);
      const sessionId = nodeSession?.sessionId || backendSessionId || currentGraphSession.id;
      
      // Get unique branch ID for frontend filtering
      const uniqueBranchId = getTargetBranchFromSelectedNode();
      // Extract backend branch ID for API calls (use imported function)
      const backendBranchId = extractBackendBranchId(uniqueBranchId);
      // Check if this is a main branch (B0)
      const isMainBranch = backendBranchId === "B0";
      
      console.log(`[GraphCanvas] Next Step: selectedNodeId=${selectedNodeId}, uniqueBranchId=${uniqueBranchId}, backendBranchId=${backendBranchId}, stepInterval=${stepInterval}`);
      setIsStepping(true);
      
      // Get the last step in the branch (not the selected node's step)
      // This ensures loading node is created based on the branch's actual last step
      let lastStepInBranch = 0;
      let lastNodeInBranch: GraphNode | null = null;
      
      if (isMainBranch) {
        // Main branch - find the last main branch node for this session
        // Filter by uniqueBranchId to get nodes only from this session's main branch
        const mainImageNodes = (currentGraphSession.nodes || []).filter((n) => {
          if (n.type !== "image") return false;
          const nodeBranchId = getNodeBranchId(n.id);
          return nodeBranchId === uniqueBranchId;
        });
        const lastMain = mainImageNodes
          .slice()
          .sort((a, b) => (a.data?.step || 0) - (b.data?.step || 0))
          .pop();
        if (lastMain) {
          lastStepInBranch = lastMain.data?.step || 0;
          lastNodeInBranch = lastMain;
        }
      } else {
        // Non-main branch - find the last node in this branch
        const branchNodes = currentGraphSession.nodes.filter((n) => {
          if (n.type !== "image") return false;
          const nodeBranchId = getNodeBranchId(n.id);
          return nodeBranchId === uniqueBranchId;
        });
        const lastBranchNode = branchNodes
          .slice()
          .sort((a, b) => (a.data?.step || 0) - (b.data?.step || 0))
          .pop();
        if (lastBranchNode) {
          lastStepInBranch = lastBranchNode.data?.step || 0;
          lastNodeInBranch = lastBranchNode;
        }
      }
      
      const nextStep = lastStepInBranch + stepInterval;
      
      // Add loading node before starting (based on branch's last step)
      // Use graph session ID for loading nodes
      const graphSessionId = currentGraphSession.id;
      let loadingNodeId: string | null = null;
      
      // Find the prompt node for this session (needed for offset calculation)
      const gs = useImageStore.getState().currentGraphSession;
      const promptNode = gs?.nodes.find((n) => 
        n.type === "prompt" && n.data?.backendSessionId === sessionId
      ) || gs?.nodes.find((n) => n.type === "prompt");
      const promptNodeId = promptNode?.id || null;
      
      if (nextStep < 50) { // Only show loading if not at max step
        const parentNodeId = lastNodeInBranch?.id || promptNodeId || null;
        if (parentNodeId) {
          // Get row index - for non-main branches, use the branch's row index
          const rowIndex = isMainBranch 
            ? (promptNode?.data?.rowIndex ?? 0)
            : getBranchRowIndexLocal(uniqueBranchId);
          const pos = calculatePositionWithOffset(nextStep, rowIndex, promptNodeId);
          loadingNodeId = addLoadingNode(graphSessionId, parentNodeId, nextStep, pos, uniqueBranchId);
        }
      }
      
      // Run stepInterval steps, only showing the last preview
      let lastResp: Awaited<ReturnType<typeof stepOnce>> | null = null;
      
      for (let i = 0; i < stepInterval; i++) {
        const resp = await stepOnce({
          session_id: sessionId, // Use backend session ID for API calls
          branch_id: backendBranchId, // Use backend branch ID for API calls
        });
        lastResp = resp;
        
        // Check if we've reached the end
        if (resp.i >= resp.num_steps) {
          console.log(`[GraphCanvas] Reached end at step ${resp.i}/${resp.num_steps}`);
          // Remove loading node if we reached the end
          if (loadingNodeId) {
            removeLoadingNode(graphSessionId, loadingNodeId);
          }
          break;
        }
      }
      
      // Only add preview for the last step
      if (lastResp?.preview_png_base64) {
        const gsAfterStep = useImageStore.getState().currentGraphSession;
        // Use graph session ID for adding nodes (not backend session ID)
        const currentGsId = gsAfterStep?.id || currentGraphSession.id;
        
        // Find the prompt node for this session (needed for offset calculation)
        const promptNodeAfter = gsAfterStep?.nodes.find((n) => 
          n.type === "prompt" && n.data?.backendSessionId === sessionId
        ) || gsAfterStep?.nodes.find((n) => n.type === "prompt");
        const promptNodeIdAfter = promptNodeAfter?.id || null;
        
        // Find the last node in this branch
        const branchImageNodes = (gsAfterStep?.nodes || []).filter((n) => {
          if (n.type !== "image") return false;
          const nodeBranchId = getNodeBranchId(n.id);
          return nodeBranchId === uniqueBranchId;
        });
        const lastBranchNode = branchImageNodes
          .slice()
          .sort((a, b) => (a.data?.step || 0) - (b.data?.step || 0))
          .pop();
        const parentNodeId = lastBranchNode?.id || promptNodeIdAfter || null;
        
        if (parentNodeId) {
          // Get row index - for non-main branches, use the branch's row index
          const rowIndex = isMainBranch 
            ? (promptNodeAfter?.data?.rowIndex ?? 0)
            : getBranchRowIndexLocal(uniqueBranchId);
          // Position calculated with offset if prompt has been moved
          const pos = calculatePositionWithOffset(lastResp.i, rowIndex, promptNodeIdAfter);
          // Pass uniqueBranchId explicitly to ensure correct branch association
          addImageNode(currentGsId, parentNodeId, lastResp.preview_png_base64, lastResp.i, pos, undefined, uniqueBranchId);
        }
      } else if (loadingNodeId) {
        // Remove loading node if no preview was generated
        const gs = useImageStore.getState().currentGraphSession;
        const currentGsId = gs?.id || currentGraphSession.id;
        removeLoadingNode(currentGsId, loadingNodeId);
      }
    } catch (e) {
      console.error("[GraphCanvas] Next Step failed:", e);
    } finally {
      setIsStepping(false);
    }
  }, [currentGraphSession, selectedNodeId, getTargetBranchFromSelectedNode, stepInterval, calculatePositionWithOffset, getNodeBranchId, getBranchRowIndexLocal]);

  // Handle Run to End click - runs step by step showing each preview based on stepInterval
  // IMPORTANT: This hook must be before any early returns to follow Rules of Hooks
  const handleRunToEnd = useCallback(async () => {
    try {
      if (!currentGraphSession || !selectedNodeId) return;
      const { backendSessionId, addLoadingNode, removeLoadingNode, getBackendSessionForNode } =
        useImageStore.getState();
      
      // Get the correct backend session for this node (may be from a parallel session)
      const nodeSession = getBackendSessionForNode(selectedNodeId);
      const backendSession = nodeSession?.sessionId || backendSessionId || currentGraphSession.id;
      
      // Get unique branch ID for frontend filtering
      const uniqueBranchId = getTargetBranchFromSelectedNode();
      // Extract backend branch ID for API calls
      const backendBranchId = extractBackendBranchId(uniqueBranchId);
      // Check if this is a main branch (B0)
      const isMainBranch = backendBranchId === "B0";
      
      // Use graph session ID for node operations
      const graphSessionId = currentGraphSession.id;
      
      console.log(`[GraphCanvas] Run to End: selectedNodeId=${selectedNodeId}, uniqueBranchId=${uniqueBranchId}, backendBranchId=${backendBranchId}, stepInterval=${stepInterval}, backendSession=${backendSession}, graphSession=${graphSessionId}`);
      setIsRunningToEnd(true);
      isPausedRef.current = false;
      setIsPaused(false);
      
      // Run step by step until completion, showing preview based on stepInterval
      let stepCount = 0;
      const maxSteps = 100; // Safety limit
      let currentLoadingNodeId: string | null = null;
      
      while (stepCount < maxSteps) {
        // Check if paused
        if (isPausedRef.current) {
          console.log(`[GraphCanvas] Run to End paused at step ${stepCount}`);
          if (currentLoadingNodeId) {
            removeLoadingNode(graphSessionId, currentLoadingNodeId);
            currentLoadingNodeId = null;
          }
          break;
        }
        
        // Get fresh state for each step
        const { addImageNode: addImageNodeFresh, currentGraphSession: gs } =
          useImageStore.getState();
        
        // Helper to get node unique branch ID (needed for loading node logic)
        const getNodeBranchIdLocal = (nodeId: string): string | null => {
          if (!gs) return null;
          const node = gs.nodes.find((n) => n.id === nodeId);
          if (node?.data?.uniqueBranchId) {
            return node.data.uniqueBranchId as string;
          }
          const incoming = gs.edges.find((e) => e.target === nodeId);
          if (incoming?.data?.branchId) {
            return incoming.data.branchId as string;
          }
          // Legacy fallback
          if (node?.data?.backendBranchId && node?.data?.backendSessionId) {
            return createUniqueBranchId(node.data.backendSessionId as string, node.data.backendBranchId as string);
          }
          return node?.data?.backendBranchId as string || null;
        };
        
        const resp = await stepOnce({
          session_id: backendSession, // Use backend session ID for API calls
          branch_id: backendBranchId, // Use backend branch ID for API calls
        });
        
        console.log(`[GraphCanvas] Run to End step ${resp.i}/${resp.num_steps}`);
        
        stepCount++;
        
        // Only add preview to graph based on stepInterval
        // Show if: step is divisible by interval, or it's the last step
        const isLastStep = resp.status === "done" || resp.i >= resp.num_steps;
        const shouldShowPreview = (resp.i % stepInterval === 0) || isLastStep;
        
        // Remove previous loading node if preview is shown
        if (shouldShowPreview && currentLoadingNodeId) {
          removeLoadingNode(graphSessionId, currentLoadingNodeId);
          currentLoadingNodeId = null;
        }
        
        // Add loading node for next step if not at end
        // Use the branch's last step (from the response) to determine next step
        if (!isLastStep && resp.i < 50) {
          const nextStep = resp.i + stepInterval;
          if (nextStep <= 50 && !currentLoadingNodeId) {
            // Find the last node in the branch after this step (filter by unique branch ID)
            const branchNodes = gs?.nodes.filter((n) => {
              if (n.type !== "image") return false;
              const nodeBranchId = getNodeBranchIdLocal(n.id);
              return nodeBranchId === uniqueBranchId;
            }) || [];
            
            const lastBranchNode = branchNodes
              .slice()
              .sort((a, b) => (a.data?.step || 0) - (b.data?.step || 0))
              .pop();
            
            // Find the prompt node for this session (needed for offset calculation)
            const promptNode = gs?.nodes.find((n) => 
              n.type === "prompt" && n.data?.backendSessionId === backendSession
            ) || gs?.nodes.find((n) => n.type === "prompt");
            const promptNodeIdLocal = promptNode?.id || null;
            const parentNodeId = lastBranchNode?.id || promptNodeIdLocal || null;
            if (parentNodeId) {
              // Get row index - for non-main branches, use the branch's row index
              const rowIndex = isMainBranch 
                ? (promptNode?.data?.rowIndex ?? 0)
                : getBranchRowIndexLocal(uniqueBranchId);
              const pos = calculatePositionWithOffset(nextStep, rowIndex, promptNodeIdLocal);
              currentLoadingNodeId = addLoadingNode(graphSessionId, parentNodeId, nextStep, pos, uniqueBranchId);
            }
          }
        } else if (isLastStep && currentLoadingNodeId) {
          removeLoadingNode(graphSessionId, currentLoadingNodeId);
          currentLoadingNodeId = null;
        }
        
        if (resp.preview_png_base64 && shouldShowPreview) {
          console.log(`[GraphCanvas] Adding preview for step ${resp.i} (interval: ${stepInterval})`);
          // Use graph session ID for adding nodes
          const currentGsId = gs?.id || graphSessionId;
          
          // Find the prompt node for this session (needed for offset calculation)
          const promptNode = gs?.nodes.find((n) => 
            n.type === "prompt" && n.data?.backendSessionId === backendSession
          ) || gs?.nodes.find((n) => n.type === "prompt");
          const promptNodeIdLocal = promptNode?.id || null;
          
          // Find the last node in this branch (filter by unique branch ID)
          const branchImageNodes = (gs?.nodes || []).filter((n) => {
            if (n.type !== "image") return false;
            const nodeBranchId = getNodeBranchIdLocal(n.id);
            return nodeBranchId === uniqueBranchId;
          });
          const lastBranchNode = branchImageNodes
            .slice()
            .sort((a, b) => (a.data?.step || 0) - (b.data?.step || 0))
            .pop();
          const parentNodeId = lastBranchNode?.id || promptNodeIdLocal || null;
          
          if (parentNodeId) {
            // Get row index - for non-main branches, use the branch's row index
            const rowIndex = isMainBranch 
              ? (promptNode?.data?.rowIndex ?? 0)
              : getBranchRowIndexLocal(uniqueBranchId);
            // Position calculated with offset if prompt has been moved
            const pos = calculatePositionWithOffset(resp.i, rowIndex, promptNodeIdLocal);
            // Pass uniqueBranchId explicitly to ensure correct branch association
            addImageNodeFresh(currentGsId, parentNodeId, resp.preview_png_base64, resp.i, pos, undefined, uniqueBranchId);
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
  }, [currentGraphSession, selectedNodeId, getTargetBranchFromSelectedNode, stepInterval, calculatePositionWithOffset, getBranchRowIndexLocal]);

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
    
    const { backendSessionId, removeNodeAndDescendants, getBackendSessionForNode } = useImageStore.getState();
    
    // Get the correct backend session for this node (may be from a parallel session)
    const nodeSession = getBackendSessionForNode(selectedNodeId);
    const sessionId = nodeSession?.sessionId || backendSessionId || currentGraphSession.id;
    
    // Get unique branch ID and extract backend branch ID
    const uniqueBranchId = getTargetBranchFromSelectedNode();
    const backendBranchId = extractBackendBranchId(uniqueBranchId);
    
    console.log(`[GraphCanvas] Backtracking: node=${selectedNodeId}, step=${step}, uniqueBranch=${uniqueBranchId}, backendBranch=${backendBranchId}`);
    setIsBacktracking(true);
    
    try {
      // Call backend to backtrack (use backend branch ID for API)
      const result = await backtrackTo({
        session_id: sessionId,
        branch_id: backendBranchId, // Use backend branch ID for API calls
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

  // Handle Add Placeholder Node button click
  const handleAddPlaceholderNode = useCallback(() => {
    if (!currentGraphSession) return;
    
    // Canvas 중앙 위치 계산 (대략적인 값)
    const centerPosition = { x: 400, y: 300 };
    
    try {
      addPlaceholderNode(
        currentGraphSession.id,
        centerPosition,
        onAddNodeClick // placeholder node 클릭 시 CompositionModal 열기
      );
      console.log("[GraphCanvas] Placeholder node 추가됨");
    } catch (error) {
      console.error("[GraphCanvas] Placeholder node 추가 실패:", error);
    }
  }, [currentGraphSession, addPlaceholderNode, onAddNodeClick]);

  // currentGraphSession이 없으면 빈 세션을 생성하므로 여기서는 항상 세션이 존재함
  // EmptyState는 더 이상 필요하지 않음

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
          {/* 새로운 node 만들기 Button */}
          <button
            onClick={handleAddPlaceholderNode}
            disabled={!currentGraphSession}
            style={{
              padding: "10px 16px",
              borderRadius: 8,
              border: "none",
              fontWeight: 700,
              cursor: !currentGraphSession ? "not-allowed" : "pointer",
              background: !currentGraphSession
                ? "linear-gradient(135deg, #4b5563 0%, #6b7280 100%)"
                : "linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%)",
              color: "#fff",
              opacity: !currentGraphSession ? 0.6 : 1,
            }}
            title="새로운 node를 canvas에 추가합니다"
          >
            + New Prompt
          </button>

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
          onMove={(event, viewport) => {
            // Dispatch custom event when viewport moves (zoom/pan)
            window.dispatchEvent(new CustomEvent('reactflow-viewport-change', { detail: { viewport } }));
          }}
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
