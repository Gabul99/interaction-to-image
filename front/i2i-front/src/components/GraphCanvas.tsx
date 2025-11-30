import React, { useCallback, useMemo, useState } from "react";
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
import styled from "styled-components";
import { useImageStore } from "../stores/imageStore";
import { USE_MOCK_MODE } from "../config/api";
import { connectImageStream } from "../api/websocket";
import PromptNode from "./PromptNode";
import ImageNode from "./ImageNode";
import BranchingModal from "./BranchingModal";
import FeedbackEdge from "./FeedbackEdge";

const nodeTypes: NodeTypes = {
  prompt: PromptNode,
  image: ImageNode,
};

const edgeTypes: EdgeTypes = {
  branch: FeedbackEdge,
  default: FeedbackEdge,
};

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
    updateNodePosition,
  } = useImageStore();
  const [branchingModalVisible, setBranchingModalVisible] = useState(false);
  const [branchingNodeId, setBranchingNodeId] = useState<string | null>(null);

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

  // React-flow의 nodes와 edges를 store의 데이터와 동기화
  const nodes: Node[] = useMemo(() => {
    if (!currentGraphSession) return [];
    return currentGraphSession.nodes.map((node) => ({
      id: node.id,
      type: node.type,
      position: node.position,
      data: {
        ...node.data,
        onBranchClick:
          node.type === "image"
            ? () => {
                setBranchingNodeId(node.id);
                setBranchingModalVisible(true);
              }
            : undefined,
      },
      selected: selectedNodeId === node.id,
    }));
  }, [currentGraphSession, selectedNodeId]);

  // 브랜치 ID를 기반으로 색상 생성
  const getBranchColor = useCallback((branchId?: string): string => {
    if (!branchId) return "#6366f1"; // 기본 색상

    // branchId에서 숫자 추출하여 색상 결정
    const colors = [
      "#8b5cf6", // 보라색
      "#ec4899", // 핑크
      "#f43f5e", // 빨간색
      "#f59e0b", // 주황색
      "#eab308", // 노란색
      "#84cc16", // 연두색
      "#22c55e", // 초록색
      "#10b981", // 청록색
      "#14b8a6", // 청록색
      "#06b6d4", // 시안색
      "#0ea5e9", // 하늘색
      "#3b82f6", // 파란색
    ];

    // branchId의 해시값을 계산하여 색상 선택
    let hash = 0;
    for (let i = 0; i < branchId.length; i++) {
      hash = branchId.charCodeAt(i) + ((hash << 5) - hash);
    }
    const index = Math.abs(hash) % colors.length;
    return colors[index];
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

  // 경로에 포함된 edge ID들 찾기
  const getPathEdgeIds = useCallback(
    (path: string[]): Set<string> => {
      if (!currentGraphSession || path.length < 2) return new Set();

      const edgeIds = new Set<string>();
      // path는 [선택된노드, ..., root] 순서이므로
      // edge는 path[i+1] -> path[i] 방향
      for (let i = 0; i < path.length - 1; i++) {
        const source = path[i + 1]; // 부모
        const target = path[i]; // 자식
        const edge = currentGraphSession.edges.find(
          (e) => e.source === source && e.target === target
        );
        if (edge) {
          edgeIds.add(edge.id);
        }
      }

      return edgeIds;
    },
    [currentGraphSession]
  );

  // 선택된 노드의 경로 계산
  const selectedPathEdgeIds = useMemo(() => {
    if (!selectedNodeId || !currentGraphSession) return new Set<string>();
    const path = getPathToRoot(selectedNodeId);
    return getPathEdgeIds(path);
  }, [selectedNodeId, currentGraphSession, getPathToRoot, getPathEdgeIds]);

  const edges: Edge[] = useMemo(() => {
    if (!currentGraphSession) return [];
    return currentGraphSession.edges.map((edge) => {
      const branchId = edge.data?.branchId;
      const branchColor = getBranchColor(branchId);
      const isInPath = selectedPathEdgeIds.has(edge.id);

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edge.type || "default",
        data: edge.data,
        animated: edge.type === "branch",
        style: {
          stroke: isInPath
            ? "#fbbf24" // highlight 색상 (노란색)
            : branchId
            ? branchColor
            : "#6366f1", // 기본 색상
          strokeWidth: isInPath
            ? 3.5 // highlight 두께
            : edge.type === "branch"
            ? 2
            : 1.5,
          strokeDasharray: isInPath ? "none" : undefined, // highlight는 실선
        },
      };
    });
  }, [currentGraphSession, getBranchColor, selectedPathEdgeIds]);

  const [reactFlowNodes, setNodes, onNodesChange] = useNodesState(nodes);
  const [reactFlowEdges, setEdges, onEdgesChange] = useEdgesState(edges);

  // nodes와 edges가 변경되면 React-flow 상태 업데이트
  React.useEffect(() => {
    setNodes(nodes);
  }, [nodes, setNodes]);

  React.useEffect(() => {
    setEdges(edges);
  }, [edges, setEdges]);

  // 노드 드래그 종료 시 위치 업데이트
  const onNodeDragStop = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      if (!currentGraphSession) return;
      updateNodePosition(currentGraphSession.id, node.id, node.position);
    },
    [currentGraphSession, updateNodePosition]
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
      <div style={{ width: "100%", height: "100%" }} className={className}>
        <ReactFlow
          nodes={reactFlowNodes}
          edges={reactFlowEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
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
    </ReactFlowProvider>
  );
};

export default GraphCanvas;
