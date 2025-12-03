import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  type Edge,
  type Connection,
  type Node,
  type NodeTypes,
  useNodesState,
  useEdgesState,
  useReactFlow,
} from "reactflow";
import "reactflow/dist/style.css";
import styled from "styled-components";
import SimpleImageNode from "./SimpleImageNode";
import SimpleLoadingNode from "./SimpleLoadingNode";
import SimplePromptNode, { type SimplePromptNodeData } from "./SimplePromptNode";
import BookmarkPanel from "./BookmarkPanel";
import { useImageStore } from "../stores/imageStore";
import {
  generateSimpleImages,
  generateWithImage,
  type SimpleGenerateResponse,
} from "../api/simplePixArt";

const CanvasContainer = styled.div`
  width: 100%;
  height: 100vh;
  position: relative;
  overflow: hidden;
  background: linear-gradient(
    135deg,
    #1a1a2e 0%,
    #16213e 50%,
    #0f3460 100%
  );
`;

const BottomCenterControls = styled.div`
  position: absolute;
  left: 50%;
  bottom: 24px;
  transform: translateX(-50%);
  z-index: 1200;
  display: flex;
  gap: 12px;
`;

const GenerateButton = styled.button<{ disabled?: boolean }>`
  padding: 12px 28px;
  border-radius: 999px;
  border: none;
  font-size: 15px;
  font-weight: 700;
  cursor: ${(props) => (props.disabled ? "not-allowed" : "pointer")};
  background: ${(props) =>
    props.disabled
      ? "linear-gradient(135deg, #4b5563 0%, #6b7280 100%)"
      : "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)"};
  color: #fff;
  opacity: ${(props) => (props.disabled ? 0.6 : 1)};
  box-shadow: 0 6px 18px rgba(99, 102, 241, 0.4);
  transition: all 0.2s ease;

  &:hover {
    transform: ${(props) => (props.disabled ? "none" : "translateY(-2px)")};
    box-shadow: ${(props) =>
      props.disabled
        ? "0 4px 12px rgba(75, 85, 99, 0.4)"
        : "0 8px 24px rgba(99, 102, 241, 0.5)"};
  }

  &:active {
    transform: translateY(0);
  }
`;

const HelperText = styled.div`
  position: absolute;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 1100;
  padding: 10px 16px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.85);
  color: #e5e7eb;
  font-size: 12px;
  font-weight: 500;
  border: 1px solid rgba(148, 163, 184, 0.4);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
`;

type SimpleNodeData =
  | SimplePromptNodeData
  | { imageUrl?: string; step?: number; parentPromptId?: string }
  | { step?: number; parentPromptId?: string }; // loading node

const nodeTypes: NodeTypes = {
  prompt: SimplePromptNode,
  image: SimpleImageNode,
  loading: SimpleLoadingNode,
};

const SimpleGraphCanvas: React.FC = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nodes, setNodes, onNodesChange] =
    useNodesState<SimpleNodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [activePromptId, setActivePromptId] = useState<string | null>(null);
  const [nextPromptIndex, setNextPromptIndex] = useState(1);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const draggingPromptRef = useRef<{
    id: string;
    startPosition: { x: number; y: number };
    childIds: string[];
    childStartPositions: Map<string, { x: number; y: number }>;
  } | null>(null);

  const createPromptNode = useCallback(
    (id: string, position: { x: number; y: number }): Node<SimpleNodeData> => {
      const handleChangePrompt = (value: string) => {
        setNodes((prev) =>
          prev.map((node) =>
            node.id === id
              ? {
                  ...node,
                  data: {
                    ...(node.data as SimplePromptNodeData),
                    prompt: value,
                  },
                }
              : node
          )
        );
        setActivePromptId(id);
      };

      const handleUploadImage = (file: File | null) => {
        if (!file) {
          setNodes((prev) =>
            prev.map((node) =>
              node.id === id
                ? {
                    ...node,
                    data: {
                      ...(node.data as SimplePromptNodeData),
                      inputImagePreviewUrl: null,
                      inputImageDataUrl: null,
                      inputImageSourceNodeId: null,
                    },
                  }
                : node
            )
          );
          return;
        }

        const reader = new FileReader();
        reader.onload = () => {
          const result = typeof reader.result === "string" ? reader.result : null;
          if (!result) return;
          setNodes((prev) =>
            prev.map((node) =>
              node.id === id
                ? {
                    ...node,
                    data: {
                      ...(node.data as SimplePromptNodeData),
                      inputImagePreviewUrl: result,
                      inputImageDataUrl: result,
                      inputImageSourceNodeId: null,
                    },
                  }
                : node
            )
          );
        };
        reader.readAsDataURL(file);
      };

      return {
        id,
        type: "prompt",
        position,
        data: {
          prompt: "",
          onChangePrompt: handleChangePrompt,
          inputImagePreviewUrl: null,
          inputImageDataUrl: null,
          inputImageSourceNodeId: null,
          onUploadImage: handleUploadImage,
        } as SimplePromptNodeData,
      };
    },
    [setNodes]
  );

  // Initialize with a single prompt node
  useEffect(() => {
    if (nodes.length === 0) {
      const initialId = "prompt-1";
      const initialNode = createPromptNode(initialId, { x: 0, y: 0 });
      setNodes([initialNode]);
      setActivePromptId(initialId);
      setNextPromptIndex(2);
    }
  }, [nodes.length, createPromptNode]);

  const activePromptText = useMemo(() => {
    if (!activePromptId) return "";
    const node = nodes.find(
      (n) => n.id === activePromptId && n.type === "prompt"
    );
    if (!node) return "";
    return (node.data as SimplePromptNodeData).prompt ?? "";
  }, [activePromptId, nodes]);

  const handleGenerate = useCallback(async () => {
    if (!activePromptId) {
      return;
    }

    const promptNode = nodes.find(
      (n) => n.id === activePromptId && n.type === "prompt"
    );
    if (!promptNode) return;

    const promptData = promptNode.data as SimplePromptNodeData;
    const promptText = promptData.prompt ?? "";
    if (!promptText.trim()) return;

    // Determine previous prompt and optional input image (uploaded or from graph)
    let previousPromptText: string | null = null;
    let inputImageDataUrl: string | null =
      promptData.inputImageDataUrl ?? null;

    // If this prompt is connected from an image node, use that image and its parent's prompt
    const incomingImageEdge = edges.find((e) => e.target === activePromptId);
    if (incomingImageEdge) {
      const sourceNode = nodes.find((n) => n.id === incomingImageEdge.source);
      if (sourceNode?.type === "image") {
        const imageData = (sourceNode.data as {
          imageUrl?: string;
          parentPromptId?: string;
        }).imageUrl;
        if (!inputImageDataUrl && imageData) {
          inputImageDataUrl = imageData;
        }
        const parentPromptId = (sourceNode.data as {
          parentPromptId?: string;
        }).parentPromptId;
        if (parentPromptId) {
          const parentPromptNode = nodes.find(
            (n) => n.id === parentPromptId && n.type === "prompt"
          );
          if (parentPromptNode) {
            previousPromptText =
              (parentPromptNode.data as SimplePromptNodeData).prompt ?? null;
          }
        }
      }
    }

    const baseX = promptNode.position.x;
    const baseY = promptNode.position.y;
    const offsetX = 320;
    const verticalSpacing = 240; // more vertical space between branched images
    const startY = baseY - verticalSpacing * 1.5;

    const loadingNodeIds = Array.from({ length: 4 }, (_, index) => {
      return `${activePromptId}-loading-${index + 1}`;
    });

    // Place four loading nodes where the images will appear
    setNodes((prev) => {
      const filtered = prev.filter(
        (node) =>
          !(
            node.type === "loading" &&
            (node.data as { parentPromptId?: string }).parentPromptId ===
              activePromptId
          )
      );

      const loadingNodes: Node<SimpleNodeData>[] = loadingNodeIds.map(
        (id, index) => {
          const y = startY + index * verticalSpacing;
          return {
            id,
            type: "loading",
            position: { x: baseX + offsetX, y },
            data: {
              step: undefined,
              parentPromptId: activePromptId || undefined,
            },
          };
        }
      );

      return [...filtered, ...loadingNodes];
    });

    setEdges((prev) => {
      // Remove previous prompt->image edges; keep other graph edges
      const filtered = prev.filter(
        (edge) => edge.source !== activePromptId
      );
      const loadingEdges: Edge[] = loadingNodeIds.map((id, index) => ({
        id: `e-${activePromptId}-loading-${index + 1}`,
        source: activePromptId as string,
        target: id,
      }));
      return [...filtered, ...loadingEdges];
    });

    setIsGenerating(true);
    setError(null);

    const applyImages = (res: SimpleGenerateResponse) => {

      setNodes((prev) => {
        // Remove existing images for this prompt
        const filtered = prev.filter(
          (node) =>
            !(
              (node.type === "image" || node.type === "loading") &&
              (node.data as { parentPromptId?: string }).parentPromptId ===
                activePromptId
            )
        );

        const newImageNodes: Node<SimpleNodeData>[] = res.images.map(
          (b64, index) => {
            const y = startY + index * verticalSpacing;
            const id = `${activePromptId}-image-${index + 1}`;
            const imageUrl = b64
              ? `data:image/png;base64,${b64}`
              : undefined;
            return {
              id,
              type: "image",
              position: { x: baseX + offsetX, y },
              data: {
                imageUrl,
                step: index + 1,
                parentPromptId: activePromptId,
              },
            };
          }
        );

        return [...filtered, ...newImageNodes];
      });

      setEdges((prev) => {
        const filtered = prev.filter((edge) => edge.source !== activePromptId);
        const newEdges: Edge[] = res.images.map((_b64, index) => {
          const targetId = `${activePromptId}-image-${index + 1}`;
          return {
            id: `e-${activePromptId}-${index + 1}`,
            source: activePromptId,
            target: targetId,
          };
        });
        return [...filtered, ...newEdges];
      });
    };

    try {
      let res: SimpleGenerateResponse;
      if (inputImageDataUrl) {
        res = await generateWithImage({
          current_prompt: promptText.trim(),
          previous_prompt: previousPromptText,
          num_images: 4,
          imageDataUrl: inputImageDataUrl,
        });
      } else {
        res = await generateSimpleImages({
          prompt: promptText.trim(),
          previous_prompt: previousPromptText,
          num_images: 4,
        });
      }

      applyImages(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      // Remove loading nodes on error
      setNodes((prev) =>
        prev.filter(
          (n) =>
            !(
              n.type === "loading" &&
              (n.data as { parentPromptId?: string }).parentPromptId ===
                activePromptId
            )
        )
      );
      // eslint-disable-next-line no-console
      console.error("[SimpleGraphCanvas] generate failed:", e);
    } finally {
      setIsGenerating(false);
    }
  }, [activePromptId, nodes, edges, setNodes, setEdges]);

  const handleAddPrompt = useCallback(() => {
    const id = `prompt-${nextPromptIndex}`;
    const promptCount = nodes.filter((n) => n.type === "prompt").length;
    // Place input nodes horizontally across instead of vertically
    const position = { x: promptCount * 360, y: 0 };
    const newNode = createPromptNode(id, position);

    setNodes((prev) => [...prev, newNode]);
    setActivePromptId(id);
    setNextPromptIndex((prev) => prev + 1);
  }, [nextPromptIndex, nodes, createPromptNode]);

  const isGenerateDisabled =
    isGenerating || !activePromptId || !activePromptText.trim();

  const handleNodeDragStart = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      if (node.type !== "prompt") return;

      // Find all image nodes that belong to this prompt
      const childIds = nodes
        .filter(
          (n) =>
            n.type === "image" &&
            (n.data as { parentPromptId?: string }).parentPromptId === node.id
        )
        .map((n) => n.id);

      const childStartPositions = new Map<string, { x: number; y: number }>();
      for (const id of childIds) {
        const child = nodes.find((n) => n.id === id);
        if (child) {
          childStartPositions.set(id, { ...child.position });
        }
      }

      draggingPromptRef.current = {
        id: node.id,
        startPosition: { ...node.position },
        childIds,
        childStartPositions,
      };
    },
    [nodes]
  );

  const handleNodeDrag = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      const dragging = draggingPromptRef.current;
      if (!dragging || dragging.id !== node.id) return;

      const { startPosition, childIds, childStartPositions } = dragging;
      const dx = node.position.x - startPosition.x;
      const dy = node.position.y - startPosition.y;

      setNodes((prev) =>
        prev.map((n) => {
          if (!childIds.includes(n.id)) return n;
          const start = childStartPositions.get(n.id);
          if (!start) return n;
          return {
            ...n,
            position: {
              x: start.x + dx,
              y: start.y + dy,
            },
          };
        })
      );
    },
    [setNodes]
  );

  const handleNodeDragStop = useCallback(() => {
    draggingPromptRef.current = null;
  }, []);

  const handleConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target) return;

      const sourceNode = nodes.find((n) => n.id === connection.source);
      const targetNode = nodes.find((n) => n.id === connection.target);
      if (!sourceNode || !targetNode) return;

      // Only allow image -> prompt connections for using image as next input
      if (sourceNode.type !== "image" || targetNode.type !== "prompt") return;

      const imageUrl = (sourceNode.data as { imageUrl?: string }).imageUrl;

      setNodes((prev) =>
        prev.map((n) =>
          n.id === targetNode.id
            ? {
                ...n,
                data: {
                  ...(n.data as SimplePromptNodeData),
                  inputImagePreviewUrl:
                    imageUrl ??
                    (n.data as SimplePromptNodeData).inputImagePreviewUrl ??
                    null,
                  inputImageDataUrl:
                    imageUrl ??
                    (n.data as SimplePromptNodeData).inputImageDataUrl ??
                    null,
                  inputImageSourceNodeId: sourceNode.id,
                },
              }
            : n
        )
      );

      setEdges((prev) => {
        if (
          prev.some(
            (e) => e.source === sourceNode.id && e.target === targetNode.id
          )
        ) {
          return prev;
        }
        const newEdge: Edge = {
          id: `e-${sourceNode.id}-${targetNode.id}-${Date.now()}`,
          source: sourceNode.id,
          target: targetNode.id,
        };
        return [...prev, newEdge];
      });
    },
    [nodes, setNodes, setEdges]
  );

  const handleNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      setSelectedNodeId(node.id);
      if (node.type === "prompt") {
        setActivePromptId(node.id);
      } else {
        setActivePromptId(null);
      }
    },
    []
  );

  const handlePaneClick = useCallback(() => {
    setSelectedNodeId(null);
    setActivePromptId(null);
  }, []);

  const { highlightedNodes, highlightedEdges } = useMemo(() => {
    if (!selectedNodeId) {
      return { highlightedNodes: nodes, highlightedEdges: edges };
    }

    // Walk the ancestor chain from the selected node back to the root
    const pathNodeIds = new Set<string>();
    const pathEdgeIds = new Set<string>();

    let currentId: string | null = selectedNodeId;
    while (currentId) {
      pathNodeIds.add(currentId);
      // Find the first incoming edge (treat its source as the parent)
      const incoming = edges.find((e) => e.target === currentId);
      if (!incoming) break;
      pathEdgeIds.add(incoming.id);
      currentId = incoming.source;
      if (pathNodeIds.has(currentId)) {
        // Prevent cycles (shouldn't happen in our tree-like graph)
        break;
      }
    }

    const highlightedNodes = nodes.map((node) =>
      pathNodeIds.has(node.id)
        ? {
            ...node,
            style: {
              ...(node.style ?? {}),
              boxShadow: "0 0 0 3px rgba(99, 102, 241, 0.8)",
              borderRadius: 16,
            },
          }
        : {
            ...node,
            style: {
              ...(node.style ?? {}),
              boxShadow: undefined,
            },
          }
    );

    const highlightedEdges = edges.map((edge) =>
      pathEdgeIds.has(edge.id)
        ? {
            ...edge,
            animated: true,
            style: {
              ...(edge.style ?? {}),
              stroke: "#6366f1",
              strokeWidth: 3,
            },
          }
        : {
            ...edge,
            animated: false,
            style: {
              ...(edge.style ?? {}),
              strokeWidth: 1,
            },
          }
    );

    return { highlightedNodes, highlightedEdges };
  }, [nodes, edges, selectedNodeId]);

  return (
    <ReactFlowProvider>
      <SimpleGraphCanvasContent
        nodes={highlightedNodes}
        edges={highlightedEdges}
        nodeTypes={nodeTypes}
        onConnect={handleConnect}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeDragStart={handleNodeDragStart}
        onNodeDrag={handleNodeDrag}
        onNodeDragStop={handleNodeDragStop}
        onNodeClick={handleNodeClick}
        onPaneClick={handlePaneClick}
        isGenerating={isGenerating}
        isGenerateDisabled={isGenerateDisabled}
        error={error}
        handleGenerate={handleGenerate}
        handleAddPrompt={handleAddPrompt}
      />
    </ReactFlowProvider>
  );
};

// Inner component to use useReactFlow hook
const SimpleGraphCanvasContent: React.FC<{
  nodes: Node[];
  edges: Edge[];
  nodeTypes: NodeTypes;
  onConnect: (connection: Connection) => void;
  onNodesChange: any;
  onEdgesChange: any;
  onNodeDragStart: any;
  onNodeDrag: any;
  onNodeDragStop: any;
  onNodeClick: any;
  onPaneClick: any;
  isGenerating: boolean;
  isGenerateDisabled: boolean;
  error: string | null;
  handleGenerate: () => void;
  handleAddPrompt: () => void;
}> = ({
  nodes,
  edges,
  nodeTypes,
  onConnect,
  onNodesChange,
  onEdgesChange,
  onNodeDragStart,
  onNodeDrag,
  onNodeDragStop,
  onNodeClick,
  onPaneClick,
  isGenerating,
  isGenerateDisabled,
  error,
  handleGenerate,
  handleAddPrompt,
}) => {
  const { setCenter } = useReactFlow();
  const { selectNode } = useImageStore();

  return (
    <CanvasContainer>
      <HelperText>
        Type your prompt in any input node, optionally upload or connect an
        image, then click &ldquo;Generate&rdquo; to create 4 images branching
        to the right. Drag an image node connection into a new prompt to use
        it as the next input.
        {error ? `  •  Error: ${error}` : null}
      </HelperText>

      {/* 북마크 패널 */}
      <BookmarkPanelWrapper
        reactFlowNodes={nodes}
        selectNode={selectNode}
        nodes={nodes}
      />

        <BottomCenterControls>
          <GenerateButton onClick={handleGenerate} disabled={isGenerateDisabled}>
            {isGenerating ? "Generating..." : "Generate"}
          </GenerateButton>
          <GenerateButton onClick={handleAddPrompt} disabled={isGenerating}>
            + Add Prompt
          </GenerateButton>
        </BottomCenterControls>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        // Allow connecting image nodes into prompt nodes
        nodesConnectable
        onConnect={onConnect}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeDragStart={onNodeDragStart}
        onNodeDrag={onNodeDrag}
        onNodeDragStop={onNodeDragStop}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        zoomOnScroll
        zoomOnPinch
        panOnScroll
        panOnDrag
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </CanvasContainer>
  );
};

// 북마크 패널 래퍼 - useReactFlow를 사용하기 위해 ReactFlowProvider 내부에 있어야 함
const BookmarkPanelWrapper: React.FC<{
  reactFlowNodes: Node[];
  selectNode: (nodeId: string | null) => void;
  nodes: Node[];
}> = ({ reactFlowNodes, selectNode, nodes }) => {
  const { setCenter, getNode } = useReactFlow();
  
  return (
    <BookmarkPanel
      nodes={nodes}
      onNodeClick={(nodeId) => {
        // 노드 선택
        selectNode(nodeId);
        
        // 노드로 이동
        const node = reactFlowNodes.find((n) => n.id === nodeId);
        if (node) {
          // ReactFlow에서 노드의 실제 크기 가져오기
          const reactFlowNode = getNode(nodeId);
          let centerX = node.position.x;
          let centerY = node.position.y;
          
          // 노드의 실제 크기를 고려하여 중심점 계산
          if (reactFlowNode && reactFlowNode.width && reactFlowNode.height) {
            // 노드의 중심점 = position + (width/2, height/2)
            centerX = node.position.x + reactFlowNode.width / 2;
            centerY = node.position.y + reactFlowNode.height / 2;
          } else {
            // ReactFlow에서 크기를 가져올 수 없는 경우, 추정값 사용
            // SimpleImageNode의 경우: min-width 180px, max-width 220px, padding 8px, border 2px
            // 이미지는 aspect-ratio 1이므로 대략 180-220px 정사각형
            const estimatedWidth = 200; // 평균값
            const estimatedHeight = 200; // aspect-ratio 1
            centerX = node.position.x + estimatedWidth / 2;
            centerY = node.position.y + estimatedHeight / 2;
          }
          
          // 노드의 중심으로 뷰포트 이동
          setCenter(centerX, centerY, { zoom: 1.2, duration: 500 });
        }
      }}
    />
  );
};

export default SimpleGraphCanvas;

