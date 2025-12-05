import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useImageStore } from "../stores/imageStore";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  type Edge,
  type Connection,
  type Node,
  type NodeTypes,
  type NodeChange,
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
import {
  generateSimpleImages,
  generateWithImage,
  type SimpleGenerateResponse,
  saveSession,
} from "../api/simplePixArt";
import type { GraphSession, GraphNode, GraphEdge } from "../types";
import { logActionAndSaveSession } from "../utils/logging";

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

const AddPromptButton = styled.button<{ disabled?: boolean }>`
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
      : "linear-gradient(135deg,rgb(241, 92, 55) 0%,rgb(236, 72, 75) 100%)"};
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

interface SimpleGraphCanvasProps {
  mode?: string;
  participant?: number | null;
}

// Helper functions to convert between SimpleGraphCanvas nodes/edges and GraphSession
const convertNodesToGraphSession = (
  nodes: Node<SimpleNodeData>[],
  edges: Edge[],
  sessionId: string = "simple-session"
): GraphSession => {
  const graphNodes: GraphNode[] = nodes.map((node) => ({
    id: node.id,
    type: node.type as 'prompt' | 'image' | 'placeholder' | 'loading',
    data: {
      prompt: (node.data as SimplePromptNodeData)?.prompt,
      imageUrl: (node.data as { imageUrl?: string })?.imageUrl,
      step: (node.data as { step?: number })?.step,
    },
    position: node.position,
  }));

  const graphEdges: GraphEdge[] = edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    type: edge.type as 'default' | 'branch' | undefined,
  }));

  return {
    id: sessionId,
    nodes: graphNodes,
    edges: graphEdges,
    branches: [], // SimpleGraphCanvas doesn't use branches
  };
};

const convertGraphSessionToNodes = (
  graphSession: GraphSession
): { nodes: Node<SimpleNodeData>[]; edges: Edge[] } => {
  const nodes: Node<SimpleNodeData>[] = graphSession.nodes.map((gn) => {
    const nodeData: SimpleNodeData = 
      gn.type === 'prompt'
        ? {
            prompt: gn.data?.prompt || '',
            inputImagePreviewUrls: [],
            inputImageDataUrls: [],
            inputImageSourceNodeIds: [],
            onUploadImages: () => {},
            onRemoveInputImage: () => {},
            onChangePrompt: () => {},
          }
        : gn.type === 'image'
        ? {
            imageUrl: gn.data?.imageUrl,
            step: gn.data?.step,
          }
        : {};

    return {
      id: gn.id,
      type: gn.type,
      data: nodeData,
      position: gn.position,
    };
  });

  const edges: Edge[] = graphSession.edges.map((ge) => ({
    id: ge.id,
    source: ge.source,
    target: ge.target,
    type: ge.type,
  }));

  return { nodes, edges };
};

const SimpleGraphCanvas: React.FC<SimpleGraphCanvasProps> = ({ mode, participant }) => {
  const { loadSessionFromServer, bookmarkedNodeIds } = useImageStore();
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nodes, setNodes, onNodesChangeInternal] =
    useNodesState<SimpleNodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  
  // 노드 변경 감지 및 백트래킹 로깅
  const onNodesChange = useCallback(
    async (changes: NodeChange[]) => {
      // 노드 삭제 감지 (삭제 전에 로깅해야 하므로 먼저 처리)
      const removedChanges = changes.filter((change) => change.type === "remove");
      if (removedChanges.length > 0 && participant && mode) {
        for (const change of removedChanges) {
          if (change.type !== "remove") continue;
          const removedNodeId = change.id;
          const removedNode = nodes.find((n) => n.id === removedNodeId);
          
          if (removedNode) {
            // 삭제된 노드의 모든 자식 노드 찾기
            const getAllDescendantIds = (nodeId: string): string[] => {
              const descendants: string[] = [nodeId];
              const outgoingEdges = edges.filter((e) => e.source === nodeId);
              for (const edge of outgoingEdges) {
                descendants.push(...getAllDescendantIds(edge.target));
              }
              return descendants;
            };
            const removedNodeIds = getAllDescendantIds(removedNodeId);
            
            // 노드 타입에 따라 다른 로깅
            if (removedNode.type === "image") {
              const step = (removedNode.data as { step?: number })?.step || 0;
              
              // 백트래킹 로깅 (삭제 전 세션 사용)
              const graphSession = convertNodesToGraphSession(nodes, edges);
              if (graphSession) {
                await logActionAndSaveSession(
                  "backtrack",
                  {
                    targetNodeId: removedNodeId,
                    targetStep: step,
                    branchId: "",
                    backtrackToStep: step > 0 ? step - 1 : 0,
                    removedNodeIds: removedNodeIds,
                  },
                  participant,
                  mode,
                  graphSession,
                  bookmarkedNodeIds
                ).catch((error) => {
                  console.error("[SimpleGraphCanvas] Failed to log backtrack:", error);
                });
              }
            } else if (removedNode.type === "prompt") {
              // 프롬프트 노드 삭제는 node_deleted로 로깅
              const graphSession = convertNodesToGraphSession(nodes, edges);
              if (graphSession) {
                await logActionAndSaveSession(
                  "node_deleted",
                  {
                    nodeId: removedNodeId,
                    nodeType: "prompt",
                    deletedNodeIds: removedNodeIds,
                  },
                  participant,
                  mode,
                  graphSession,
                  bookmarkedNodeIds
                ).catch((error) => {
                  console.error("[SimpleGraphCanvas] Failed to log node_deleted:", error);
                });
              }
            }
          }
        }
      }
      
      // 기본 노드 변경 처리
      onNodesChangeInternal(changes);
    },
    [nodes, edges, participant, mode, bookmarkedNodeIds, onNodesChangeInternal]
  );
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

      const handleUploadImages = (files: File[] | null) => {
        if (!files || files.length === 0) {
          setNodes((prev) =>
            prev.map((node) =>
              node.id === id
                ? {
                    ...node,
                    data: {
                      ...(node.data as SimplePromptNodeData),
                      inputImagePreviewUrls: [],
                      inputImageDataUrls: [],
                      inputImageSourceNodeIds: [],
                    },
                  }
                : node
            )
          );
          return;
        }

        const readers = files.map(
          (file) =>
            new Promise<string | null>((resolve) => {
              const reader = new FileReader();
              reader.onload = () => {
                const result =
                  typeof reader.result === "string" ? reader.result : null;
                resolve(result);
              };
              reader.onerror = () => resolve(null);
              reader.readAsDataURL(file);
            })
        );

        Promise.all(readers).then((results) => {
          const validResults = results.filter(
            (r): r is string => r !== null && r !== undefined
          );
          if (validResults.length === 0) return;
          setNodes((prev) =>
            prev.map((node) =>
              node.id === id
                ? {
                    ...node,
                    data: {
                      ...(node.data as SimplePromptNodeData),
                      // Append new uploads to any existing images (including those from edges)
                      ...(() => {
                        const data = node.data as SimplePromptNodeData;
                        const prevPreview = data.inputImagePreviewUrls ?? [];
                        const prevData = data.inputImageDataUrls ?? [];
                        const prevSourceIds =
                          data.inputImageSourceNodeIds ?? [];

                        return {
                          inputImagePreviewUrls: [
                            ...prevPreview,
                            ...validResults,
                          ],
                          inputImageDataUrls: [...prevData, ...validResults],
                          // Uploaded images have no source node; align array lengths with nulls
                          inputImageSourceNodeIds: [
                            ...prevSourceIds,
                            ...validResults.map(() => null),
                          ],
                        };
                      })(),
                    },
                  }
                : node
            )
          );
        });
      };

      const handleRemoveInputImage = (index: number) => {
        let sourceNodeIdToDetach: string | null = null;

        setNodes((prev) =>
          prev.map((node) => {
            if (node.id !== id || node.type !== "prompt") return node;
            const data = node.data as SimplePromptNodeData;
            const previews = data.inputImagePreviewUrls ?? [];
            const datas = data.inputImageDataUrls ?? [];
            const sources = data.inputImageSourceNodeIds ?? [];

            sourceNodeIdToDetach =
              sources[index] !== undefined ? sources[index] ?? null : null;

            return {
              ...node,
              data: {
                ...data,
                inputImagePreviewUrls: previews.filter(
                  (_v, i) => i !== index
                ),
                inputImageDataUrls: datas.filter((_v, i) => i !== index),
                inputImageSourceNodeIds: sources.filter(
                  (_v, i) => i !== index
                ),
              },
            };
          })
        );

        if (sourceNodeIdToDetach) {
          setEdges((prev) =>
            prev.filter(
              (e) => !(e.source === sourceNodeIdToDetach && e.target === id)
            )
          );
        }
      };

      return {
        id,
        type: "prompt",
        position,
        data: {
          prompt: "",
          onChangePrompt: handleChangePrompt,
          inputImagePreviewUrls: [],
          inputImageDataUrls: [],
          inputImageSourceNodeIds: [],
          onUploadImages: handleUploadImages,
          onRemoveInputImage: handleRemoveInputImage,
        } as SimplePromptNodeData,
      };
    },
    [setNodes, setEdges]
  );

  // Initialize with a single prompt node or load from server
  useEffect(() => {
    if (nodes.length === 0) {
      // participant가 있으면 서버에서 로드 시도
      if (participant !== null && participant !== undefined && mode) {
        loadSessionFromServer(mode, participant)
          .then(() => {
            const { currentGraphSession } = useImageStore.getState();
            if (currentGraphSession) {
              // GraphSession을 nodes/edges로 변환
              const { nodes: loadedNodes, edges: loadedEdges } = convertGraphSessionToNodes(currentGraphSession);
              setNodes(loadedNodes);
              setEdges(loadedEdges);
              
              // activePromptId 찾기
              const firstPromptNode = loadedNodes.find(n => n.type === 'prompt');
              if (firstPromptNode) {
                setActivePromptId(firstPromptNode.id);
                // nextPromptIndex 계산
                const promptNodes = loadedNodes.filter(n => n.type === 'prompt');
                const maxIndex = Math.max(...promptNodes.map(n => {
                  const match = n.id.match(/prompt-(\d+)/);
                  return match ? parseInt(match[1], 10) : 0;
                }));
                setNextPromptIndex(maxIndex + 1);
              }
              console.log("[SimpleGraphCanvas] Session loaded and converted to nodes/edges");
            } else {
              // 로드된 세션이 없으면 기본 노드 생성
              const initialId = "prompt-1";
              const initialNode = createPromptNode(initialId, { x: 0, y: 0 });
              setNodes([initialNode]);
              setActivePromptId(initialId);
              setNextPromptIndex(2);
            }
          })
          .catch((error) => {
            console.warn("[SimpleGraphCanvas] Failed to load session, initializing default:", error);
            // 로드 실패 시 기본 노드 생성
            const initialId = "prompt-1";
            const initialNode = createPromptNode(initialId, { x: 0, y: 0 });
            setNodes([initialNode]);
            setActivePromptId(initialId);
            setNextPromptIndex(2);
          });
      } else {
        // participant가 없으면 기본 노드 생성
        const initialId = "prompt-1";
        const initialNode = createPromptNode(initialId, { x: 0, y: 0 });
        setNodes([initialNode]);
        setActivePromptId(initialId);
        setNextPromptIndex(2);
      }
    }
  }, [nodes.length, createPromptNode, mode, participant, loadSessionFromServer, setNodes, setEdges]);
  
  // 북마크 변경 감지 및 로깅
  const prevBookmarkedNodeIdsRef = useRef<string[]>([]);
  useEffect(() => {
    if (!participant || !mode) return;
    
    const currentBookmarked = bookmarkedNodeIds || [];
    const prevBookmarked = prevBookmarkedNodeIdsRef.current;
    
    // 새로 추가된 북마크
    const newlyBookmarked = currentBookmarked.filter((id) => !prevBookmarked.includes(id));
    // 제거된 북마크
    const newlyUnbookmarked = prevBookmarked.filter((id) => !currentBookmarked.includes(id));
    
    // 로깅
    const logBookmarkChanges = async () => {
      const graphSession = convertNodesToGraphSession(nodes, edges);
      if (!graphSession) return;
      
      for (const nodeId of newlyBookmarked) {
        const node = nodes.find((n) => n.id === nodeId);
        if (node && node.type === "image") {
          const imageData = node.data as { imageUrl?: string; step?: number };
          await logActionAndSaveSession(
            "bookmark_toggled",
            {
              nodeId: nodeId,
              nodeStep: imageData.step || 0,
              branchId: "",
              isBookmarked: true,
              imageUrl: imageData.imageUrl || "",
            },
            participant,
            mode,
            graphSession,
            bookmarkedNodeIds
          ).catch((error) => {
            console.error("[SimpleGraphCanvas] Failed to log bookmark_toggled:", error);
          });
        }
      }
      
      for (const nodeId of newlyUnbookmarked) {
        const node = nodes.find((n) => n.id === nodeId);
        if (node && node.type === "image") {
          const imageData = node.data as { imageUrl?: string; step?: number };
          await logActionAndSaveSession(
            "bookmark_toggled",
            {
              nodeId: nodeId,
              nodeStep: imageData.step || 0,
              branchId: "",
              isBookmarked: false,
              imageUrl: imageData.imageUrl || "",
            },
            participant,
            mode,
            graphSession,
            bookmarkedNodeIds
          ).catch((error) => {
            console.error("[SimpleGraphCanvas] Failed to log bookmark_toggled:", error);
          });
        }
      }
    };
    
    if (newlyBookmarked.length > 0 || newlyUnbookmarked.length > 0) {
      logBookmarkChanges();
    }
    
    prevBookmarkedNodeIdsRef.current = currentBookmarked;
  }, [bookmarkedNodeIds, nodes, edges, participant, mode]);

  // nodes/edges 변경 시 자동 저장 (debounce 적용)
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  useEffect(() => {
    // participant가 없거나 mode가 없으면 저장하지 않음
    if (!participant || !mode || nodes.length === 0) {
      return;
    }

    // 이전 타이머 취소
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    // 10초 후에 저장 (debounce)
    saveTimeoutRef.current = setTimeout(() => {
      if (mode === "prompt" && participant) {
        // GraphSession으로 변환하여 직접 저장
        const graphSession = convertNodesToGraphSession(nodes, edges, `simple-${participant}`);
        const { bookmarkedNodeIds } = useImageStore.getState();
        
        // saveSession API를 직접 호출
        saveSession(mode, participant, graphSession, bookmarkedNodeIds)
          .then(() => {
            console.log("[SimpleGraphCanvas] Session auto-saved");
          })
          .catch((error) => {
            console.error("[SimpleGraphCanvas] Failed to auto-save session:", error);
          });
      }
    }, 10000);

    // cleanup
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [nodes, edges, bookmarkedNodeIds, mode, participant]);

  const activePromptText = useMemo(() => {
    if (!activePromptId) return "";
    const node = nodes.find(
      (n) => n.id === activePromptId && n.type === "prompt"
    );
    if (!node) return "";
    return (node.data as SimplePromptNodeData).prompt ?? "";
  }, [activePromptId, nodes]);

  // Check if the active prompt already has generated images
  const hasGeneratedImagesForActivePrompt = useMemo(() => {
    if (!activePromptId) return false;
    return nodes.some(
      (n) =>
        n.type === "image" &&
        (n.data as { parentPromptId?: string }).parentPromptId === activePromptId
    );
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
    let inputImageDataUrls: string[] =
      promptData.inputImageDataUrls ?? [];

    // If this prompt is connected from an image node, use that image and its parent's prompt
    const incomingImageEdges = edges.filter((e) => e.target === activePromptId);
    if (incomingImageEdges.length > 0) {
      const existingSet = new Set(inputImageDataUrls);
      for (const edge of incomingImageEdges) {
        const sourceNode = nodes.find((n) => n.id === edge.source);
        if (sourceNode?.type === "image") {
          const imageData = (sourceNode.data as {
            imageUrl?: string;
            parentPromptId?: string;
          }).imageUrl;
          if (imageData && !existingSet.has(imageData)) {
            existingSet.add(imageData);
            inputImageDataUrls.push(imageData);
          }
          if (previousPromptText == null) {
            const parentPromptId = (sourceNode.data as {
              parentPromptId?: string;
            }).parentPromptId;
            if (parentPromptId) {
              const parentPromptNode = nodes.find(
                (n) => n.id === parentPromptId && n.type === "prompt"
              );
              if (parentPromptNode) {
                previousPromptText =
                  (parentPromptNode.data as SimplePromptNodeData).prompt ??
                  null;
              }
            }
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
    
    // 로깅: Simple Generate 시작
    const graphSession = convertNodesToGraphSession(nodes, edges);
    if (participant && mode && graphSession) {
      await logActionAndSaveSession(
        "simple_generate",
        {
          promptNodeId: activePromptId,
          prompt: promptText.trim(),
          imageUrl: inputImageDataUrls.length > 0 ? inputImageDataUrls[0] : undefined,
          isRegeneration: inputImageDataUrls.length > 0,
        },
        participant,
        mode,
        graphSession,
        bookmarkedNodeIds
      ).catch((error) => {
        console.error("[SimpleGraphCanvas] Failed to log simple_generate:", error);
      });
    }

    const applyImages = async (res: SimpleGenerateResponse) => {
      const generationStartTime = Date.now();

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
      
      // 로깅: 이미지 수신 (각 이미지마다) - setNodes/setEdges 후에 업데이트된 세션 사용
      // setTimeout을 사용하여 상태 업데이트 후 로깅
      setTimeout(async () => {
        // 상태 업데이트 후 최신 nodes와 edges 사용
        const updatedGraphSession = convertNodesToGraphSession(nodes, edges);
        if (participant && mode && updatedGraphSession) {
          const generationDuration = Date.now() - generationStartTime;
          for (let index = 0; index < res.images.length; index++) {
            const b64 = res.images[index];
            if (b64) {
              const nodeId = `${activePromptId}-image-${index + 1}`;
              await logActionAndSaveSession(
                "image_received",
                {
                  nodeId,
                  branchId: "",
                  step: index + 1,
                  imageUrl: `data:image/png;base64,${b64}`,
                  generationDuration,
                  sourceAction: "simple_generate",
                },
                participant,
                mode,
                updatedGraphSession,
                bookmarkedNodeIds
              ).catch((error) => {
                console.error("[SimpleGraphCanvas] Failed to log image_received:", error);
              });
            }
          }
        }
      }, 100);
    };

    try {
      let res: SimpleGenerateResponse;
      if (inputImageDataUrls.length > 0) {
        res = await generateWithImage({
          current_prompt: promptText.trim(),
          previous_prompt: previousPromptText,
          num_images: 4,
          imageDataUrls: inputImageDataUrls,
        });
      } else {
        res = await generateSimpleImages({
          prompt: promptText.trim(),
          previous_prompt: previousPromptText,
          num_images: 4,
        });
      }

      await applyImages(res);
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

  const handleAddPrompt = useCallback(async () => {
    const id = `prompt-${nextPromptIndex}`;
    const promptCount = nodes.filter((n) => n.type === "prompt").length;
    // Place input nodes horizontally across instead of vertically
    const position = { x: promptCount * 360, y: 0 };
    const newNode = createPromptNode(id, position);

    setNodes((prev) => [...prev, newNode]);
    setActivePromptId(id);
    setNextPromptIndex((prev) => prev + 1);
    
    // 로깅: 프롬프트 노드 생성
    const graphSession = convertNodesToGraphSession([...nodes, newNode], edges);
    if (participant && mode && graphSession) {
      const promptText = (newNode.data as SimplePromptNodeData).prompt || "";
      await logActionAndSaveSession(
        "prompt_node_created",
        {
          nodeId: id,
          prompt: promptText,
        },
        participant,
        mode,
        graphSession,
        bookmarkedNodeIds
      ).catch((error) => {
        console.error("[SimpleGraphCanvas] Failed to log prompt_node_created:", error);
      });
    }
  }, [nextPromptIndex, nodes, createPromptNode, participant, mode, edges, bookmarkedNodeIds]);

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
    async (connection: Connection) => {
      if (!connection.source || !connection.target) return;

      const sourceNode = nodes.find((n) => n.id === connection.source);
      const targetNode = nodes.find((n) => n.id === connection.target);
      if (!sourceNode || !targetNode) return;

      // Only allow image -> prompt connections for using image as next input
      if (sourceNode.type !== "image" || targetNode.type !== "prompt") return;

      const imageUrl = (sourceNode.data as { imageUrl?: string }).imageUrl;
      const targetData = targetNode.data as SimplePromptNodeData;
      const prevData = targetData.inputImageDataUrls ?? [];
      
      // 중복 체크
      const isDuplicate = imageUrl ? prevData.includes(imageUrl) : false;

      setNodes((prev) =>
        prev.map((n) =>
          n.id === targetNode.id
            ? {
                ...n,
                data: {
                  ...(n.data as SimplePromptNodeData),
                  // Append this connected image to the prompt node's input image arrays
                  ...(() => {
                    const data = n.data as SimplePromptNodeData;
                    const prevPreview =
                      data.inputImagePreviewUrls ?? [];
                    const prevData = data.inputImageDataUrls ?? [];
                    const prevSourceIds =
                      data.inputImageSourceNodeIds ?? [];

                    if (!imageUrl) {
                      return {
                        inputImagePreviewUrls: prevPreview,
                        inputImageDataUrls: prevData,
                        inputImageSourceNodeIds: prevSourceIds,
                      };
                    }

                    // Avoid duplicates by data URL
                    if (prevData.includes(imageUrl)) {
                      return {
                        inputImagePreviewUrls: prevPreview,
                        inputImageDataUrls: prevData,
                        inputImageSourceNodeIds: prevSourceIds,
                      };
                    }

                    return {
                      inputImagePreviewUrls: [...prevPreview, imageUrl],
                      inputImageDataUrls: [...prevData, imageUrl],
                      inputImageSourceNodeIds: [
                        ...prevSourceIds,
                        sourceNode.id,
                      ],
                    };
                  })(),
                },
              }
            : n
        )
      );
      
      // 로깅: 이미지 노드가 프롬프트 노드에 연결됨 (중복이 아닌 경우만)
      if (!isDuplicate && imageUrl && participant && mode) {
        const updatedNodes = nodes.map((n) =>
          n.id === targetNode.id
            ? {
                ...n,
                data: {
                  ...(n.data as SimplePromptNodeData),
                  inputImagePreviewUrls: [
                    ...(targetData.inputImagePreviewUrls ?? []),
                    imageUrl,
                  ],
                  inputImageDataUrls: [
                    ...(targetData.inputImageDataUrls ?? []),
                    imageUrl,
                  ],
                  inputImageSourceNodeIds: [
                    ...(targetData.inputImageSourceNodeIds ?? []),
                    sourceNode.id,
                  ],
                },
              }
            : n
        );
        const updatedGraphSession = convertNodesToGraphSession(updatedNodes, edges);
        if (updatedGraphSession) {
          await logActionAndSaveSession(
            "image_connected_to_prompt",
            {
              promptNodeId: targetNode.id,
              imageNodeId: sourceNode.id,
              imageUrl: imageUrl,
            },
            participant,
            mode,
            updatedGraphSession,
            bookmarkedNodeIds
          ).catch((error) => {
            console.error("[SimpleGraphCanvas] Failed to log image_connected_to_prompt:", error);
          });
        }
      }

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

    // Walk all ancestor chains from the selected node back to the root,
    // supporting multiple parents (e.g., prompts with multiple input images).
    const pathNodeIds = new Set<string>();
    const pathEdgeIds = new Set<string>();

    const queue: string[] = [selectedNodeId];
    while (queue.length > 0) {
      const currentId = queue.shift()!;
      if (pathNodeIds.has(currentId)) continue;
      pathNodeIds.add(currentId);

      // All incoming edges are considered parents
      const incomingEdges = edges.filter((e) => e.target === currentId);
      for (const incoming of incomingEdges) {
        pathEdgeIds.add(incoming.id);
        if (!pathNodeIds.has(incoming.source)) {
          queue.push(incoming.source);
        }
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
        hasGeneratedImagesForActivePrompt={hasGeneratedImagesForActivePrompt}
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
  hasGeneratedImagesForActivePrompt: boolean;
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
  hasGeneratedImagesForActivePrompt,
  error,
  handleGenerate,
  handleAddPrompt,
}) => {
  const { setCenter } = useReactFlow();
  const { selectNode } = useImageStore();

  return (
    <CanvasContainer>
      {/* <HelperText>
        Type your prompt in any input node, optionally upload or connect an
        image, then click &ldquo;Generate&rdquo; to create 4 images branching
        to the right. Drag an image node connection into a new prompt to use
        it as the next input.
        {error ? `  •  Error: ${error}` : null}
      </HelperText> */}

      {/* 북마크 패널 */}
      <BookmarkPanelWrapper
        reactFlowNodes={nodes}
        selectNode={selectNode}
        nodes={nodes}
      />

        <BottomCenterControls>
        <AddPromptButton onClick={handleAddPrompt} disabled={isGenerating}>
            + Prompt Node
          </AddPromptButton>
          <GenerateButton onClick={handleGenerate} disabled={isGenerateDisabled}>
            {isGenerating
              ? "Generating..."
              : hasGeneratedImagesForActivePrompt
              ? "Regenerate"
              : "Generate"}
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
        // Use scroll wheel to zoom in/out the canvas
        zoomOnScroll
        zoomOnPinch
        // Keep panning on drag, but disable scroll-wheel panning
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

