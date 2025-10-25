export type ToolMode = "point" | "bbox" | "none";

export interface InteractionData {
  type: "point" | "bbox";
  x: number;
  y: number;
  width?: number;
  height?: number;
}
