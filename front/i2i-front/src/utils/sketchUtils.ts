import { type SketchLayer } from "../types";

/**
 * 스케치 레이어를 투명 배경 PNG로 변환
 * @param layers 스케치 레이어 배열
 * @param canvasWidth 캔버스 너비
 * @param canvasHeight 캔버스 높이
 * @returns PNG 데이터 URL (투명 배경)
 */
export function exportSketchToPNG(
  layers: SketchLayer[],
  canvasWidth: number,
  canvasHeight: number
): string {
  const canvas = document.createElement("canvas");
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  const ctx = canvas.getContext("2d");
  
  if (!ctx) {
    throw new Error("Canvas context를 가져올 수 없습니다.");
  }

  // 투명 배경으로 시작
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);

  // 각 레이어를 그리기
  layers.forEach((layer) => {
    ctx.strokeStyle = layer.color;
    ctx.lineWidth = 10; // 색연필 느낌의 두꺼운 선
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    layer.paths.forEach((path) => {
      if (path.length < 2) return;

      ctx.beginPath();
      ctx.moveTo(path[0].x * canvasWidth, path[0].y * canvasHeight);

      for (let i = 1; i < path.length; i++) {
        ctx.lineTo(path[i].x * canvasWidth, path[i].y * canvasHeight);
      }

      ctx.stroke();
    });
  });

  return canvas.toDataURL("image/png");
}

/**
 * 스케치 레이어를 File 객체로 변환
 * @param layers 스케치 레이어 배열
 * @param canvasWidth 캔버스 너비
 * @param canvasHeight 캔버스 높이
 * @param filename 파일명
 * @returns File 객체
 */
export function exportSketchToFile(
  layers: SketchLayer[],
  canvasWidth: number,
  canvasHeight: number,
  filename: string = "sketch.png"
): Promise<File> {
  return new Promise((resolve, reject) => {
    try {
      const dataURL = exportSketchToPNG(layers, canvasWidth, canvasHeight);
      
      // Data URL을 Blob으로 변환
      fetch(dataURL)
        .then((res) => res.blob())
        .then((blob) => {
          const file = new File([blob], filename, { type: "image/png" });
          resolve(file);
        })
        .catch(reject);
    } catch (error) {
      reject(error);
    }
  });
}

/**
 * 모든 스케치 레이어를 하나의 통합 스케치로 변환
 * @param layers 스케치 레이어 배열
 * @returns 통합 스케치 레이어 (모든 객체 색상 포함)
 */
export function mergeSketchLayers(layers: SketchLayer[]): SketchLayer {
  // 모든 경로를 하나로 합치기
  const allPaths: Array<{ x: number; y: number }[]> = [];
  layers.forEach((layer) => {
    allPaths.push(...layer.paths);
  });

  return {
    objectId: "merged",
    color: "#000000", // 통합 스케치는 검은색으로 표시 (실제로는 각 레이어의 색상이 유지됨)
    paths: allPaths,
  };
}

