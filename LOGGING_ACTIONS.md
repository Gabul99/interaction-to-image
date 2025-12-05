# 사용자 행동 로깅 시스템 - 액션 및 데이터 구조

이 문서는 사용자 행동 로깅 시스템에서 기록되는 모든 액션과 각 액션별로 저장되는 데이터를 정리한 것입니다.

## 공통 필드

모든 로그 엔트리는 다음 공통 필드를 포함합니다:

- `logId`: UUID v4 형식의 고유 로그 ID
- `timestamp`: 밀리초 단위 타임스탬프
- `participant`: 참가자 번호
- `mode`: 모드 ("step" | "prompt")
- `sessionId`: GraphSession ID
- `action`: 액션 타입 (아래 목록 참조)
- `data`: 액션별 데이터 객체

## 액션 목록 및 데이터 구조

### 1. prompt_node_created
**설명**: 프롬프트 노드가 생성될 때 기록됩니다.

**데이터 구조**:
```typescript
{
  nodeId: string;                    // 생성된 프롬프트 노드 ID
  prompt: string;                    // 프롬프트 텍스트
  compositionData?: {                // 객체 구도 배치 데이터 (있는 경우)
    bboxes: BoundingBox[];
    sketchLayers: SketchLayer[];
  };
}
```

**발생 위치**: GraphCanvas - `addPromptNodeToGraph` 호출 시

---

### 2. composition_configured
**설명**: Composition 모달에서 객체 구도 배치를 완료할 때 기록됩니다.

**데이터 구조**:
```typescript
{
  promptNodeId: string;             // 프롬프트 노드 ID
  objects: ObjectChip[];             // 객체 목록
  bboxes: BoundingBox[];             // 바운딩 박스 목록
  sketchLayers?: SketchLayer[];     // 스케치 레이어 (있는 경우)
}
```

**발생 위치**: GraphCanvas - CompositionModal 완료 시

---

### 3. generation_started
**설명**: 이미지 생성이 시작될 때 기록됩니다. 프롬프트 노드에서 "✨ 생성 시작" 버튼을 누르거나 재생성할 때 발생합니다.

**데이터 구조**:
```typescript
{
  sourceNodeId: string;              // 소스 노드 ID (프롬프트 노드 또는 이전 이미지 노드)
  sourceNodeType: "prompt" | "image"; // 소스 노드 타입
  sourceNodeStep?: number;           // 이미지 노드인 경우 스텝 번호
  branchId: string;                  // 브랜치 ID
  prompt: string;                    // 서버에 제출한 프롬프트
  isRegeneration: boolean;           // 재생성인지 여부
  compositionData?: {                // Composition 데이터 (있는 경우)
    bboxes: BoundingBox[];
    sketchLayers: SketchLayer[];
  };
}
```

**발생 위치**: GraphCanvas - `handleGenerateWithoutComposition` 함수

---

### 4. next_step_clicked
**설명**: "Next Step" 버튼을 클릭하여 다음 스텝의 이미지를 생성할 때 기록됩니다.

**데이터 구조**:
```typescript
{
  sourceNodeId: string;              // 현재 선택된 노드 ID
  sourceNodeStep: number;            // 현재 노드의 스텝 번호
  branchId: string;                  // 브랜치 ID
  expectedNextStep: number;          // 다음에 생성될 예상 스텝 번호
}
```

**발생 위치**: GraphCanvas - `handleNextStep` 함수

---

### 5. run_to_end_started
**설명**: "Run to End" 버튼을 클릭하여 끝까지 자동 생성하기 시작할 때 기록됩니다.

**데이터 구조**:
```typescript
{
  sourceNodeId: string;              // 시작 노드 ID
  sourceNodeStep: number;           // 시작 노드의 스텝 번호
  branchId: string;                 // 브랜치 ID
  currentStep: number;               // 현재 스텝
  targetStep: number;                // 목표 스텝 (보통 50)
}
```

**발생 위치**: GraphCanvas - `handleRunToEnd` 함수

---

### 6. run_to_end_paused
**설명**: "Run to End" 실행 중 "Pause" 버튼을 눌러 정지할 때 기록됩니다.

**데이터 구조**:
```typescript
{
  branchId: string;                  // 브랜치 ID
  pausedAtStep: number;             // 정지된 스텝 번호
  totalStepsGenerated: number;       // 정지까지 생성된 총 스텝 수
  duration: number;                  // 시작부터 정지까지의 시간(ms) - TODO: 실제 시간 추적 필요
}
```

**발생 위치**: GraphCanvas - `handlePause` 함수

---

### 7. simple_generate
**설명**: SimpleGraphCanvas에서 이미지 생성 버튼을 클릭할 때 기록됩니다.

**데이터 구조**:
```typescript
{
  promptNodeId: string;              // 프롬프트 노드 ID
  prompt: string;                   // 프롬프트 텍스트
  imageUrl?: string;                // 입력 이미지 URL (있는 경우, generateWithImage 사용 시)
  isRegeneration: boolean;          // 재생성인지 여부
}
```

**발생 위치**: SimpleGraphCanvas - `handleGenerate` 함수

---

### 8. image_received
**설명**: 이미지가 생성되어 수신되었을 때 기록됩니다. Next Step, Run to End, 브랜칭, 머지 등 다양한 액션의 결과로 발생할 수 있습니다.

**데이터 구조**:
```typescript
{
  nodeId: string;                    // 생성된 이미지 노드 ID
  branchId: string;                  // 브랜치 ID
  step: number;                      // 스텝 번호
  imageUrl: string;                  // 이미지 URL (base64 또는 URL)
  generationDuration: number;        // 생성 시작부터 수신까지의 시간(ms) - TODO: 실제 시간 추적 필요
  sourceAction: "next_step" | "run_to_end" | "generation_started" | "simple_generate" | "branch" | "merge";
                                     // 이 이미지를 생성하게 한 액션
}
```

**발생 위치**: 
- GraphCanvas - `addImageNode` 호출 후 (Next Step, Run to End)
- SimpleGraphCanvas - `applyImages` 함수 (Simple Generate)

---

### 9. image_connected_to_prompt
**설명**: SimpleGraphCanvas에서 이미지 노드를 프롬프트 노드에 연결하여 입력 이미지로 사용할 때 기록됩니다.

**데이터 구조**:
```typescript
{
  promptNodeId: string;                // 프롬프트 노드 ID
  imageNodeId: string;                // 연결된 이미지 노드 ID
  imageUrl: string;                   // 이미지 URL
}
```

**발생 위치**: SimpleGraphCanvas - `handleConnect` 함수 (이미지 노드 → 프롬프트 노드 연결 시)

---

### 10. branch_created
**설명**: 브랜치가 생성될 때 기록됩니다. 사용자가 피드백을 주어 새로운 브랜치를 만들 때 발생합니다.

**데이터 구조**:
```typescript
{
  sourceNodeId: string;              // 브랜치가 시작된 소스 노드 ID
  sourceNodeStep: number;           // 소스 노드의 스텝 번호
  sourceBranchId: string;           // 소스 브랜치 ID
  newBranchId: string;              // 새로 생성된 브랜치 ID
  feedback: {                        // 피드백 정보
    type: FeedbackType;              // "text" | "image"
    area: FeedbackArea;             // "full" | "point" | "bbox" | "sketch"
    text?: string;                  // 텍스트 피드백 (있는 경우)
    imageUrl?: string;               // 이미지 피드백 URL (있는 경우)
    point?: { x: number; y: number }; // 포인팅 좌표 (있는 경우)
    bbox?: {                        // 바운딩 박스 (있는 경우)
      x: number;
      y: number;
      width: number;
      height: number;
    };
    guidanceScale?: number;         // Guidance scale (있는 경우)
  };
  isAfterComplete: boolean;         // 소스 브랜치가 step 50까지 완료된 상태에서 브랜칭했는지
  sourceBranchMaxStep: number;      // 소스 브랜치의 최대 스텝
}
```

**발생 위치**: GraphCanvas - `handleBranchCreated` 함수

---

### 11. merge_created
**설명**: 두 브랜치가 병합될 때 기록됩니다. 사용자가 두 이미지 노드를 드래그하여 병합할 때 발생합니다.

**데이터 구조**:
```typescript
{
  sourceNode1Id: string;            // 첫 번째 소스 노드 ID
  sourceNode1Step: number;         // 첫 번째 소스 노드의 스텝 번호
  sourceNode1BranchId: string;     // 첫 번째 소스 브랜치 ID
  sourceNode2Id: string;            // 두 번째 소스 노드 ID
  sourceNode2Step: number;         // 두 번째 소스 노드의 스텝 번호
  sourceNode2BranchId: string;      // 두 번째 소스 브랜치 ID
  newBranchId: string;              // 새로 생성된 병합 브랜치 ID
  mergeStartStep: number;          // 병합이 시작되는 스텝 번호
  mergeWeight: number;             // 병합 가중치 (보통 0.5)
}
```

**발생 위치**: GraphCanvas - `handleMergeConfirm` 함수

---

### 12. backtrack
**설명**: 특정 스텝으로 되돌아갈 때 기록됩니다. Backspace 키를 누르거나 노드를 삭제할 때 발생합니다.

**데이터 구조**:
```typescript
{
  targetNodeId: string;             // 되돌아갈 대상 노드 ID
  targetStep: number;               // 대상 노드의 스텝 번호
  branchId: string;                 // 브랜치 ID
  backtrackToStep: number;         // 되돌아갈 스텝 번호
  removedNodeIds: string[];        // 제거된 노드 ID 목록
}
```

**발생 위치**: GraphCanvas - `handleBacktrack` 함수

---

### 13. bookmark_toggled
**설명**: 북마크를 추가하거나 제거할 때 기록됩니다.

**데이터 구조**:
```typescript
{
  nodeId: string;                   // 북마크된/제거된 노드 ID
  nodeStep: number;                 // 노드의 스텝 번호
  branchId: string;                 // 브랜치 ID
  isBookmarked: boolean;           // true: 북마크 추가, false: 북마크 제거
  imageUrl: string;                // 이미지 URL
}
```

**발생 위치**: 
- GraphCanvas - 북마크 변경 감지 useEffect
- SimpleGraphCanvas - 북마크 변경 감지 useEffect

---

## 로그 저장 위치

- **API 엔드포인트**: `POST /api/logs/save`
- **파일 저장 경로**: `logs/{mode}/p{participant}/logs_{timestamp}.json`
- **세션 파일과 동일한 디렉토리 구조 유지**

## 세션 저장 동기화

로그와 세션은 `logActionAndSaveSession()` 함수를 통해 동기화됩니다:
- 같은 `logId` (UUID)를 공유
- 같은 `timestamp` (밀리초)를 공유
- 세션 저장 시 `lastLogId`와 `lastLogTimestamp`가 포함됨

이를 통해 로그와 세션을 서로 참조할 수 있습니다.

