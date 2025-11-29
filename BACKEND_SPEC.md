# 백엔드 API 스펙 문서

이 문서는 프론트엔드와 백엔드 간의 통신 스펙을 정의합니다.

## 목차

1. [개요](#개요)
2. [API 엔드포인트](#api-엔드포인트)
3. [WebSocket 통신](#websocket-통신)
4. [데이터 구조](#데이터-구조)
5. [워크플로우](#워크플로우)

---

## 개요

이 시스템은 사용자가 프롬프트와 구도 설정을 통해 이미지를 생성하고, 중간 단계에서 피드백을 주어 브랜치를 생성할 수 있는 인터랙티브 이미지 생성 시스템입니다.

### 주요 기능

- 프롬프트 기반 객체 리스트 생성 (LLM 사용)
- 구도 설정을 포함한 이미지 생성 시작
- WebSocket을 통한 실시간 diffusion step 스트리밍
- 브랜치 생성 및 피드백 처리
- 병렬 이미지 생성 및 관리

---

## API 엔드포인트

### 1. 객체 리스트 생성

**엔드포인트:** `POST /api/composition/objects`

**설명:** 프롬프트를 받아 LLM을 이용하여 이미지 생성에 필요한 객체 리스트를 생성합니다.

**요청:**

```json
{
  "prompt": "a beautiful sunset over the ocean with a sailboat"
}
```

**응답:**

```json
{
  "objects": [
    {
      "id": "obj_1234567890_0",
      "label": "Sunset",
      "color": "#6366f1"
    },
    {
      "id": "obj_1234567890_1",
      "label": "Ocean",
      "color": "#8b5cf6"
    },
    {
      "id": "obj_1234567890_2",
      "label": "Sailboat",
      "color": "#ec4899"
    }
  ]
}
```

**구현 요구사항:**

- LLM (예: GPT-4, Claude 등)을 사용하여 프롬프트에서 객체를 추출
- 각 객체에 고유한 ID와 색상 할당
- 객체는 이미지 생성에 필요한 주요 요소들을 포함해야 함

---

### 2. 이미지 생성 시작

**엔드포인트:** `POST /api/composition/start`

**설명:** 사용자가 설정한 구도 정보를 받아 이미지 생성을 시작합니다. 구도 정보가 없을 수도 있습니다.

**요청:**

```json
{
  "prompt": "a beautiful sunset over the ocean with a sailboat",
  "objects": [
    {
      "id": "obj_1234567890_0",
      "label": "Sunset",
      "color": "#6366f1"
    }
  ],
  "bboxes": [
    {
      "objectId": "obj_1234567890_0",
      "x": 0.1,
      "y": 0.2,
      "width": 0.3,
      "height": 0.4
    }
  ]
}
```

**응답:**

```json
{
  "sessionId": "session_1234567890",
  "rootNodeId": "node_prompt_1234567890",
  "websocketUrl": "ws://localhost:8000/ws/image-stream/session_1234567890"
}
```

**구현 요구사항:**

- 세션 ID 생성 및 관리
- 프롬프트 노드 ID 생성
- WebSocket URL 반환
- 구도 정보가 없는 경우에도 처리 가능해야 함
- 이미지 생성 파이프라인 초기화

---

## WebSocket 통신

### 연결

**URL:** `ws://{host}/ws/image-stream/{sessionId}`

**설명:** 이미지 생성 세션이 시작되면 WebSocket 연결을 통해 실시간으로 diffusion step 이미지를 스트리밍합니다.

### 메시지 형식

#### 서버 → 클라이언트 (이미지 스트림)

**이벤트 타입:** `image_step`

**메시지:**

```json
{
  "type": "image_step",
  "sessionId": "session_1234567890",
  "nodeId": "node_image_1234567890_1",
  "parentNodeId": "node_prompt_1234567890",
  "step": 5,
  "totalSteps": 20,
  "imageUrl": "https://example.com/images/step_5.png",
  "imageData": "base64_encoded_image_data", // 또는 URL
  "timestamp": 1234567890
}
```

**구현 요구사항:**

- 각 diffusion step마다 이미지를 생성하여 전송
- `step`은 현재 단계, `totalSteps`는 전체 단계 수
- `nodeId`는 각 이미지 노드의 고유 ID
- `parentNodeId`는 이전 노드의 ID (체인 구조)
- 이미지는 base64 인코딩 또는 URL로 전송 가능

#### 서버 → 클라이언트 (생성 완료)

**이벤트 타입:** `generation_complete`

**메시지:**

```json
{
  "type": "generation_complete",
  "sessionId": "session_1234567890",
  "nodeId": "node_image_1234567890_20",
  "finalImageUrl": "https://example.com/images/final.png"
}
```

---

### 브랜치 생성 및 피드백 처리

#### 클라이언트 → 서버 (브랜치 생성 요청)

**엔드포인트:** `POST /api/branch/create`

**요청:**

```json
{
  "sessionId": "session_1234567890",
  "sourceNodeId": "node_image_1234567890_10",
  "feedback": [
    {
      "id": "feedback_1234567890_0",
      "area": "full",
      "type": "text",
      "text": "더 밝게 만들어주세요"
    },
    {
      "id": "feedback_1234567890_1",
      "area": "bbox",
      "type": "text",
      "text": "이 부분을 더 선명하게",
      "bbox": {
        "x": 0.2,
        "y": 0.3,
        "width": 0.4,
        "height": 0.5
      }
    },
    {
      "id": "feedback_1234567890_2",
      "area": "point",
      "type": "text",
      "text": "이 지점을 중심으로",
      "point": {
        "x": 0.5,
        "y": 0.6
      }
    },
    {
      "id": "feedback_1234567890_3",
      "area": "full",
      "type": "image",
      "imageUrl": "https://example.com/reference_image.png"
    }
  ]
}
```

**응답:**

```json
{
  "branchId": "branch_1234567890",
  "websocketUrl": "ws://localhost:8000/ws/image-stream/session_1234567890/branch_1234567890"
}
```

**구현 요구사항:**

- 브랜치 ID 생성
- 피드백 정보 저장
- 피드백을 기반으로 새로운 이미지 생성 시작
- WebSocket을 통해 브랜치의 이미지 스트림 전송

#### WebSocket (브랜치 이미지 스트림)

브랜치 생성 후에도 동일한 WebSocket 형식으로 이미지를 스트리밍하되, `branchId`가 포함됩니다:

```json
{
  "type": "image_step",
  "sessionId": "session_1234567890",
  "branchId": "branch_1234567890",
  "nodeId": "node_image_1234567890_branch_1",
  "parentNodeId": "node_image_1234567890_10",
  "step": 5,
  "totalSteps": 20,
  "imageUrl": "https://example.com/images/branch_step_5.png",
  "timestamp": 1234567890
}
```

---

## 데이터 구조

### 세션 (Session)

```typescript
interface Session {
  id: string; // 세션 ID
  prompt: string; // 프롬프트
  rootNodeId: string; // 루트 프롬프트 노드 ID
  createdAt: number; // 생성 시간
  compositionBboxes?: BoundingBox[]; // 구도 설정 (선택적)
}
```

### 노드 (Node)

```typescript
interface Node {
  id: string; // 노드 ID
  type: "prompt" | "image"; // 노드 타입
  sessionId: string; // 세션 ID
  branchId?: string; // 브랜치 ID (브랜치 노드인 경우)
  parentNodeId?: string; // 부모 노드 ID
  step?: number; // 이미지 스텝 (이미지 노드인 경우)
  imageUrl?: string; // 이미지 URL
  position: { x: number; y: number }; // 그래프 상 위치
}
```

### 브랜치 (Branch)

```typescript
interface Branch {
  id: string; // 브랜치 ID
  sessionId: string; // 세션 ID
  sourceNodeId: string; // 브랜치가 시작된 노드 ID
  feedback: FeedbackRecord[]; // 피드백 리스트
  nodes: string[]; // 브랜치에 속한 노드 ID들
  createdAt: number; // 생성 시간
}
```

### 피드백 (Feedback)

```typescript
interface FeedbackRecord {
  id: string; // 피드백 ID
  area: "full" | "bbox" | "point"; // 피드백 영역
  type: "text" | "image"; // 피드백 타입
  text?: string; // 텍스트 피드백
  imageUrl?: string; // 참조 이미지 URL
  point?: { x: number; y: number }; // 포인팅 좌표
  bbox?: {
    // 바운딩 박스
    x: number;
    y: number;
    width: number;
    height: number;
  };
  timestamp: number; // 생성 시간
}
```

---

## 워크플로우

### 1. 초기 이미지 생성

```
1. 사용자가 프롬프트 입력
   → POST /api/composition/objects
   → 객체 리스트 반환

2. 사용자가 구도 설정 (선택적)
   → POST /api/composition/start
   → 세션 생성, WebSocket URL 반환

3. WebSocket 연결
   → ws://host/ws/image-stream/{sessionId}

4. 이미지 스트림 수신
   → 각 step마다 image_step 이벤트 수신
   → 프론트엔드에서 노드 생성 및 연결

5. 생성 완료
   → generation_complete 이벤트 수신
```

### 2. 브랜치 생성

```
1. 사용자가 특정 이미지 노드에서 브랜치 생성 요청
   → POST /api/branch/create
   → 피드백 정보 전송

2. 백엔드에서 브랜치 생성 및 이미지 생성 시작
   → 브랜치 ID 생성
   → 피드백을 기반으로 새로운 이미지 생성 파이프라인 시작

3. WebSocket을 통해 브랜치 이미지 스트림
   → branchId가 포함된 image_step 이벤트 전송
   → 프론트엔드에서 브랜치 노드 생성 및 연결
```

### 3. 병렬 생성 관리

**요구사항:**

- 각 세션, 브랜치는 독립적으로 관리되어야 함
- 여러 브랜치가 동시에 생성될 수 있음
- 각 브랜치는 병렬로 이미지를 생성해야 함
- 각 step image는 고유한 ID를 가져야 함

**구현 예시:**

```
세션: session_123
  ├─ 메인 브랜치: branch_main_123
  │   ├─ node_image_1 (step 1)
  │   ├─ node_image_2 (step 2)
  │   └─ ...
  ├─ 브랜치 1: branch_456
  │   ├─ node_image_branch1_1 (step 1)
  │   └─ ...
  └─ 브랜치 2: branch_789
      ├─ node_image_branch2_1 (step 1)
      └─ ...
```

---

## 저장 및 동기화

### 데이터 저장

백엔드는 다음 정보를 저장해야 합니다:

1. **세션 정보**

   - 세션 ID, 프롬프트, 생성 시간
   - 구도 설정 정보

2. **노드 정보**

   - 노드 ID, 타입, 세션 ID, 브랜치 ID
   - 부모 노드 ID, step 정보
   - 이미지 URL 또는 경로

3. **브랜치 정보**

   - 브랜치 ID, 세션 ID, 소스 노드 ID
   - 피드백 리스트
   - 브랜치에 속한 노드 ID들

4. **이미지 파일**
   - 각 step의 이미지 파일 저장
   - URL 또는 파일 경로 관리

### 동기화

프론트엔드와 백엔드 간의 동기화를 위해:

1. **세션 복원 API** (선택적)

   - `GET /api/session/{sessionId}`
   - 세션의 전체 그래프 구조 반환

2. **노드 정보 조회 API** (선택적)
   - `GET /api/node/{nodeId}`
   - 특정 노드의 상세 정보 반환

---

## 에러 처리

### 에러 응답 형식

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "에러 메시지",
    "details": {}
  }
}
```

### 주요 에러 코드

- `INVALID_PROMPT`: 프롬프트가 유효하지 않음
- `SESSION_NOT_FOUND`: 세션을 찾을 수 없음
- `NODE_NOT_FOUND`: 노드를 찾을 수 없음
- `BRANCH_CREATION_FAILED`: 브랜치 생성 실패
- `IMAGE_GENERATION_FAILED`: 이미지 생성 실패
- `WEBSOCKET_CONNECTION_FAILED`: WebSocket 연결 실패

---

## 참고사항

1. **이미지 전송 방식**

   - Base64 인코딩: 작은 이미지에 적합, 실시간 전송에 유리
   - URL 전송: 큰 이미지에 적합, 서버에 저장 후 URL 전송

2. **성능 최적화**

   - 여러 브랜치를 병렬로 처리
   - 이미지 생성 파이프라인 최적화
   - WebSocket 연결 풀 관리

3. **확장성**
   - 세션 수 제한 없이 처리 가능해야 함
   - 브랜치 수 제한 없이 처리 가능해야 함
   - 각 브랜치는 독립적으로 생성되어야 함
