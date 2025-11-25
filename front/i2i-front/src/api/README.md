# API Endpoints Documentation

이 문서는 프론트엔드에서 사용하는 API 엔드포인트를 정리한 것입니다.
백엔드 개발 시 이 문서를 참고하여 API를 구현하세요.

## 1. 객체 리스트 요청

**Endpoint:** `POST /api/composition/objects`

**Request:**

```json
{
  "prompt": "a beautiful landscape with mountains and trees"
}
```

**Response:**

```json
{
  "objects": [
    {
      "id": "obj_1234567890_0",
      "label": "Mountain",
      "color": "#6366f1"
    },
    {
      "id": "obj_1234567890_1",
      "label": "Tree",
      "color": "#8b5cf6"
    }
  ]
}
```

**설명:**

- 프롬프트를 받아 LLM을 통해 필요한 객체 리스트를 생성합니다.
- 각 객체는 고유한 ID, 라벨, 색상을 가집니다.
- 색상은 프론트엔드에서 BBOX 표시에 사용됩니다.

---

## 2. 이미지 생성 시작

**Endpoint:** `POST /api/composition/start`

**Request:**

```json
{
  "prompt": "a beautiful landscape with mountains and trees",
  "objects": [
    {
      "id": "obj_1234567890_0",
      "label": "Mountain",
      "color": "#6366f1"
    }
  ],
  "bboxes": [
    {
      "objectId": "obj_1234567890_0",
      "x": 0.2,
      "y": 0.3,
      "width": 0.4,
      "height": 0.5
    }
  ]
}
```

**Response:**

```json
{
  "sessionId": "session_1234567890",
  "websocketUrl": "ws://localhost:8000/ws/image-stream/session_1234567890"
}
```

**설명:**

- 프롬프트, 객체 리스트, 구도 설정(bboxes)을 받아 이미지 생성 세션을 시작합니다.
- `bboxes`는 선택적입니다 (구도 설정이 없는 경우 빈 배열).
- `objects`도 선택적입니다 (구도 설정이 없는 경우 빈 배열).
- 세션 ID와 WebSocket URL을 반환합니다.
- 이후 WebSocket을 통해 이미지 스트림을 받을 수 있습니다.

---

## 3. WebSocket 이미지 스트림

**WebSocket URL:** `ws://{host}/ws/image-stream/{sessionId}`

**연결 시 전송할 메시지 (선택적):**

```json
{
  "type": "subscribe",
  "sessionId": "session_1234567890"
}
```

**서버에서 받을 메시지:**

### 이미지 스텝 수신

```json
{
  "type": "image_step",
  "stepId": "step_session_1234567890_5",
  "imageUrl": "https://example.com/images/step_5.png",
  "step": 5,
  "timestamp": 1234567890
}
```

### 생성 완료

```json
{
  "type": "complete"
}
```

### 에러 발생

```json
{
  "type": "error",
  "message": "Error message here"
}
```

**설명:**

- 이미지 생성이 진행되면서 각 스텝마다 이미지 URL을 전송합니다.
- `step`은 현재 생성 스텝 번호입니다 (예: 1, 5, 10, 15, 20).
- `imageUrl`은 해당 스텝의 이미지 URL입니다.
- 생성이 완료되면 `complete` 메시지를 전송합니다.

---

## 4. 피드백 제출

**Endpoint:** `POST /api/feedback`

**Request (FormData):**

```
sessionId: "session_1234567890"
feedbacks: "[{\"area\":\"full\",\"type\":\"text\",\"text\":\"좀 더 밝게 해주세요\"},{\"area\":\"bbox\",\"type\":\"text\",\"text\":\"이 부분을 수정해주세요\",\"bbox\":{\"x\":0.2,\"y\":0.3,\"width\":0.4,\"height\":0.5}}]"
image_0: (File, 선택적)
image_1: (File, 선택적)
```

**Request Body 설명:**

- `sessionId`: 현재 세션 ID
- `feedbacks`: JSON 문자열로 직렬화된 피드백 배열
  - 각 피드백 객체:
    - `area`: "full" | "bbox" | "point"
    - `type`: "text" | "image"
    - `text`: 텍스트 피드백 (선택적)
    - `point`: { x: number, y: number } (area가 "point"인 경우)
    - `bbox`: { x: number, y: number, width: number, height: number } (area가 "bbox"인 경우)
- `image_0`, `image_1`, ...: 이미지 피드백 파일들 (type이 "image"인 경우)

**Response:**

```json
{
  "success": true
}
```

**설명:**

- 여러 개의 피드백을 한번에 제출할 수 있습니다.
- 각 피드백은 전체 이미지, 특정 BBOX, 또는 특정 포인트에 대한 피드백일 수 있습니다.
- 이미지 피드백의 경우 파일을 함께 전송합니다.

---

## 5. 피드백 건너뛰기

**Endpoint:** `POST /api/feedback/skip`

**Request:**

```json
{
  "sessionId": "session_1234567890"
}
```

**Response:**

```json
{
  "success": true
}
```

**설명:**

- 사용자가 피드백을 제공하지 않고 건너뛰는 경우 호출됩니다.
- 세션 ID를 전송하여 어떤 세션의 피드백을 건너뛰는지 명시합니다.

---

## 현재 상태

현재 모든 API는 **Mockup 모드**로 동작합니다.
실제 백엔드가 준비되면 각 API 파일의 `TODO` 주석 부분을 해제하고 mockup 코드를 제거하면 됩니다.

### Mockup 해제 방법

각 API 파일 (`composition.ts`, `feedback.ts`, `websocket.ts`)에서:

1. `TODO: 실제 백엔드 연결 시...` 주석 아래의 코드 주석 해제
2. `===== MOCKUP =====` 섹션의 코드 제거
