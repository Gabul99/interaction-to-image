# Interaction-to-Image Frontend

이미지 생성 및 브랜칭 기능을 제공하는 프론트엔드 애플리케이션입니다.

## 백엔드 연동 가이드

**중요:** 백엔드 개발을 시작하기 전에 반드시 `BACKEND_SPEC.md` 파일을 확인하세요.

### 백엔드 스펙 문서

- **위치:** `/BACKEND_SPEC.md`
- **내용:** API 엔드포인트, WebSocket 통신, 데이터 구조, 워크플로우 등 상세 스펙

### 주요 백엔드 연동 포인트

1. **객체 리스트 생성 API**

   - 파일: `src/api/composition.ts` - `requestObjectList()`
   - 엔드포인트: `POST /api/composition/objects`
   - TODO 주석 참고

2. **이미지 생성 시작 API**

   - 파일: `src/api/composition.ts` - `startImageGeneration()`
   - 엔드포인트: `POST /api/composition/start`
   - TODO 주석 참고

3. **브랜치 생성 API**

   - 파일: `src/api/branch.ts` - `createBranch()`
   - 엔드포인트: `POST /api/branch/create`
   - TODO 주석 참고

4. **WebSocket 이미지 스트림**

   - 파일: `src/api/websocket.ts` - `connectImageStream()`
   - URL 형식: `ws://{host}/ws/image-stream/{sessionId}` 또는 `ws://{host}/ws/image-stream/{sessionId}/{branchId}`
   - TODO 주석 참고

5. **이미지 스트림 처리**
   - 파일: `src/stores/imageStore.ts`
   - 함수: `simulateGraphImageStream()`, `simulateBranchImageStream()`
   - TODO 주석 참고

### 코드에서 TODO 찾기

프론트엔드 코드에서 `TODO: 백엔드` 또는 `TODO: 실제 백엔드`로 검색하면 백엔드 연동이 필요한 모든 위치를 찾을 수 있습니다.

## 개발 환경 설정

```bash
cd front/i2i-front
npm install
npm run dev
```

## Mock 모드 (더미 시뮬레이터)

서버를 돌릴 수 없는 상황에서 테스트하기 위해 Mock 모드를 사용할 수 있습니다.

### Mock 모드 활성화 방법

**방법 1: 환경 변수 사용 (권장)**

```bash
# .env 파일 생성 또는 수정
VITE_USE_MOCK_MODE=true

# 개발 서버 실행
npm run dev
```

**방법 2: 코드에서 직접 수정**
`front/i2i-front/src/config/api.ts` 파일을 열고:

```typescript
export const USE_MOCK_MODE = true; // false를 true로 변경
```

### Mock 모드 동작

- Mock 모드가 활성화되면 실제 서버 연결 없이 더미 데이터로 시뮬레이션합니다
- 이미지 생성, 브랜치 생성 등이 모두 시뮬레이션으로 동작합니다
- 실제 서버 코드는 그대로 유지되며, Mock 모드일 때만 시뮬레이션이 실행됩니다
- Mock 모드를 비활성화하면 다시 실제 서버 연결을 시도합니다

## 프로젝트 구조

```
front/i2i-front/
├── src/
│   ├── api/              # 백엔드 API 호출 함수들
│   │   ├── composition.ts    # 객체 리스트, 이미지 생성 시작
│   │   ├── branch.ts         # 브랜치 생성
│   │   ├── websocket.ts      # WebSocket 연결
│   │   └── feedback.ts       # 피드백 제출 (deprecated)
│   ├── components/       # React 컴포넌트들
│   │   ├── GraphCanvas.tsx   # 메인 그래프 캔버스
│   │   ├── CompositionModal.tsx  # 구도 설정 모달
│   │   ├── BranchingModal.tsx    # 브랜치 생성 모달
│   │   └── ...
│   ├── stores/           # Zustand 상태 관리
│   │   └── imageStore.ts    # 이미지 및 그래프 상태 관리
│   └── types/           # TypeScript 타입 정의
│       └── index.ts
└── BACKEND_SPEC.md      # 백엔드 스펙 문서 (프로젝트 루트)
```

## 주요 기능

1. **프롬프트 기반 이미지 생성**

   - 프롬프트 입력 → 객체 리스트 생성 → 구도 설정 → 이미지 생성

2. **인터랙티브 그래프**

   - 이미지 생성 과정을 그래프로 시각화
   - 노드 선택, 드래그, 브랜치 생성

3. **브랜치 생성**

   - 특정 이미지 노드에서 피드백을 주어 브랜치 생성
   - 여러 브랜치를 병렬로 생성 가능

4. **실시간 이미지 스트림**
   - WebSocket을 통한 diffusion step 스트리밍
   - 각 step마다 이미지 노드 생성
