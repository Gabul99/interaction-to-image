# I2I Frontend

이미지 생성 프론트엔드 애플리케이션입니다.

## 로컬에서 실행 (권장)

로컬 Mac에서 프론트엔드를 실행하고 원격 서버의 백엔드와 통신하는 방법:

### 1. 원격 서버에서 백엔드 실행

```bash
ssh gpu10
conda activate hci_i2i
cd /home/ella/courses/HCI/interaction-to-image/back
python -m back.main
```

### 2. 로컬 Mac에서 SSH 터널링 설정

```bash
# 백엔드 포트(8000) 포워딩
ssh -N -L 8000:localhost:8000 gpu10
```

### 3. 로컬 Mac에서 프론트엔드 실행

```bash
cd /path/to/interaction-to-image/front/i2i-front

# .env 파일 생성
echo "VITE_API_BASE_URL=http://localhost:8000" > .env
echo "VITE_WS_BASE_URL=ws://localhost:8000" >> .env

# 의존성 설치 (최초 1회만)
npm install

# 개발 서버 실행
npm run dev
```

### 4. 브라우저 접속

```
http://localhost:5173
```

## 원격 서버에서 실행

원격 서버에서 프론트엔드를 실행하는 경우:

```bash
ssh gpu10
cd /home/ella/courses/HCI/interaction-to-image/front/i2i-front
npm install
npm run dev
```

그리고 로컬에서 SSH 터널링:
```bash
ssh -N -L 5173:localhost:5173 gpu10
```

## 환경 변수

프로젝트 루트에 `.env` 파일을 생성하여 설정:

```bash
# 로컬 프론트엔드 + 원격 백엔드 (SSH 터널링)
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000

# 다른 포트 사용 시
# VITE_API_BASE_URL=http://localhost:8001
# VITE_WS_BASE_URL=ws://localhost:8001
```

---

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
