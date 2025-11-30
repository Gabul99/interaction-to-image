import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // 원격 서버에서 실행 시 외부 접속 허용
    port: 5173, // 원격 서버에서 실행 시 포트 고정
    strictPort: true, // 포트가 사용 중이면 에러 발생 (자동 변경 방지)
  },
})
