import { fileURLToPath, URL } from "node:url"
import { defineConfig, loadEnv } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

const projectRoot = fileURLToPath(new URL("..", import.meta.url))

export default defineConfig(({ mode }) => {
  // API_PORT lives in the repository-level .env shared with the backend and scripts.
  const env = { ...loadEnv(mode, projectRoot, ""), ...process.env }
  const apiTarget = env.PETMIND_API_PROXY || `http://127.0.0.1:${env.API_PORT || "8000"}`

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        "@": fileURLToPath(new URL("./src", import.meta.url)),
      },
    },
    server: {
      host: "0.0.0.0",
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": apiTarget,
      },
    },
    preview: {
      host: "0.0.0.0",
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": apiTarget,
      },
    },
  }
})
