import { useEffect, useState } from "react";
import type { ApiState } from "./types";

export const API_BASE_URL = "/api";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function errorMessage(detail: unknown, status: number): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (typeof item === "object" && item !== null && "msg" in item) return String(item.msg);
      return typeof item === "string" ? item : "";
    }).filter(Boolean);
    if (messages.length) return messages.join(" ");
  }
  if (status === 404) return "Не знайдено запитаний ресурс.";
  if (status >= 500) return "Сервер тимчасово недоступний. Спробуйте пізніше.";
  return "Не вдалося виконати запит. Перевірте дані та спробуйте ще раз.";
}

export async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
  let response: Response;
  const accessToken = localStorage.getItem("petmind_access_token") || "";
  try {
    response = await fetch(url, {
      ...init,
      headers: { Accept: "application/json", ...(init.body ? { "Content-Type": "application/json" } : {}), ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}), ...init.headers },
    });
  } catch {
    throw new ApiError("Немає з’єднання з сервером. Перевірте, чи запущений PetMind API.", 0);
  }

  const contentType = response.headers.get("content-type") || "";
  const body: unknown = contentType.includes("application/json") ? await response.json().catch(() => null) : await response.text().catch(() => "");
  if (!response.ok) {
    const detail = typeof body === "object" && body !== null && "detail" in body ? body.detail : body;
    throw new ApiError(errorMessage(detail, response.status), response.status);
  }
  return body as T;
}

export function useApiResource<T>(path: string | null, refreshKey: string | number = ""): ApiState<T> & { reload: () => void } {
  const [state, setState] = useState<ApiState<T>>({ data: null, loading: Boolean(path), error: null });
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    if (!path) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    const controller = new AbortController();
    setState({ data: null, loading: true, error: null });
    apiRequest<T>(path, { signal: controller.signal })
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setState((previous) => ({ ...previous, loading: false, error: error instanceof Error ? error.message : "Сталася невідома помилка." }));
      });
    return () => controller.abort();
  }, [path, refreshKey, revision]);

  return {
    ...state,
    reload: () => {
      setState((previous) => ({ ...previous, loading: true, error: null }));
      setRevision((value) => value + 1);
    },
  };
}
