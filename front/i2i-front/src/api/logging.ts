import { API_BASE_URL } from "../config/api";
import type { LogEntry } from "../types";

/**
 * Save a log entry to the server
 * @param logEntry - LogEntry to save
 */
export async function saveLog(logEntry: LogEntry): Promise<{ status: string; message?: string }> {
  const res = await fetch(`${API_BASE_URL}/api/logs/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(logEntry),
  });
  
  if (!res.ok) {
    throw new Error(`saveLog failed: ${res.status}`);
  }
  
  return await res.json();
}

