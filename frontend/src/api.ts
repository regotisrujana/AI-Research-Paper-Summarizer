import type { ChatResponse, CorpusStatus, EvaluationQuestion, EvaluationResponse, Message, Paper } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function parseResponse<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail ?? "Request failed");
  }
  return body as T;
}

export async function fetchPapers(): Promise<Paper[]> {
  return parseResponse<Paper[]>(await fetch(`${API_BASE}/papers`));
}

export async function fetchCorpusStatus(): Promise<CorpusStatus> {
  return parseResponse<CorpusStatus>(await fetch(`${API_BASE}/corpus/status`));
}

export async function fetchPaperHistory(paperId: number | undefined, sessionId: string): Promise<Message[]> {
  const params = new URLSearchParams({ session_id: sessionId });
  if (!paperId) {
    const rows = await parseResponse<Array<{ id: number; role: "user" | "assistant"; content: string; created_at: string }>>(
      await fetch(`${API_BASE}/chat/history?${params}`)
    );
    return rows.map((row) => ({ id: row.id, role: row.role, content: row.content, created_at: row.created_at }));
  }
  const rows = await parseResponse<Array<{ id: number; role: "user" | "assistant"; content: string; created_at: string }>>(
    await fetch(`${API_BASE}/papers/${paperId}/history?${params}`)
  );
  return rows.map((row) => ({ id: row.id, role: row.role, content: row.content, created_at: row.created_at }));
}

export async function uploadPaper(file: File, onProgress: (value: number) => void): Promise<Paper> {
  const formData = new FormData();
  formData.append("file", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/upload`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.onload = () => {
      try {
        const parsed = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(parsed as Paper);
        else reject(new Error(parsed.detail ?? "Upload failed"));
      } catch (error) {
        reject(error);
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed"));
    xhr.send(formData);
  });
}

export async function uploadPapers(files: File[], onProgress: (value: number) => void): Promise<Paper[]> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/upload/batch`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.onload = () => {
      try {
        const parsed = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(parsed as Paper[]);
        else reject(new Error(parsed.detail ?? "Batch upload failed"));
      } catch (error) {
        reject(error);
      }
    };
    xhr.onerror = () => reject(new Error("Batch upload failed"));
    xhr.send(formData);
  });
}

export async function askQuestion(question: string, paperId?: number, sessionId = "default"): Promise<ChatResponse> {
  return parseResponse<ChatResponse>(
    await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, paper_id: paperId, session_id: sessionId }),
    })
  );
}

export async function fetchSummary(paperId: number): Promise<{ summary: string; cached: boolean }> {
  return parseResponse<{ summary: string; cached: boolean }>(await fetch(`${API_BASE}/papers/${paperId}/summary`));
}

export async function generateSummary(paperId: number, force = false): Promise<{ summary: string; cached: boolean }> {
  return parseResponse<{ summary: string; cached: boolean }>(
    await fetch(`${API_BASE}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper_id: paperId, force }),
    })
  );
}

export async function deletePaper(paperId: number): Promise<void> {
  await parseResponse(await fetch(`${API_BASE}/papers/${paperId}`, { method: "DELETE" }));
}

export async function clearPaperHistory(paperId: number, sessionId: string): Promise<void> {
  const params = new URLSearchParams({ session_id: sessionId });
  await parseResponse(await fetch(`${API_BASE}/papers/${paperId}/history?${params}`, { method: "DELETE" }));
}

export async function evaluateCorpus(questions: EvaluationQuestion[], sessionId: string): Promise<EvaluationResponse> {
  return parseResponse<EvaluationResponse>(
    await fetch(`${API_BASE}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ questions, session_id: sessionId }),
    })
  );
}
