/** Typed API client mirroring backend/app/schemas.py, plus an SSE reader for /api/chat. */

export interface IngestStats {
  pages: number;
  chunks: number;
  text_chunks: number;
  tables: number;
  images_described: number;
  images_skipped: number;
}

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  created_at: string;
  stats: IngestStats;
}

export interface Source {
  ref: number;
  doc_id: string;
  filename: string;
  page: number | null;
  content_type: "text" | "table" | "image";
  snippet: string;
  score: number;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  estimated_cost_usd: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

async function expectOk(response: Response): Promise<Response> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return response;
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  return (await expectOk(await fetch("/api/documents"))).json();
}

export async function uploadDocument(file: File): Promise<DocumentInfo> {
  const form = new FormData();
  form.append("file", file);
  return (await expectOk(await fetch("/api/documents", { method: "POST", body: form }))).json();
}

export async function deleteDocument(docId: string): Promise<void> {
  await expectOk(await fetch(`/api/documents/${docId}`, { method: "DELETE" }));
}

export interface ChatCallbacks {
  onSources: (sources: Source[]) => void;
  onToken: (text: string) => void;
  onSearching: (query: string) => void;
  onDone: (usage: Usage) => void;
  onError: (message: string) => void;
}

/** POST /api/chat and dispatch the SSE events (sources -> token* -> done). */
export async function streamChat(
  question: string,
  history: ChatTurn[],
  callbacks: ChatCallbacks,
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
  if (!response.ok || !response.body) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    callbacks.onError(detail);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (block: string) => {
    let event = "";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7).trim();
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (!event || !data) return;
    const payload = JSON.parse(data);
    if (event === "sources") callbacks.onSources(payload.sources);
    else if (event === "token") callbacks.onToken(payload.text);
    else if (event === "searching") callbacks.onSearching(payload.query);
    else if (event === "done") callbacks.onDone(payload.usage);
    else if (event === "error") callbacks.onError(payload.message);
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      dispatch(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
    }
  }
}
