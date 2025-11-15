const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== "undefined" ? "/api" : "http://localhost:8000");

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  conversation_id?: string;
  max_length?: number;
  temperature?: number;
  top_k?: number;
  stream?: boolean;
}

export interface ChatResponse {
  response: string;
  messages: ChatMessage[];
  conversation_id: string;
}

export async function sendChatMessage(
  messages: ChatMessage[],
  conversationId?: string,
  options?: {
    max_length?: number;
    temperature?: number;
    top_k?: number;
  }
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messages,
      conversation_id: conversationId,
      max_length: options?.max_length || 300,
      temperature: options?.temperature || 0.8,
      top_k: options?.top_k || 50,
      stream: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return response.json();
}

export async function* streamChatMessage(
  messages: ChatMessage[],
  conversationId?: string,
  options?: {
    max_length?: number;
    temperature?: number;
    top_k?: number;
  }
): AsyncGenerator<string, void, unknown> {
  const response = await fetch(`${API_URL}/v1/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messages,
      conversation_id: conversationId,
      max_length: options?.max_length || 300,
      temperature: options?.temperature || 0.8,
      top_k: options?.top_k || 50,
      stream: true,
    }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error("No response body");
  }

  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();

    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          if (data.content) {
            yield data.content;
          }
          if (data.done) {
            return;
          }
        } catch (e) {
          // Skip invalid JSON
        }
      }
    }
  }
}
