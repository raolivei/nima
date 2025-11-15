import { useState, useCallback, useRef, useEffect } from "react";
import { ChatMessage, sendChatMessage, streamChatMessage } from "@/utils/api";

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const abortControllerRef = useRef<AbortController | null>(null);

  // Load conversation from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("nima_conversation");
    if (saved) {
      try {
        const data = JSON.parse(saved);
        setMessages(data.messages || []);
        setConversationId(data.conversationId);
      } catch (e) {
        // Ignore parse errors
      }
    }
  }, []);

  // Save conversation to localStorage
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem(
        "nima_conversation",
        JSON.stringify({
          messages,
          conversationId,
        })
      );
    }
  }, [messages, conversationId]);

  const sendMessage = useCallback(
    async (content: string, useStreaming: boolean = true) => {
      if (!content.trim() || isLoading) return;

      const userMessage: ChatMessage = { role: "user", content };
      const newMessages = [...messages, userMessage];
      setMessages(newMessages);
      setIsLoading(true);
      setError(null);

      try {
        if (useStreaming) {
          // Streaming response
          let assistantContent = "";
          const assistantMessage: ChatMessage = {
            role: "assistant",
            content: "",
          };
          setMessages([...newMessages, assistantMessage]);

          try {
            for await (const chunk of streamChatMessage(
              newMessages,
              conversationId
            )) {
              assistantContent += chunk;
              setMessages([
                ...newMessages,
                { ...assistantMessage, content: assistantContent },
              ]);
            }

            // Update conversation ID if provided
            // Note: In a real implementation, the stream endpoint should return the conversation_id
            setMessages([
              ...newMessages,
              { ...assistantMessage, content: assistantContent },
            ]);
          } catch (streamError: any) {
            if (streamError.name !== "AbortError") {
              throw streamError;
            }
          }
        } else {
          // Non-streaming response
          const response = await sendChatMessage(newMessages, conversationId);
          setMessages(response.messages);
          setConversationId(response.conversation_id);
        }
      } catch (err: any) {
        setError(err.message || "Failed to send message");
        setMessages(newMessages); // Remove the assistant message on error
      } finally {
        setIsLoading(false);
      }
    },
    [messages, conversationId, isLoading]
  );

  const clearConversation = useCallback(() => {
    setMessages([]);
    setConversationId(undefined);
    localStorage.removeItem("nima_conversation");
  }, []);

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  }, []);

  return {
    messages,
    isLoading,
    error,
    conversationId,
    sendMessage,
    clearConversation,
    stopGeneration,
  };
}

