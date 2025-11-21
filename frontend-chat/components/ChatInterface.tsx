import { useChat } from "@/hooks/useChat";
import MessageList from "./MessageList";
import ChatInput from "./ChatInput";
import Header from "./Header";

export default function ChatInterface() {
  const { messages, isLoading, error, sendMessage, clearConversation } =
    useChat();

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto w-full">
      <Header onClear={clearConversation} />

      <div className="flex-1 overflow-hidden flex flex-col">
        <MessageList messages={messages} isLoading={isLoading} />

        {error && (
          <div className="px-4 py-2 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 text-sm">
            Error: {error}
          </div>
        )}

        <ChatInput onSend={sendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
}



