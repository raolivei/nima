import ChatInterface from "@/components/ChatInterface";
import Head from "next/head";

export default function Home() {
  return (
    <>
      <Head>
        <title>Nima Chat</title>
        <meta
          name="description"
          content="Chat with Nima - Your application-aware AI assistant"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="h-screen flex flex-col bg-white dark:bg-gray-900">
        <ChatInterface />
      </main>
    </>
  );
}

