# Nima Chat Frontend

Modern chat interface for Nima AI assistant, built with Next.js and TypeScript.

## Features

- 💬 Real-time chat interface with streaming support
- 🎨 Dark/light theme support
- 📱 Responsive mobile design
- 💾 Conversation persistence (localStorage)
- 🎯 Markdown rendering for code blocks
- ⌨️ Keyboard shortcuts (Enter to send, Shift+Enter for newline)

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Nima API running on `http://localhost:8000` (or set `NEXT_PUBLIC_API_URL`)

### Installation

```bash
cd frontend-chat
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm start
```

## Environment Variables

- `NEXT_PUBLIC_API_URL` - API URL (default: `http://localhost:8000`)

## Project Structure

```
frontend-chat/
├── pages/
│   ├── _app.tsx          # App wrapper with theme initialization
│   └── index.tsx         # Main chat page
├── components/
│   ├── ChatInterface.tsx # Main chat container
│   ├── MessageList.tsx   # Message display component
│   ├── MessageBubble.tsx # Individual message component
│   ├── ChatInput.tsx     # Input field component
│   └── Header.tsx        # Header with clear button
├── hooks/
│   └── useChat.ts        # Chat state management hook
├── utils/
│   └── api.ts            # API client functions
└── styles/
    └── globals.css       # Global styles and Tailwind
```

## Usage

The chat interface connects to the Nima API and supports:

- **Chat messages**: Send questions and receive responses
- **Streaming**: Real-time response streaming (SSE)
- **Conversation history**: Automatically saved to localStorage
- **Clear conversation**: Use the trash icon in the header

## API Integration

The frontend uses the following endpoints:

- `POST /v1/chat` - Send chat message (non-streaming)
- `POST /v1/chat/stream` - Send chat message (streaming, SSE)

See `utils/api.ts` for implementation details.

