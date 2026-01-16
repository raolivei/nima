import { Trash2 } from "lucide-react";

interface HeaderProps {
  onClear: () => void;
}

export default function Header({ onClear }: HeaderProps) {
  return (
    <header className="border-b border-gray-200 dark:border-gray-800 px-4 py-3 flex items-center justify-between">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
          Nima Chat
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Your application-aware AI assistant
        </p>
      </div>
      <button
        onClick={onClear}
        className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-400 transition-colors"
        title="Clear conversation"
      >
        <Trash2 size={20} />
      </button>
    </header>
  );
}




