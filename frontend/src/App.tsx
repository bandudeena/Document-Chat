import { useCallback, useEffect, useState } from "react";
import {
  type ChatTurn,
  type DocumentInfo,
  deleteDocument,
  listDocuments,
  streamChat,
  uploadDocument,
} from "./api";
import { Chat } from "./components/Chat";
import type { ChatMessage } from "./components/Message";
import { Sidebar } from "./components/Sidebar";

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pendingUploads, setPendingUploads] = useState<string[]>([]);
  const [answering, setAnswering] = useState(false);
  const [banner, setBanner] = useState<string | null>(null);

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
    } catch (error) {
      setBanner(`Could not load documents: ${(error as Error).message}`);
    }
  }, []);

  useEffect(() => {
    void refreshDocuments();
  }, [refreshDocuments]);

  const showError = (message: string) => {
    setBanner(message);
    setTimeout(() => setBanner(null), 6000);
  };

  const handleUpload = async (files: FileList) => {
    const queue = Array.from(files);
    setPendingUploads(queue.map((f) => f.name));
    // Sequential on purpose: each finished file appears in the library
    // immediately, and the queue shows what's processing vs waiting.
    for (const file of queue) {
      try {
        await uploadDocument(file);
        await refreshDocuments();
      } catch (error) {
        showError(`Upload of ${file.name} failed: ${(error as Error).message}`);
      } finally {
        setPendingUploads((prev) => prev.slice(1));
      }
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await deleteDocument(docId);
      await refreshDocuments();
    } catch (error) {
      showError(`Delete failed: ${(error as Error).message}`);
    }
  };

  const handleAsk = (question: string) => {
    const history: ChatTurn[] = messages
      .filter((m) => !m.error && m.content)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", streaming: true },
    ]);
    setAnswering(true);

    const patchLast = (patch: Partial<ChatMessage> | ((m: ChatMessage) => Partial<ChatMessage>)) =>
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        next[next.length - 1] = {
          ...last,
          ...(typeof patch === "function" ? patch(last) : patch),
        };
        return next;
      });

    void streamChat(question, history, {
      onSources: (sources) => patchLast({ sources }),
      onSearching: (query) =>
        patchLast((m) => ({ searches: [...(m.searches ?? []), query] })),
      onToken: (text) => patchLast((m) => ({ content: m.content + text })),
      onDone: (usage) => {
        patchLast({ usage, streaming: false });
        setAnswering(false);
      },
      onError: (message) => {
        patchLast({
          content: `Sorry — ${message}`,
          streaming: false,
          error: true,
        });
        setAnswering(false);
      },
    }).catch((error) => {
      patchLast({ content: `Sorry — ${(error as Error).message}`, streaming: false, error: true });
      setAnswering(false);
    });
  };

  return (
    <div className="flex h-full">
      <Sidebar
        documents={documents}
        pendingUploads={pendingUploads}
        onUpload={handleUpload}
        onDelete={handleDelete}
      />
      <div className="relative flex min-w-0 flex-1 flex-col">
        {banner && (
          <div className="absolute inset-x-0 top-0 z-10 bg-red-500/90 px-4 py-2 text-center text-sm text-white">
            {banner}
          </div>
        )}
        <Chat
          messages={messages}
          busy={answering}
          hasDocuments={documents.length > 0}
          onAsk={handleAsk}
          onNewChat={() => setMessages([])}
        />
      </div>
    </div>
  );
}
