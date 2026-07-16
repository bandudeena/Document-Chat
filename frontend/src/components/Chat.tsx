import { useEffect, useRef, useState } from "react";
import { Message, type ChatMessage } from "./Message";

interface Props {
  messages: ChatMessage[];
  busy: boolean;
  hasDocuments: boolean;
  onAsk: (question: string) => void;
  onNewChat: () => void;
}

const SUGGESTIONS = [
  "Summarise the key points of my documents",
  "What data is in the tables?",
  "What do the images show?",
];

export function Chat({ messages, busy, hasDocuments, onAsk, onNewChat }: Props) {
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  const submit = () => {
    const question = draft.trim();
    if (!question || busy) return;
    setDraft("");
    onAsk(question);
  };

  return (
    <main className="flex min-w-0 flex-1 flex-col">
      {messages.length > 0 && (
        <div className="flex justify-end border-b border-line bg-panel px-6 py-2">
          <button
            type="button"
            onClick={onNewChat}
            disabled={busy}
            title="Clear the conversation and start fresh"
            className="rounded-lg border border-line px-3 py-1.5 text-sm text-muted transition-colors hover:border-accent hover:text-accent disabled:opacity-40"
          >
            + New chat
          </button>
        </div>
      )}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-4 px-6 py-8">
          {messages.length === 0 && (
            <div className="mt-16 text-center">
              <p className="font-display text-2xl text-muted">
                {hasDocuments
                  ? "Ask anything about your documents."
                  : "Upload a document, then ask away."}
              </p>
              {hasDocuments && (
                <div className="mt-6 flex flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => onAsk(s)}
                      className="rounded-full border border-line bg-panel px-4 py-1.5 text-sm text-muted transition-colors hover:border-accent hover:text-accent"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          {messages.map((message, i) => (
            <Message key={i} message={message} />
          ))}
        </div>
      </div>

      <div className="border-t border-line bg-panel">
        <form
          className="mx-auto flex max-w-3xl items-end gap-2 px-6 py-4"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <textarea
            value={draft}
            rows={1}
            maxLength={2000}
            placeholder={hasDocuments ? "Ask a question…" : "Upload a document first…"}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            className="max-h-40 min-h-[44px] flex-1 resize-y rounded-xl border border-line bg-bg px-4 py-2.5 text-[15px] outline-none placeholder:text-muted focus:border-accent"
          />
          <button
            type="submit"
            disabled={busy || !draft.trim()}
            className="rounded-xl bg-accent px-5 py-2.5 text-[15px] font-medium text-white transition-opacity disabled:opacity-40 dark:text-black"
          >
            {busy ? "…" : "Ask"}
          </button>
        </form>
      </div>
    </main>
  );
}
