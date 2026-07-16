import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source, Usage } from "../api";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  usage?: Usage;
  searches?: string[]; // queries the agent ran while answering
  streaming?: boolean;
  error?: boolean;
}

const TYPE_LABEL: Record<Source["content_type"], string> = {
  text: "text",
  table: "table",
  image: "image",
};

function SourceChips({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState<number | null>(null);
  if (sources.length === 0) return null;
  return (
    <div className="mt-3 border-t border-line pt-2">
      <div className="flex flex-wrap gap-1.5">
        {sources.map((source) => (
          <button
            key={source.ref}
            type="button"
            onClick={() => setOpen(open === source.ref ? null : source.ref)}
            className={`rounded-full border px-2.5 py-0.5 text-xs transition-colors
              ${open === source.ref ? "border-accent bg-accent-soft text-accent" : "border-line text-muted hover:border-accent hover:text-accent"}`}
          >
            [{source.ref}] {source.filename}
            {source.page ? ` · p.${source.page}` : ""} · {TYPE_LABEL[source.content_type]}
          </button>
        ))}
      </div>
      {open !== null && (
        <div className="mt-2 rounded-lg bg-accent-soft p-3 text-xs leading-relaxed">
          <p className="mb-1 font-semibold text-accent">
            Retrieved chunk (similarity {sources.find((s) => s.ref === open)?.score.toFixed(2)})
          </p>
          <pre className="whitespace-pre-wrap font-body">
            {sources.find((s) => s.ref === open)?.snippet}
          </pre>
        </div>
      )}
    </div>
  );
}

export function Message({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-user-bubble px-4 py-2.5 text-[15px]">
          {message.content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex">
      <div
        className={`max-w-[85%] rounded-2xl rounded-bl-sm border border-line bg-panel px-4 py-3 text-[15px] leading-relaxed
          ${message.error ? "border-red-400" : ""}`}
      >
        {message.searches && message.searches.length > 0 && (
          <div className="mb-2 space-y-0.5">
            {message.searches.map((query, i) => (
              <p key={i} className="text-xs italic text-muted">
                ⌕ searched: “{query}”
              </p>
            ))}
          </div>
        )}
        {message.content ? (
          <div className="prose-answer overflow-x-auto">
            <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
          </div>
        ) : (
          <span className="inline-block animate-pulse text-muted">
            {message.searches?.length ? "Searching your documents…" : "Reading your documents…"}
          </span>
        )}
        {message.streaming && message.content && (
          <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-accent align-text-bottom" />
        )}
        {!message.streaming && message.sources && <SourceChips sources={message.sources} />}
        {!message.streaming && message.usage && message.usage.output_tokens > 0 && (
          <p className="mt-2 text-[11px] text-muted">
            {message.usage.input_tokens}→{message.usage.output_tokens} tokens · $
            {message.usage.estimated_cost_usd.toFixed(5)}
          </p>
        )}
      </div>
    </div>
  );
}
