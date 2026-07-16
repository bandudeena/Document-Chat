import { useRef, useState } from "react";
import type { DocumentInfo } from "../api";

interface Props {
  documents: DocumentInfo[];
  pendingUploads: string[]; // filenames currently being ingested, in queue order
  onUpload: (files: FileList) => void;
  onDelete: (docId: string) => void;
}

function StatLine({ stats }: { stats: DocumentInfo["stats"] }) {
  const parts = [
    `${stats.chunks} chunks`,
    stats.tables > 0 && `${stats.tables} table${stats.tables > 1 ? "s" : ""}`,
    stats.images_described > 0 &&
      `${stats.images_described} image${stats.images_described > 1 ? "s" : ""}`,
  ].filter(Boolean);
  return <span className="text-xs text-muted">{parts.join(" · ")}</span>;
}

function PendingCard({ filename, active }: { filename: string; active: boolean }) {
  return (
    <li className="rounded-lg border border-dashed border-accent/50 bg-accent-soft px-3 py-2.5">
      <div className="flex items-center gap-2">
        {active ? (
          <span
            aria-hidden
            className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-accent border-t-transparent"
          />
        ) : (
          <span aria-hidden className="h-3.5 w-3.5 shrink-0 rounded-full border-2 border-line" />
        )}
        <span className="break-all text-sm font-medium leading-snug">{filename}</span>
      </div>
      <p className="mt-1 text-xs text-muted">
        {active ? "Ingesting — parsing, describing images, embedding…" : "Waiting in queue…"}
      </p>
    </li>
  );
}

export function Sidebar({ documents, pendingUploads, onUpload, onDelete }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const uploading = pendingUploads.length > 0;

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-line bg-panel">
      <div className="border-b border-line px-5 py-4">
        <h1 className="font-display text-lg font-semibold tracking-tight">
          Chat with your docs
        </h1>
        <p className="mt-0.5 text-xs text-muted">RAG · Bedrock Haiku · Titan v2 · FAISS</p>
      </div>

      <div className="p-4">
        <button
          type="button"
          disabled={uploading}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (e.dataTransfer.files.length) onUpload(e.dataTransfer.files);
          }}
          className={`w-full rounded-lg border border-dashed px-4 py-5 text-sm transition-colors
            ${dragging ? "border-accent bg-accent-soft" : "border-line hover:border-accent hover:bg-accent-soft"}
            ${uploading ? "cursor-wait opacity-60" : "cursor-pointer"}`}
        >
          <span className="font-medium text-accent">Upload a document</span>
          <span className="mt-1 block text-xs text-muted">PDF, TXT or MD — drop it here</span>
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,.md"
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) onUpload(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4">
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-muted">
          Library ({documents.length})
        </p>
        {documents.length === 0 && !uploading && (
          <p className="text-sm text-muted">
            Nothing here yet. Upload a document to start asking questions.
          </p>
        )}
        <ul className="space-y-2">
          {pendingUploads.map((name, i) => (
            <PendingCard key={`${name}-${i}`} filename={name} active={i === 0} />
          ))}
          {documents.map((doc) => (
            <li key={doc.doc_id} className="group rounded-lg border border-line bg-bg px-3 py-2.5">
              <div className="flex items-start justify-between gap-2">
                <span className="break-all text-sm font-medium leading-snug">{doc.filename}</span>
                <button
                  type="button"
                  title={`Remove ${doc.filename}`}
                  onClick={() => onDelete(doc.doc_id)}
                  className="text-muted opacity-0 transition-opacity hover:text-red-500 group-hover:opacity-100"
                >
                  ✕
                </button>
              </div>
              <StatLine stats={doc.stats} />
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
