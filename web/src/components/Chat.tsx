import { type KeyboardEvent, type ReactNode, useEffect, useRef, useState } from "react";

import type { Question } from "../api";

/** One line of the conversation. */
export interface Entry {
  id: number;
  role: "rider" | "service" | "error";
  text: string;
  notes: string[]; // warnings that came with a service answer
}

export interface ChatProps {
  entries: Entry[];
  /** The question waiting for an answer: its options are offered as buttons. */
  question: Question | null;
  /** A request is on its way: no input meanwhile, and a sign that work is going on. */
  busy: boolean;
  /** False when the server has no language model: free text cannot be understood. */
  canParse: boolean;
  /** There is something to start over from: show "Новый маршрут". */
  hasResult: boolean;
  onSay(text: string): void;
  onOption(index: number): void;
  onReset(): void;
  /** Chips, route cards and the profile: shown under the conversation, scrolling with it. */
  children?: ReactNode;
}

const EXAMPLES = [
  "кольцо около 45 км к воде, желательно по лесу",
  "80 км от Дидинского тоннеля до Екатеринбурга, поменьше дорог",
  "шоссейный заезд на 100–150 км, кольцевой",
];

export function Chat({ entries, question, busy, canParse, hasResult, onSay, onOption, onReset, children }: ChatProps) {
  const [draft, setDraft] = useState("");
  const input = useRef<HTMLTextAreaElement>(null);
  const lastEntry = useRef<HTMLDivElement>(null);
  const busyLine = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // "nearest": a long list of cards under the answer must not drag the view past it.
    (busyLine.current ?? lastEntry.current)?.scrollIntoView({ block: "nearest" });
  }, [entries.length, busy]);

  useEffect(() => {
    if (!busy && canParse) input.current?.focus();
  }, [busy, canParse]);

  const send = () => {
    const text = draft.trim();
    if (!text || busy || !canParse) return;
    setDraft("");
    onSay(text);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      send();
    }
  };

  const options = question?.options ?? [];
  const placeholder = question
    ? "Ваш ответ…"
    : hasResult
      ? "Уточните: короче, не через Ревду…"
      : "Например: кольцо 60 км по лесу";

  return (
    <div className="chat">
      <div className="chat-scroll">
        {entries.length === 0 && !busy && (
          <div className="chat-hint">
            {canParse ? (
              <>
                <p className="chat-hint-title">Опишите маршрут словами</p>
                {EXAMPLES.map((example) => (
                  <button key={example} type="button" className="chat-example" onClick={() => onSay(example)}>
                    {example}
                  </button>
                ))}
                <p className="chat-muted">Старт — место по умолчанию или клик по карте.</p>
              </>
            ) : (
              <p className="chat-notice">Разбор текста не настроен: задайте LLM_BASE_URL, LLM_MODEL и LLM_API_KEY на сервере.</p>
            )}
          </div>
        )}

        {entries.map((entry, index) => (
          <div key={entry.id} ref={index === entries.length - 1 ? lastEntry : undefined} className={`chat-entry chat-${entry.role}`}>
            <div className="chat-bubble">
              {entry.text}
              {entry.notes.length > 0 && (
                <ul className="chat-notes">
                  {entry.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        ))}

        {options.length > 0 && (
          <div className="chat-options">
            {options.map((option, index) => (
              <button key={index} type="button" className="chat-option" disabled={busy} onClick={() => onOption(index)}>
                <span className="chat-option-number">{index + 1}</span>
                {option.label}
              </button>
            ))}
          </div>
        )}
        {question && options.length === 0 && question.about !== "length" && <p className="chat-muted">Можно кликнуть по карте.</p>}

        {busy && (
          <div ref={busyLine} className="chat-busy">
            Думаю…
            <span className="chat-dots" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
          </div>
        )}

        {children}
      </div>

      <form
        className="chat-composer"
        onSubmit={(event) => {
          event.preventDefault();
          send();
        }}
      >
        <textarea
          ref={input}
          className="chat-input"
          rows={2}
          value={draft}
          placeholder={placeholder}
          aria-label="Запрос"
          disabled={busy || !canParse}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="chat-actions">
          {hasResult && (
            <button type="button" className="btn" disabled={busy} onClick={onReset}>
              Новый маршрут
            </button>
          )}
          <button type="submit" className="btn btn-primary" disabled={busy || !canParse || draft.trim() === ""}>
            Отправить
          </button>
        </div>
      </form>
    </div>
  );
}
