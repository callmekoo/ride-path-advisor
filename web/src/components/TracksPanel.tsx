import { useEffect, useRef, useState } from "react";

import type { TracksStatus } from "../api";
import { formatSize, isTrackFile, stateText } from "../tracks";

export interface TracksPanelProps {
  status: TracksStatus | null; // null until first loaded
  /** Km of distinct roads ridden, from meta; null when there is no history. */
  riddenKm: number | null;
  /** What is going up right now, e.g. "2 из 3: export.zip"; null when nothing is uploading. */
  uploading: string | null;
  /** Why the last action was refused, in Russian. */
  error: string | null;
  /** Upload these files one by one, then start processing. */
  onUpload(files: File[]): void;
  /** Process again what is already on the server. */
  onImport(): void;
}

export function TracksPanel({ status, riddenKm, uploading, error, onUpload, onImport }: TracksPanelProps) {
  const [chosen, setChosen] = useState<File[]>([]);
  const [pickerKey, setPickerKey] = useState(0); // a new key empties the file input
  const log = useRef<HTMLPreElement>(null);

  const running = status?.state === "running";
  const acceptable = chosen.filter((file) => isTrackFile(file.name));
  const waiting = status?.inbox.length ?? 0;

  useEffect(() => {
    if (log.current) log.current.scrollTop = log.current.scrollHeight;
  }, [status?.log]);

  const upload = () => {
    onUpload(acceptable);
    setChosen([]);
    setPickerKey((key) => key + 1);
  };

  return (
    <div className="tracks">
      <p className="tracks-muted">
        Здесь добавляются заезды: архив Strava («Настройки → Мой аккаунт → Скачать или удалить аккаунт → Запросить
        архив», придёт на почту) или отдельные файлы GPX, FIT, TCX. Маршруты будут охотнее идти по дорогам, где вы уже
        ездили.
      </p>

      {riddenKm !== null && <p>Сейчас в истории: {Math.round(riddenKm)} км дорог.</p>}

      <div className="tracks-picker">
        <label className="tracks-label" htmlFor="track-files">
          Файлы
        </label>
        <input
          key={pickerKey}
          id="track-files"
          type="file"
          multiple
          accept=".zip,.gpx,.fit,.tcx,.gz"
          onChange={(event) => setChosen(Array.from(event.target.files ?? []))}
        />
        {chosen.length > 0 && (
          <ul className="tracks-files">
            {chosen.map((file, index) => (
              <li key={index} className={isTrackFile(file.name) ? "tracks-file" : "tracks-file tracks-file-bad"}>
                <span className="tracks-file-name">{file.name}</span>
                <span className="tracks-file-size">{isTrackFile(file.name) ? formatSize(file.size) : "не трек"}</span>
              </li>
            ))}
          </ul>
        )}
        <button
          type="button"
          className="btn btn-primary"
          disabled={acceptable.length === 0 || uploading !== null || running}
          onClick={upload}
        >
          Загрузить и обработать
        </button>
      </div>

      {waiting > 0 && !running && (
        <div className="tracks-waiting">
          <span>Файлов на сервере: {waiting}</span>
          <button type="button" className="btn" disabled={uploading !== null} onClick={onImport}>
            Обработать заново
          </button>
        </div>
      )}

      <div className="tracks-status">
        <p>{stateText(status)}</p>
        {uploading !== null && <p className="tracks-muted">Загружаю {uploading}…</p>}
        {running && (
          <div className="chat-busy">
            <span className="chat-dots" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
          </div>
        )}
      </div>

      {status && status.log.length > 0 && (
        <div className="tracks-log">
          <span className="tracks-label">Журнал</span>
          <pre ref={log}>{status.log.join("\n")}</pre>
        </div>
      )}

      {error && <p className="tracks-error">{error}</p>}

      <p className="tracks-muted">Обработка занимает одну–две минуты. Файлы остаются на вашем сервере.</p>
    </div>
  );
}
