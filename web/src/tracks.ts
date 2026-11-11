import type { TracksStatus } from "./api";

export const TRACK_SUFFIXES = [".zip", ".gpx", ".fit", ".tcx", ".gpx.gz", ".fit.gz", ".tcx.gz"];

export function isTrackFile(name: string): boolean {
  const lower = name.toLowerCase();
  return !lower.startsWith(".") && TRACK_SUFFIXES.some((suffix) => lower.endsWith(suffix));
}

export function formatSize(bytes: number): string {
  const kb = Math.max(1, Math.round(bytes / 1024));
  if (kb < 1024) return `${kb} КБ`;
  const mb = bytes / 1024 ** 2;
  if (mb < 1023.95) return `${mb.toFixed(1).replace(".", ",")} МБ`;
  return `${(bytes / 1024 ** 3).toFixed(1).replace(".", ",")} ГБ`;
}

export function stateText(status: TracksStatus | null): string {
  switch (status?.state) {
    case "running":
      return "Идёт обработка…";
    case "done":
      return "Готово.";
    case "failed":
      return "Обработка не удалась — смотрите журнал ниже.";
    default: {
      const waiting = status?.inbox.length ?? 0;
      // Uploaded files stay on the server and every run works through all of them, so "waiting" would mislead.
      return waiting > 0 ? "Загрузите новые файлы или обработайте заново те, что уже на сервере." : "Загрузите файлы, чтобы начать.";
    }
  }
}
