import type { KeyboardEvent, ReactNode } from "react";

import type { SavedRoute } from "../api";
import { formatKm } from "./CandidateCard";
import { DeleteButton } from "./PlacesPanel";

export interface SavedRoutesPanelProps {
  routes: SavedRoute[]; // newest first
  openId: number | null; // the one shown on the map
  onOpen(id: number): void;
  onDelete(id: number): void;
  /** The card and the profile of the opened route. */
  children?: ReactNode;
}

/** "2026-11-10 08:30:00" (UTC) → "10.11.2026". */
const date = (created: string) => created.slice(0, 10).split("-").reverse().join(".");

export function SavedRoutesPanel({ routes, openId, onOpen, onDelete, children }: SavedRoutesPanelProps) {
  const onKeyDown = (event: KeyboardEvent<HTMLLIElement>, id: number) => {
    if (event.target !== event.currentTarget) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onOpen(id);
    }
  };

  return (
    <div className="saved">
      {routes.length === 0 ? (
        <p className="saved-empty">Сохранённых маршрутов пока нет. Кнопка „Сохранить“ есть на карточке каждого варианта.</p>
      ) : (
        <ul className="saved-list">
          {routes.map((route) => (
            <li
              key={route.id}
              role="button"
              tabIndex={0}
              className={route.id === openId ? "saved-item saved-item-open" : "saved-item"}
              onClick={() => onOpen(route.id)}
              onKeyDown={(event) => onKeyDown(event, route.id)}
            >
              <div className="saved-main">
                <span className="saved-name">{route.name}</span>
                <span className="saved-meta">
                  {formatKm(route.length_km)}
                  {route.ascent_m !== null && ` · ↑ ${Math.round(route.ascent_m)} м`} · {date(route.created)}
                </span>
              </div>
              <DeleteButton onDelete={() => onDelete(route.id)} />
            </li>
          ))}
        </ul>
      )}
      {children}
    </div>
  );
}
