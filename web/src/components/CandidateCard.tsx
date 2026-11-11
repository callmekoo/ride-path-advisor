import type { CSSProperties, KeyboardEvent } from "react";

import type { CandidateBody } from "../api";

export interface CandidateCardProps {
  heading: string; // "Вариант 2", or the name of a saved route
  route: CandidateBody;
  colour: string; // the colour of its line on the map
  selected: boolean;
  gpxUrl: string;
  canSave: boolean; // false for a route that is already a saved one
  saved: boolean; // it has just been saved
  onSelect(): void;
  onSave(): void;
}

/** Kilometres with a decimal comma and one decimal: "44,8 км". */
export const formatKm = (km: number) => `${km.toFixed(1).replace(".", ",")} км`;

const percent = (share: number) => `${Math.round(share * 100)}%`;

const SHARES: [string, string][] = [
  ["forest_share", "лес"],
  ["ridden_share", "знакомые дороги"],
  ["major_share", "крупные дороги"],
  ["unpaved_share", "грунт"],
];

export function CandidateCard({ heading, route, colour, selected, gpxUrl, canSave, saved, onSelect, onSave }: CandidateCardProps) {
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) return; // keys belong to the inner buttons and link
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect();
    }
  };
  const shares = SHARES.flatMap(([key, label]) => {
    const value = route.features[key];
    return value === undefined ? [] : [[label, value] as const];
  });

  return (
    <div
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      className={selected ? "card card-selected" : "card"}
      style={{ "--route-colour": colour } as CSSProperties}
      onClick={onSelect}
      onKeyDown={onKeyDown}
    >
      <div className="card-header">
        <span className="card-index">{heading}</span>
        {route.via_label && <span className="card-title">{route.via_label}</span>}
      </div>
      <div className="card-numbers">
        <span className="card-length">{formatKm(route.length_km)}</span>
        {route.ascent_m !== null && <span className="card-ascent">↑ {Math.round(route.ascent_m)} м</span>}
      </div>
      {shares.length > 0 && (
        <dl className="card-shares">
          {shares.map(([label, value]) => (
            <div key={label} className="card-share">
              <dt>{label}</dt>
              <dd>{percent(value)}</dd>
            </div>
          ))}
        </dl>
      )}
      {route.note && <p className="card-note">{route.note}</p>}
      <div className="card-actions" onClick={(event) => event.stopPropagation()}>
        <a className="btn btn-small" href={gpxUrl} download>
          GPX
        </a>
        {canSave && (
          <button type="button" className="btn btn-small" disabled={saved} onClick={onSave}>
            {saved ? "Сохранён" : "Сохранить"}
          </button>
        )}
      </div>
    </div>
  );
}
