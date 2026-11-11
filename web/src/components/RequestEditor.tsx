import { useState } from "react";

import type { Hills, ParsedRequest, Surface } from "../api";

export interface RequestEditorProps {
  /** The request as understood; the editor works on a copy. */
  request: ParsedRequest;
  onApply(request: ParsedRequest): void;
  onCancel(): void;
}

type Mode = "about" | "range";
type Slider = "avoid_major" | "forest" | "familiarity";

// The server's defaults, shown by a slider whose field was not said.
const SLIDERS: { key: Slider; label: string; usual: number }[] = [
  { key: "avoid_major", label: "Крупные дороги: не мешают ↔ избегать", usual: 0.5 },
  { key: "forest", label: "Лес: не важен ↔ только лесом", usual: 0.3 },
  { key: "familiarity", label: "Знакомые дороги: новые ↔ только знакомые", usual: 0.4 },
];

const text = (value: number | null) => (value === null ? "" : String(value));

/** Empty is "not said" (null); anything that is not a positive number is NaN. */
function parse(value: string): number | null {
  if (value.trim() === "") return null;
  const number = Number(value.replace(",", "."));
  return number > 0 ? number : NaN;
}

export function RequestEditor({ request, onApply, onCancel }: RequestEditorProps) {
  const [draft, setDraft] = useState(request);
  const [mode, setMode] = useState<Mode>(request.length_km === null && (request.length_min_km !== null || request.length_max_km !== null) ? "range" : "about");
  const [about, setAbout] = useState(text(request.length_km));
  const [low, setLow] = useState(text(request.length_min_km));
  const [high, setHigh] = useState(text(request.length_max_km));

  const change = (part: Partial<ParsedRequest>) => setDraft((current) => ({ ...current, ...part }));

  const aboutKm = parse(about);
  const lowKm = parse(low);
  const highKm = parse(high);
  const valid =
    mode === "about"
      ? !Number.isNaN(aboutKm)
      : !Number.isNaN(lowKm) && !Number.isNaN(highKm) && (lowKm === null || highKm === null || lowKm <= highKm);

  const apply = () => {
    if (!valid) return;
    onApply(
      mode === "about"
        ? { ...draft, length_km: aboutKm, length_min_km: null, length_max_km: null }
        : { ...draft, length_km: null, length_min_km: lowKm, length_max_km: highKm },
    );
  };

  return (
    <form
      className="editor"
      onSubmit={(event) => {
        event.preventDefault();
        apply();
      }}
    >
      <div className="editor-field">
        <label htmlFor="editor-mode">Длина</label>
        <div className="editor-row">
          <select id="editor-mode" className="field" value={mode} onChange={(event) => setMode(event.target.value as Mode)}>
            <option value="about">около</option>
            <option value="range">от–до</option>
          </select>
          {mode === "about" ? (
            <input className="field" type="number" min="0" step="any" aria-label="Длина, км" value={about} onChange={(event) => setAbout(event.target.value)} />
          ) : (
            <>
              <input className="field" type="number" min="0" step="any" aria-label="От, км" placeholder="от" value={low} onChange={(event) => setLow(event.target.value)} />
              <input className="field" type="number" min="0" step="any" aria-label="До, км" placeholder="до" value={high} onChange={(event) => setHigh(event.target.value)} />
            </>
          )}
          <span className="editor-unit">км</span>
        </div>
      </div>

      <div className="editor-columns">
        <div className="editor-field">
          <label htmlFor="editor-surface">Покрытие</label>
          <select id="editor-surface" className="field" value={draft.surface ?? "any"} onChange={(event) => change({ surface: event.target.value as Surface })}>
            <option value="any">любое</option>
            <option value="paved">асфальт</option>
            <option value="prefer_unpaved">грунт</option>
          </select>
        </div>
        <div className="editor-field">
          <label htmlFor="editor-hills">Рельеф</label>
          <select id="editor-hills" className="field" value={draft.hills ?? "any"} onChange={(event) => change({ hills: event.target.value as Hills })}>
            <option value="any">любой</option>
            <option value="avoid">поровнее</option>
            <option value="seek">с подъёмами</option>
          </select>
        </div>
      </div>

      {SLIDERS.map(({ key, label, usual }) => (
        <div key={key} className="editor-field">
          <label htmlFor={`editor-${key}`}>{label}</label>
          <div className="editor-row">
            <input
              id={`editor-${key}`}
              className="editor-slider"
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={draft[key] ?? usual}
              onChange={(event) => change({ [key]: Number(event.target.value) })}
            />
            <span className="editor-value">{(draft[key] ?? usual).toFixed(1).replace(".", ",")}</span>
          </div>
        </div>
      ))}

      <label className="editor-check">
        <input
          type="checkbox"
          checked={draft.via_water === true}
          disabled={request.via !== null}
          onChange={(event) => change({ via_water: event.target.checked ? true : null })}
        />
        К воде
      </label>

      <div className="editor-actions">
        <button type="button" className="btn" onClick={onCancel}>
          Отмена
        </button>
        <button type="submit" className="btn btn-primary" disabled={!valid}>
          Применить
        </button>
      </div>
    </form>
  );
}
