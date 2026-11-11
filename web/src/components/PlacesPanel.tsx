import { useEffect, useRef, useState } from "react";

import type { Place, Point } from "../api";

export interface PlacesPanelProps {
  places: Place[];
  /** Where the start marker stands, if the rider has put it: a new place is saved there. */
  marker: Point | null;
  /** Why the last change was refused, in Russian. */
  error: string | null;
  onAdd(name: string): void;
  onMakeDefault(id: number): void;
  /** Put the start marker on this place. */
  onStartFrom(place: Place): void;
  onDelete(id: number): void;
}

/** Deletes on the second click; a click anywhere else takes the first one back. */
export function DeleteButton({ onDelete }: { onDelete(): void }) {
  const [armed, setArmed] = useState(false);
  const button = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!armed) return;
    const reset = (event: PointerEvent) => {
      if (!button.current?.contains(event.target as Node)) setArmed(false);
    };
    document.addEventListener("pointerdown", reset);
    return () => document.removeEventListener("pointerdown", reset);
  }, [armed]);

  return (
    <button
      ref={button}
      type="button"
      className={armed ? "btn btn-small btn-danger" : "btn btn-small"}
      onClick={(event) => {
        event.stopPropagation();
        if (armed) onDelete();
        else setArmed(true);
      }}
    >
      {armed ? "Точно удалить?" : "Удалить"}
    </button>
  );
}

export function PlacesPanel({ places, marker, error, onAdd, onMakeDefault, onStartFrom, onDelete }: PlacesPanelProps) {
  const [name, setName] = useState("");
  const trimmed = name.trim();

  const add = () => {
    if (!marker || !trimmed) return;
    onAdd(trimmed);
    setName("");
  };

  return (
    <div className="places">
      <p className="places-intro">
        Сохранённые места можно называть в запросе: „от дачи до дома“. Место по умолчанию — старт, если он не назван.
      </p>

      {places.length === 0 ? (
        <p className="places-empty">Пока нет сохранённых мест.</p>
      ) : (
        <ul className="places-list">
          {places.map((place) => (
            <li key={place.id} className="places-item">
              <span className="places-name">{place.name}</span>
              <label className="places-default">
                <input
                  type="radio"
                  name="default-place"
                  checked={place.is_default}
                  onChange={() => onMakeDefault(place.id)}
                />
                по умолчанию
              </label>
              <div className="places-actions">
                <button type="button" className="btn btn-small" onClick={() => onStartFrom(place)}>
                  Старт отсюда
                </button>
                <DeleteButton onDelete={() => onDelete(place.id)} />
              </div>
            </li>
          ))}
        </ul>
      )}

      {error && <p className="places-error">{error}</p>}

      <form
        className="places-form"
        onSubmit={(event) => {
          event.preventDefault();
          add();
        }}
      >
        <label className="places-label" htmlFor="place-name">
          Название
        </label>
        <div className="places-row">
          <input id="place-name" className="field" value={name} onChange={(event) => setName(event.target.value)} />
          <button type="submit" className="btn btn-primary" disabled={!marker || !trimmed}>
            Сохранить точку
          </button>
        </div>
        <p className="places-hint">
          {marker ? `${marker.lat.toFixed(4)}, ${marker.lon.toFixed(4)}` : "Кликните по карте, чтобы поставить точку."}
        </p>
      </form>
    </div>
  );
}
