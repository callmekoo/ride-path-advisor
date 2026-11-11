import type { Chip } from "../chips";

export interface ChipsProps {
  chips: Chip[];
  /** No changes right now (a request is on its way, or a question is waiting). */
  disabled: boolean;
  onRemove(key: string): void;
  /** Open or close the editor of the request. */
  onEdit(): void;
}

export function Chips({ chips, disabled, onRemove, onEdit }: ChipsProps) {
  return (
    <div className="chips">
      {chips.map((chip) => (
        <span key={chip.key} className="chip">
          {chip.text}
          {chip.removable && (
            <button type="button" className="chip-remove" aria-label="Убрать условие" disabled={disabled} onClick={() => onRemove(chip.key)}>
              ×
            </button>
          )}
        </span>
      ))}
      <button type="button" className="chips-edit" disabled={disabled} onClick={onEdit}>
        Изменить
      </button>
    </div>
  );
}
