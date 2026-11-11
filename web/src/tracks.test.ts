import { describe, expect, it } from "vitest";

import type { TracksStatus } from "./api";
import { formatSize, isTrackFile, stateText } from "./tracks";

const status = (state: TracksStatus["state"], inbox = 0): TracksStatus => ({
  state,
  started: null,
  finished: null,
  log: [],
  inbox: Array.from({ length: inbox }, (_, index) => ({ name: `${index}.gpx`, size: 1 })),
});

describe("isTrackFile", () => {
  it("accepts the known extensions in any case", () => {
    for (const name of ["a.zip", "a.GPX", "a.fit", "a.Tcx", "a.gpx.gz", "a.FIT.GZ", "a.tcx.gz"]) {
      expect(isTrackFile(name), name).toBe(true);
    }
  });

  it("rejects other files and hidden names", () => {
    for (const name of ["a.txt", "a.gz", "a.zip.bak", "gpx", ".gpx", ".hidden.zip", ""]) {
      expect(isTrackFile(name), name).toBe(false);
    }
  });
});

describe("formatSize", () => {
  it("rounds small files up to one kilobyte", () => {
    expect(formatSize(0)).toBe("1 КБ");
    expect(formatSize(300)).toBe("1 КБ");
  });

  it("shows whole kilobytes", () => {
    expect(formatSize(812 * 1024)).toBe("812 КБ");
  });

  it("shows megabytes and gigabytes with a decimal comma", () => {
    expect(formatSize(3.4 * 1024 ** 2)).toBe("3,4 МБ");
    expect(formatSize(1.2 * 1024 ** 3)).toBe("1,2 ГБ");
  });

  it("does not show 1024 of a unit", () => {
    expect(formatSize(1024 * 1024 - 1)).toBe("1,0 МБ");
  });
});

describe("stateText", () => {
  it("asks for files when there is nothing yet", () => {
    expect(stateText(null)).toBe("Загрузите файлы, чтобы начать.");
    expect(stateText(status("idle"))).toBe("Загрузите файлы, чтобы начать.");
  });

  it("counts files waiting in the inbox", () => {
    expect(stateText(status("idle", 3))).toBe("Загрузите новые файлы или обработайте заново те, что уже на сервере.");
  });

  it("describes a run", () => {
    expect(stateText(status("running"))).toBe("Идёт обработка…");
    expect(stateText(status("done"))).toBe("Готово.");
    expect(stateText(status("failed"))).toBe("Обработка не удалась — смотрите журнал ниже.");
  });
});
