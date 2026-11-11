import { describe, expect, it } from "vitest";

import type { RouteSpec } from "./api";
import { chips, without } from "./chips";
import { NOTHING_SAID } from "./session";

const texts = (...args: Parameters<typeof chips>) => chips(...args).map((chip) => chip.text);

const spec = {
  start: { lon: 60.6, lat: 56.84, label: "дом" },
  shape: "loop",
} as RouteSpec;

describe("chips", () => {
  it("say what was asked for, in the rider's words", () => {
    const request = {
      ...NOTHING_SAID,
      shape: "loop" as const,
      via_water: true,
      length_km: 45,
      avoid_major: 0.8,
      forest: 0.7,
      avoid: ["Ревда", "Дегтярск"],
      direction: "W" as const,
    };
    expect(texts(request, spec)).toEqual([
      "кольцо",
      "старт: дом",
      "к воде",
      "около 45 км",
      "меньше крупных дорог",
      "по лесу",
      "не через: Ревда",
      "не через: Дегтярск",
      "на запад",
    ]);
  });

  it("keep quiet about defaults that the model spelled out", () => {
    const request = {
      ...NOTHING_SAID,
      avoid_major: 0.5,
      surface: "any" as const,
      forest: 0.3,
      familiarity: 0.4,
      hills: "any" as const,
    };
    expect(texts(request, null)).toEqual([]);
  });

  it("show a range, a ceiling and a floor of length as one chip each", () => {
    expect(texts({ ...NOTHING_SAID, length_min_km: 100, length_max_km: 150 }, null)).toEqual(["100–150 км"]);
    expect(texts({ ...NOTHING_SAID, length_max_km: 60 }, null)).toEqual(["до 60 км"]);
    expect(texts({ ...NOTHING_SAID, length_min_km: 80 }, null)).toEqual(["от 80 км"]);
    expect(texts({ ...NOTHING_SAID, length_km: 33.75 }, null)).toEqual(["около 33,8 км"]);
  });

  it("name the extremes and the opposite leanings", () => {
    const request = {
      ...NOTHING_SAID,
      avoid_major: 1,
      forest: 0,
      familiarity: 0,
      surface: "paved" as const,
      hills: "seek" as const,
    };
    expect(texts(request, null)).toEqual(["по асфальту", "без крупных дорог", "лес не важен", "по новым дорогам", "с подъёмами"]);
  });

  it("let a named start be taken back, but not the one the route simply has", () => {
    const named = chips({ ...NOTHING_SAID, start: "Ревда", end: "Екатеринбург" }, { ...spec, shape: "p2p" });
    expect(named.map((chip) => [chip.text, chip.removable])).toEqual([
      ["старт: Ревда", true],
      ["финиш: Екатеринбург", false],
    ]);
    const clicked = chips(NOTHING_SAID, { ...spec, start: { lon: 60.6, lat: 56.84 } });
    expect(clicked.map((chip) => [chip.text, chip.removable])).toEqual([
      ["кольцо", false],
      ["старт: точка на карте", false],
    ]);
  });
});

describe("without", () => {
  const request = { ...NOTHING_SAID, via: "Ключи", forest: 0.7, length_min_km: 100, length_max_km: 150, avoid: ["Ревда", "Дегтярск"] };

  it("takes back one condition and nothing else", () => {
    expect(without(request, "forest")).toEqual({ ...request, forest: null });
    expect(without(request, "via")).toEqual({ ...request, via: null });
    expect(without(request, "avoid:Ревда")).toEqual({ ...request, avoid: ["Дегтярск"] });
    expect(without(request, "length")).toEqual({ ...request, length_min_km: null, length_max_km: null });
  });

  it("ignores what it does not know", () => {
    expect(without(request, "nonsense")).toEqual(request);
    expect(without(request, "avoid")).toEqual(request);
  });
});
