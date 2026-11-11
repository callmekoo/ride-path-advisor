// How the request was understood, as short removable labels.

import type { Direction, ParsedRequest, RouteSpec } from "./api";

export interface Chip {
  key: string; // names the condition for `without`
  text: string;
  removable: boolean;
}

// The server's defaults (Prefs in src/rpa/routing/cost.py). After a refinement the model
// spells them out, and a chip saying "nothing special" would only be noise.
const DEFAULTS = { avoid_major: 0.5, familiarity: 0.4, forest: 0.3 };
const SAME = 0.05;

const SIDES: Record<Direction, string> = {
  N: "на север",
  NE: "на северо-восток",
  E: "на восток",
  SE: "на юго-восток",
  S: "на юг",
  SW: "на юго-запад",
  W: "на запад",
  NW: "на северо-запад",
};

export const CLICKED_POINT = "точка на карте";

const km = (value: number) => String(Math.round(value * 10) / 10).replace(".", ",");

function lengthText(request: ParsedRequest): string | null {
  const { length_km: about, length_min_km: low, length_max_km: high } = request;
  if (about !== null) return `около ${km(about)} км`;
  if (low !== null && high !== null) return `${km(low)}–${km(high)} км`;
  if (high !== null) return `до ${km(high)} км`;
  if (low !== null) return `от ${km(low)} км`;
  return null;
}

/** A label for a 0…1 preference, or null when it is at its default. */
function leaning(value: number | null, usual: number, more: string, most: string, less: string): string | null {
  if (value === null || Math.abs(value - usual) < SAME) return null;
  if (value < usual) return less;
  return value >= 0.95 ? most : more;
}

export function chips(request: ParsedRequest, spec: RouteSpec | null): Chip[] {
  const found: Chip[] = [];
  const add = (key: string, text: string | null | false, removable = true) => {
    if (text) found.push({ key, text, removable });
  };

  add("shape", (request.shape ?? spec?.shape) === "loop" && "кольцо", false);
  if (request.start) add("start", `старт: ${request.start}`);
  else if (spec) add("start", `старт: ${spec.start.label ?? CLICKED_POINT}`, false);
  add("end", request.end && `финиш: ${request.end}`, false);
  add("via", request.via && `через: ${request.via}`);
  add("via_water", request.via_water && "к воде");
  add("length", lengthText(request), false);
  add("surface", request.surface === "paved" ? "по асфальту" : request.surface === "prefer_unpaved" && "по грунтам");
  add(
    "avoid_major",
    leaning(request.avoid_major, DEFAULTS.avoid_major, "меньше крупных дорог", "без крупных дорог", "крупные дороги не мешают"),
  );
  add("forest", leaning(request.forest, DEFAULTS.forest, "по лесу", "только лесом", "лес не важен"));
  add(
    "familiarity",
    leaning(request.familiarity, DEFAULTS.familiarity, "по знакомым дорогам", "только знакомые дороги", "по новым дорогам"),
  );
  add("hills", request.hills === "avoid" ? "без горок" : request.hills === "seek" && "с подъёмами");
  for (const name of request.avoid) add(`avoid:${name}`, `не через: ${name}`);
  add("direction", request.direction && SIDES[request.direction]);
  add("toward", request.toward && `в сторону: ${request.toward}`);
  add("candidate", request.candidate !== null && `как вариант ${request.candidate}`);
  return found;
}

/** The request with the condition behind `key` taken back. */
export function without(request: ParsedRequest, key: string): ParsedRequest {
  if (key.startsWith("avoid:")) return { ...request, avoid: request.avoid.filter((name) => `avoid:${name}` !== key) };
  if (key === "length") return { ...request, length_km: null, length_min_km: null, length_max_km: null };
  if (key in request && key !== "avoid") return { ...request, [key]: null };
  return request;
}
