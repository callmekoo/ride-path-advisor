// Shapes of the server's answers (see src/rpa/api/schemas.py) and one function per endpoint.

export type LonLat = [number, number];

export interface Point {
  lon: number;
  lat: number;
  label?: string | null;
}

export type Surface = "any" | "paved" | "prefer_unpaved";
export type Hills = "any" | "avoid" | "seek";
export type Direction = "N" | "NE" | "E" | "SE" | "S" | "SW" | "W" | "NW";

export interface Prefs {
  avoid_major: number;
  surface: Surface;
  familiarity: number;
  forest: number;
  hills: Hills;
}

export interface RouteSpec {
  start: Point;
  end: Point | null;
  shape: "loop" | "p2p";
  length_km: { target: number; min: number; max: number } | null;
  prefs: Prefs;
  via: Point | null;
  via_category: "water" | null;
  avoid: { point: Point; radius_m: number }[];
  bearing_deg: number | null;
  n_candidates: number;
}

/** A request as the model understood it: places are still names, null means "not said". */
export interface ParsedRequest {
  shape: "loop" | "p2p" | null;
  start: string | null;
  end: string | null;
  via: string | null;
  via_water: boolean | null;
  length_km: number | null;
  length_min_km: number | null;
  length_max_km: number | null;
  avoid_major: number | null;
  surface: Surface | null;
  forest: number | null;
  familiarity: number | null;
  hills: Hills | null;
  avoid: string[];
  direction: Direction | null;
  toward: string | null;
  candidate: number | null;
}

export interface Question {
  text: string;
  about: string; // a place name in doubt, or "start", "end", "length"
  options: Point[];
}

export interface Profile {
  distance_m: number[];
  elevation_m: number[];
}

/** A route as shown; `id` is absent on a route opened from the saved ones. */
export interface CandidateBody {
  title: string;
  geometry: { type: "LineString"; coordinates: LonLat[] };
  length_km: number;
  ascent_m: number | null;
  score: number;
  note: string | null;
  via_label: string | null;
  features: Record<string, number>;
  km_by_highway: Record<string, number>;
  km_by_surface: Record<string, number>;
  profile: Profile | null;
}

export interface Candidate extends CandidateBody {
  id: number;
}

export interface RouteResponse {
  request_id: number;
  spec: RouteSpec;
  candidates: Candidate[];
  warnings: string[];
  message: string | null;
}

export interface ParseBody {
  text: string;
  click: Point | null;
  request_id: number | null;
  picks: Record<string, Point>;
  understood: ParsedRequest | null;
}

export interface ParseResponse {
  spec: RouteSpec | null;
  question: Question | null;
  understood: ParsedRequest;
  warnings: string[];
}

export interface Place {
  id: number;
  name: string;
  lon: number;
  lat: number;
  is_default: boolean;
}

export interface SavedRoute {
  id: number;
  name: string;
  created: string;
  length_km: number;
  ascent_m: number | null;
}

export interface SavedRouteFull extends SavedRoute {
  spec: RouteSpec;
  candidate: CandidateBody;
}

export interface Meta {
  bbox: [number, number, number, number]; // west, south, east, north
  has_rides: boolean;
  ridden_km: number;
  has_gazetteer: boolean;
  has_llm: boolean;
  can_import_tracks: boolean;
}

export interface TracksStatus {
  state: "idle" | "running" | "done" | "failed";
  started: string | null; // UTC, "YYYY-MM-DD HH:MM:SS"
  finished: string | null;
  log: string[]; // last lines of the processing log, oldest first
  inbox: { name: string; size: number }[]; // uploaded files waiting for processing
}

/** Ridden roads: one MultiLineString per level, from rarely (0) to often (4) ridden. */
export interface Heatmap {
  type: "FeatureCollection";
  bbox: [number, number, number, number];
  features: {
    type: "Feature";
    properties: { level: number; label: string };
    geometry: { type: "MultiLineString"; coordinates: LonLat[][] };
  }[];
}

/** A refusal from the server, with its explanation in Russian. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  let answer: Response;
  try {
    answer = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "Сервер не отвечает. Проверьте, что он запущен.");
  }
  if (!answer.ok) throw new ApiError(answer.status, await explanation(answer));
  return answer.status === 204 ? (undefined as T) : ((await answer.json()) as T);
}

/** Sends a file as the raw request body; errors are explained like those of `call`. */
async function upload(path: string, file: File): Promise<{ name: string; size: number }> {
  let answer: Response;
  try {
    answer = await fetch(path, { method: "PUT", body: file });
  } catch {
    throw new ApiError(0, "Сервер не отвечает. Проверьте, что он запущен.");
  }
  if (!answer.ok) throw new ApiError(answer.status, await explanation(answer));
  return (await answer.json()) as { name: string; size: number };
}

async function explanation(answer: Response): Promise<string> {
  try {
    const { detail } = (await answer.json()) as { detail?: unknown };
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return "Сервер не принял запрос: " + detail.map((item) => item.msg).join("; ");
  } catch {
    // Not JSON: fall through to the status.
  }
  return `Ошибка сервера (${answer.status}).`;
}

export const api = {
  meta: () => call<Meta>("GET", "/api/meta"),
  parse: (body: ParseBody) => call<ParseResponse>("POST", "/api/parse", body),
  route: (spec: RouteSpec) => call<RouteResponse>("POST", "/api/route", spec),
  heatmap: () => call<Heatmap>("GET", "/api/heatmap"),

  places: () => call<Place[]>("GET", "/api/places"),
  addPlace: (place: { name: string; lon: number; lat: number; is_default?: boolean }) =>
    call<Place>("POST", "/api/places", place),
  changePlace: (id: number, change: Partial<Omit<Place, "id">>) => call<Place>("PUT", `/api/places/${id}`, change),
  deletePlace: (id: number) => call<void>("DELETE", `/api/places/${id}`),

  savedRoutes: () => call<SavedRoute[]>("GET", "/api/routes"),
  savedRoute: (id: number) => call<SavedRouteFull>("GET", `/api/routes/${id}`),
  saveRoute: (candidateId: number, name?: string) =>
    call<SavedRoute>("POST", "/api/routes", { candidate_id: candidateId, name }),
  deleteSavedRoute: (id: number) => call<void>("DELETE", `/api/routes/${id}`),

  tracks: () => call<TracksStatus>("GET", "/api/tracks"),
  uploadTrack: (file: File) => upload(`/api/tracks/files/${encodeURIComponent(file.name)}`, file),
  importTracks: () => call<TracksStatus>("POST", "/api/tracks/import"),

  candidateGpx: (id: number) => `/api/candidates/${id}.gpx`,
  savedRouteGpx: (id: number) => `/api/routes/${id}.gpx`,
};
