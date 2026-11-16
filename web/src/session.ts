// The conversation with the server, as pure functions: each returns the body of the next POST /api/parse.

import type { LonLat, ParseBody, ParsedRequest, Point, Question, RouteSpec } from "./api";
import { CLICKED_POINT } from "./chips";
import { bounds } from "./geo";

export const NOTHING_SAID: ParsedRequest = {
  shape: null,
  start: null,
  end: null,
  via: [],
  via_water: null,
  length_km: null,
  length_min_km: null,
  length_max_km: null,
  avoid_major: null,
  surface: null,
  forest: null,
  familiarity: null,
  hills: null,
  avoid: [],
  direction: null,
  toward: null,
  candidate: null,
};

/** What was sent to the server and how it was understood. */
export interface Turn {
  text: string;
  requestId: number | null; // the shown result that this turn refines
  points: Point[]; // clicked on the map, in order; the server assigns their roles
  picks: Record<string, Point>; // answers already given, by Question.about
  understood: ParsedRequest;
}

/** A turn that the server answered with a question. */
export interface Pending extends Turn {
  question: Question;
}

/** The setting of a route built from clicks alone: no words, no earlier result. */
export function freshTurn(points: Point[]): Turn {
  return { text: "", requestId: null, points, picks: {}, understood: NOTHING_SAID };
}

/** New words from the rider; `points` is the whole list of clicked points. */
export function utterance(text: string, requestId: number | null, points: Point[]): ParseBody {
  return { text: text.trim(), request_id: requestId, points, picks: {}, understood: null };
}

function again(turn: Turn, change: Partial<ParseBody>): ParseBody {
  return {
    text: turn.text,
    request_id: turn.requestId,
    points: turn.points,
    picks: turn.picks,
    understood: turn.understood,
    ...change,
  };
}

/** The rider chose one of the offered places. */
export function answerWithOption(pending: Pending, option: Point): ParseBody {
  // The server adds ", 24 км на юго-запад" to tell namesakes apart (resolve.py, _option); the name ends before it.
  const label = option.label?.replace(/, \d+ км на [а-яё-]+$/, "") ?? null;
  return again(pending, { picks: { ...pending.picks, [pending.question.about]: { ...option, label } } });
}

/** The rider clicked on the map instead of answering in words; a start goes to the front of the list. */
export function answerWithPoint(pending: Pending, point: Point): ParseBody {
  const about = pending.question.about;
  const where = { lon: point.lon, lat: point.lat };
  if (about === "start") return again(pending, { points: [where, ...pending.points] });
  const label = about === "end" ? null : about;
  return again(pending, { picks: { ...pending.picks, [about]: { ...where, label } } });
}

/** The rider answered in words. */
export function answerWithText(pending: Pending, text: string): ParseBody {
  const asked = pending.text.trim().replace(/[.!?…]+$/, "");
  return again(pending, { text: `${asked}. ${text.trim()}`, understood: null });
}

/**
 * The rider changed how the request was understood or which points were clicked. It is resolved again in
 * the setting of the original turn — same earlier result, same answers — so the start and "like option 2"
 * keep their meaning, and the model is not asked.
 */
export function edited(turn: Turn, understood: ParsedRequest): ParseBody {
  return { text: "", request_id: turn.requestId, points: turn.points, picks: turn.picks, understood };
}

export const MAX_POINTS = 9;

/** The turn with a point added at the end of the list. */
export function withPoint(turn: Turn, point: Point): Turn {
  return { ...turn, points: [...turn.points, point] };
}

/** The turn without the point at `index`. */
export function withoutPoint(turn: Turn, index: number): Turn {
  return { ...turn, points: turn.points.filter((_, at) => at !== index) };
}

/** The turn with the point at `index` moved. */
export function movedPoint(turn: Turn, index: number, point: Point): Turn {
  return { ...turn, points: turn.points.map((old, at) => (at === index ? point : old)) };
}

/** The turn with a point inserted so that it ends up at `index` in the list. */
export function withPointAt(turn: Turn, index: number, point: Point): Turn {
  return { ...turn, points: [...turn.points.slice(0, index), point, ...turn.points.slice(index)] };
}

/**
 * The turn with the ends of the shown route made explicit, so that adding a point cannot change who is the start
 * or the finish: the server gives those roles to the first and last clicked points unless words name them.
 */
export function pinnedEnds(turn: Turn, spec: RouteSpec): Turn {
  let { points, understood } = turn;
  if (points.length === 0 && !names(understood.start)) {
    points = [{ lon: spec.start.lon, lat: spec.start.lat, label: spec.start.label }];
  }
  if (spec.shape === "loop") {
    understood = { ...understood, shape: "loop" };
  } else if (spec.end && !names(understood.end) && !pointRoles(points.length, understood).includes("finish")) {
    points = [...points, { lon: spec.end.lon, lat: spec.end.lat, label: spec.end.label }];
  }
  return { ...turn, points, understood };
}

/** The route closed into a loop, or opened again (the server then infers start → finish). */
export function closedLoop(turn: Turn, closed: boolean): Turn {
  return { ...turn, understood: { ...turn.understood, shape: closed ? "loop" : null } };
}

/** The place becomes the start: it takes the first point if that is the start, otherwise goes in front. */
export function startingAt(turn: Turn, place: Point): Turn {
  const [first] = pointRoles(turn.points.length, turn.understood);
  const rest = first === "start" ? turn.points.slice(1) : turn.points;
  return { ...turn, points: [place, ...rest] };
}

export type Role = "start" | "finish" | number; // a number is the order among the intermediate points

const names = (place: string | null) => !!place && place.trim().toLowerCase() !== CLICKED_POINT;

/** Roles of the clicked points; mirrors `assign_points` on the server (src/rpa/nlu/resolve.py). */
export function pointRoles(count: number, understood: ParsedRequest | null): Role[] {
  const startNamed = names(understood?.start ?? null);
  const finishNamed = names(understood?.end ?? null) || understood?.shape === "loop";
  let via = 0;
  return Array.from({ length: count }, (_, index): Role => {
    if (index === 0 && !startNamed) return "start";
    if (index === count - 1 && !finishNamed) return "finish";
    return ++via;
  });
}

/** Whether a change of the points builds a route right away: always in the setting of a turn, else from two points. */
export function rebuilds(hasTurn: boolean, pointCount: number): boolean {
  return hasTurn || pointCount >= 2;
}

/**
 * Where to bring the map after routes were found, or null to leave it alone: a result of editing the points
 * on the map must not move the view under the rider's hand while they place the rest of the points.
 */
export function frameAfter(fromMap: boolean, lines: LonLat[][]): [LonLat, LonLat] | null {
  return fromMap ? null : bounds(lines);
}

/**
 * The log with `entry` added. A "found N" line from a point edit replaces the one before it,
 * so that dragging a marker around does not fill the dialogue with the same sentence.
 */
export function appended<T extends { found?: boolean }>(log: T[], entry: T, fromEdit: boolean): T[] {
  const last = log[log.length - 1];
  return fromEdit && entry.found && last?.found ? [...log.slice(0, -1), entry] : [...log, entry];
}
