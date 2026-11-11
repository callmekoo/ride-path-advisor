// The conversation with the server, as pure functions: each returns the body of the next POST /api/parse.

import type { ParseBody, ParsedRequest, Point, Question } from "./api";

export const NOTHING_SAID: ParsedRequest = {
  shape: null,
  start: null,
  end: null,
  via: null,
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
  click: Point | null;
  picks: Record<string, Point>; // answers already given, by Question.about
  understood: ParsedRequest;
}

/** A turn that the server answered with a question. */
export interface Pending extends Turn {
  question: Question;
}

/** New words from the rider; `click` only if the start marker was moved since the last result. */
export function utterance(text: string, requestId: number | null, click: Point | null): ParseBody {
  return { text: text.trim(), request_id: requestId, click, picks: {}, understood: null };
}

function again(turn: Turn, change: Partial<ParseBody>): ParseBody {
  return {
    text: turn.text,
    request_id: turn.requestId,
    click: turn.click,
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

/** The rider clicked on the map instead of answering in words; null if a click cannot answer this question. */
export function answerWithPoint(pending: Pending, point: Point): ParseBody | null {
  const about = pending.question.about;
  if (about === "length") return null;
  const where = { lon: point.lon, lat: point.lat };
  if (about === "start") return again(pending, { click: where });
  const label = about === "end" ? null : about;
  return again(pending, { picks: { ...pending.picks, [about]: { ...where, label } } });
}

/** The rider answered in words. */
export function answerWithText(pending: Pending, text: string): ParseBody {
  const km = pending.question.about === "length" ? lengthFromAnswer(text) : null;
  if (km !== null) {
    return again(pending, { understood: { ...pending.understood, length_km: km, length_min_km: null, length_max_km: null } });
  }
  const asked = pending.text.trim().replace(/[.!?…]+$/, "");
  return again(pending, { text: `${asked}. ${text.trim()}`, understood: null });
}

/** "40", "около 45,5 км": a length that needs no model to understand; null for anything else. */
export function lengthFromAnswer(text: string): number | null {
  const match = /^(?:около|примерно|где-то|на)?\s*(\d+(?:[.,]\d+)?)\s*(?:км|километр[а-я]*)?[.!]?$/i.exec(text.trim());
  if (!match) return null;
  const km = Number(match[1]!.replace(",", "."));
  return km > 0 ? km : null;
}

/**
 * The rider changed how the request was understood. It is resolved again in the setting of the original
 * turn — same earlier result, same answers — so the start and "like option 2" keep their meaning.
 */
export function edited(turn: Turn, understood: ParsedRequest, click: Point | null): ParseBody {
  return { text: "", request_id: turn.requestId, click: click ?? turn.click, picks: turn.picks, understood };
}
