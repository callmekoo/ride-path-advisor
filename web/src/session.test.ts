import { describe, expect, it } from "vitest";

import type { ParsedRequest, Point } from "./api";
import {
  NOTHING_SAID,
  type Pending,
  answerWithOption,
  answerWithPoint,
  answerWithText,
  edited,
  lengthFromAnswer,
  utterance,
} from "./session";

const HERE: Point = { lon: 60.6, lat: 56.84 };
const loop: ParsedRequest = { ...NOTHING_SAID, shape: "loop", via: "Ключи" };

function pending(about: string, options: Point[] = [], extra: Partial<Pending> = {}): Pending {
  return {
    text: "кольцо через Ключи",
    requestId: 7,
    click: null,
    picks: {},
    understood: loop,
    question: { text: "?", about, options },
    ...extra,
  };
}

describe("utterance", () => {
  it("sends the words, the request being refined and a fresh click", () => {
    expect(utterance(" короче ", 7, HERE)).toEqual({
      text: "короче",
      request_id: 7,
      click: HERE,
      picks: {},
      understood: null,
    });
  });
});

describe("answering a question", () => {
  it("picks an offered place without asking the model again", () => {
    const option = { lon: 60.3, lat: 56.7, label: "Ключи (село), 24 км на юго-запад" };
    const body = answerWithOption(pending("Ключи", [option], { picks: { Ревда: HERE } }), option);

    expect(body.understood).toEqual(loop);
    expect(body.request_id).toBe(7);
    // The way there is a hint for choosing, not part of the place's name.
    expect(body.picks).toEqual({ Ревда: HERE, Ключи: { lon: 60.3, lat: 56.7, label: "Ключи (село)" } });
  });

  it("takes a click on the map as the start when the start was asked for", () => {
    const body = answerWithPoint(pending("start"), HERE);
    expect(body?.click).toEqual(HERE);
    expect(body?.picks).toEqual({});
    expect(body?.understood).toEqual(loop);
  });

  it("takes a click on the map as the place in doubt, under its name", () => {
    expect(answerWithPoint(pending("Ключи"), HERE)?.picks).toEqual({ Ключи: { ...HERE, label: "Ключи" } });
    expect(answerWithPoint(pending("end"), HERE)?.picks).toEqual({ end: { ...HERE, label: null } });
  });

  it("cannot answer a question about length with a click", () => {
    expect(answerWithPoint(pending("length"), HERE)).toBeNull();
  });

  it("puts a number into the request when the length was asked for", () => {
    const body = answerWithText(pending("length", [], { understood: { ...loop, length_max_km: 90 } }), "около 40 км");
    expect(body.understood).toEqual({ ...loop, length_km: 40, length_min_km: null, length_max_km: null });
  });

  it("sends any other answer to the model together with the original words", () => {
    const asked = pending("length", [], { click: HERE, picks: { Ключи: HERE }, text: "кольцо через Ключи." });
    expect(answerWithText(asked, " часа на три ")).toEqual({
      text: "кольцо через Ключи. часа на три",
      request_id: 7,
      click: HERE,
      picks: { Ключи: HERE },
      understood: null,
    });
    expect(answerWithText(pending("Ключи"), "40").understood).toBeNull(); // a number answers only a length question
  });
});

describe("lengthFromAnswer", () => {
  it("reads a bare length", () => {
    expect(lengthFromAnswer("40")).toBe(40);
    expect(lengthFromAnswer("45,5 км")).toBe(45.5);
    expect(lengthFromAnswer("Примерно 60 километров.")).toBe(60);
  });

  it("leaves anything else to the model", () => {
    for (const text of ["100-150", "часа три", "0", "сорок", "40 км по лесу"]) expect(lengthFromAnswer(text)).toBeNull();
  });
});

describe("edited", () => {
  it("resolves the changed request in the setting of the original turn", () => {
    const turn = { text: "кольцо", requestId: 3, click: HERE, picks: { Ключи: HERE }, understood: loop };
    const changed = { ...loop, forest: 1 };
    expect(edited(turn, changed, null)).toEqual({
      text: "",
      request_id: 3,
      click: HERE,
      picks: { Ключи: HERE },
      understood: changed,
    });
  });

  it("starts from the marker if it was moved since", () => {
    const turn = { text: "кольцо", requestId: 3, click: HERE, picks: {}, understood: loop };
    expect(edited(turn, loop, { lon: 60, lat: 57 }).click).toEqual({ lon: 60, lat: 57 });
  });
});
