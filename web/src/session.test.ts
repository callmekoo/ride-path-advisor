import { describe, expect, it } from "vitest";

import type { ParsedRequest, Point, RouteSpec } from "./api";
import {
  NOTHING_SAID,
  type Pending,
  answerWithOption,
  answerWithPoint,
  answerWithText,
  appended,
  closedLoop,
  edited,
  frameAfter,
  freshTurn,
  movedPoint,
  pinnedEnds,
  pointRoles,
  rebuilds,
  startingAt,
  utterance,
  withPoint,
  withPointAt,
  withoutPoint,
} from "./session";

const HERE: Point = { lon: 60.6, lat: 56.84 };
const THERE: Point = { lon: 60.9, lat: 56.9 };
const FAR: Point = { lon: 61.2, lat: 57.1 };
const loop: ParsedRequest = { ...NOTHING_SAID, shape: "loop", via: ["Ключи"] };

function pending(about: string, options: Point[] = [], extra: Partial<Pending> = {}): Pending {
  return {
    text: "кольцо через Ключи",
    requestId: 7,
    points: [],
    picks: {},
    understood: loop,
    question: { text: "?", about, options },
    ...extra,
  };
}

describe("utterance", () => {
  it("sends the words, the request being refined and the whole list of clicked points", () => {
    expect(utterance(" короче ", 7, [HERE])).toEqual({
      text: "короче",
      request_id: 7,
      points: [HERE],
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

  it("puts a click that answers the question about the start at the beginning of the points", () => {
    const body = answerWithPoint(pending("start", [], { points: [THERE, FAR] }), HERE);
    expect(body.points).toEqual([HERE, THERE, FAR]);
    expect(body.picks).toEqual({});
    expect(body.understood).toEqual(loop);
  });

  it("takes a click on the map as the place in doubt, under its name", () => {
    expect(answerWithPoint(pending("Ключи"), HERE).picks).toEqual({ Ключи: { ...HERE, label: "Ключи" } });
    expect(answerWithPoint(pending("end"), HERE).picks).toEqual({ end: { ...HERE, label: null } });
  });

  it("sends any other answer to the model together with the original words", () => {
    const asked = pending("start", [], { points: [HERE], picks: { Ключи: HERE }, text: "кольцо через Ключи." });
    expect(answerWithText(asked, " часа на три ")).toEqual({
      text: "кольцо через Ключи. часа на три",
      request_id: 7,
      points: [HERE],
      picks: { Ключи: HERE },
      understood: null,
    });
  });
});

describe("edited", () => {
  it("resolves the changed request in the setting of the original turn, with its points", () => {
    const turn = { text: "кольцо", requestId: 3, points: [HERE, THERE], picks: { Ключи: HERE }, understood: loop };
    const changed = { ...loop, forest: 1 };
    expect(edited(turn, changed)).toEqual({
      text: "",
      request_id: 3,
      points: [HERE, THERE],
      picks: { Ключи: HERE },
      understood: changed,
    });
  });
});

describe("editing the points", () => {
  const turn = { ...freshTurn([HERE, THERE]), requestId: 3 };

  it("starts from clicks alone with no words and nothing understood", () => {
    expect(edited(freshTurn([HERE, THERE]), NOTHING_SAID)).toEqual({
      text: "",
      request_id: null,
      points: [HERE, THERE],
      picks: {},
      understood: NOTHING_SAID,
    });
  });

  it("adds a point at the end, removes one and moves one without touching the rest", () => {
    expect(withPoint(turn, FAR)).toEqual({ ...turn, points: [HERE, THERE, FAR] });
    expect(withoutPoint(withPoint(turn, FAR), 1).points).toEqual([HERE, FAR]);
    expect(movedPoint(turn, 0, FAR)).toEqual({ ...turn, points: [FAR, THERE] });
    expect(turn.points).toEqual([HERE, THERE]);
  });

  it("closes the route into a loop and opens it again", () => {
    const closed = closedLoop(turn, true);
    expect(closed.understood.shape).toBe("loop");
    expect(closed.points).toEqual(turn.points);
    expect(closedLoop(closed, false).understood.shape).toBeNull();
  });

  it("starts from a place: it takes the first point when that is the start, else goes in front", () => {
    expect(startingAt(turn, FAR).points).toEqual([FAR, THERE]);
    const named = { ...turn, understood: { ...NOTHING_SAID, start: "Ревда" } };
    expect(startingAt(named, FAR).points).toEqual([FAR, HERE, THERE]);
    expect(startingAt(freshTurn([]), FAR).points).toEqual([FAR]);
    expect(startingAt(freshTurn([]), FAR).understood).toEqual(NOTHING_SAID);
  });

  it("keeps a lone point as the start when the loop is closed", () => {
    const lone = closedLoop(freshTurn([HERE]), true);
    expect(pointRoles(lone.points.length, lone.understood)).toEqual(["start"]);
  });

  it("builds at once in the setting of a turn, otherwise from two points", () => {
    expect(rebuilds(true, 0)).toBe(true);
    expect(rebuilds(false, 1)).toBe(false);
    expect(rebuilds(false, 2)).toBe(true);
  });
});

describe("pointRoles", () => {
  const said = (change: Partial<ParsedRequest>) => ({ ...NOTHING_SAID, ...change });
  const cases: [string, number, ParsedRequest, (string | number)[]][] = [
    ["nothing named", 4, NOTHING_SAID, ["start", 1, 2, "finish"]],
    ["two points", 2, NOTHING_SAID, ["start", "finish"]],
    ["one point is only the start", 1, NOTHING_SAID, ["start"]],
    ["no points", 0, NOTHING_SAID, []],
    ["start named", 3, said({ start: "Ревда" }), [1, 2, "finish"]],
    ["start named, one point", 1, said({ start: "Ревда" }), ["finish"]],
    ["end named", 3, said({ end: "Ревда" }), ["start", 1, 2]],
    ["loop has no finish", 3, said({ shape: "loop" }), ["start", 1, 2]],
    ["loop with a named start", 3, said({ shape: "loop", start: "Ревда" }), [1, 2, 3]],
    ["the clicked label names nothing", 3, said({ start: "точка на карте", end: " Точка на карте " }), ["start", 1, "finish"]],
  ];
  it.each(cases)("%s", (_, count, understood, roles) => {
    expect(pointRoles(count, understood)).toEqual(roles);
  });

  it("treats a missing turn as nothing named", () => {
    expect(pointRoles(3, null)).toEqual(["start", 1, "finish"]);
  });
});

describe("pinning the ends before inserting a point", () => {
  const spec = (shape: "loop" | "p2p", end: Point | null): RouteSpec =>
    ({ start: { ...HERE, label: "Дом" }, end, shape }) as RouteSpec;
  const FINISH: Point = { ...FAR, label: "Парк" };

  it("puts the route's start first when nothing clicked or said names it", () => {
    const turn = pinnedEnds(freshTurn([]), spec("p2p", FINISH));
    expect(turn.points).toEqual([{ ...HERE, label: "Дом" }, FINISH]);
  });

  it("leaves a start named in words alone, and a clicked list keeps its start", () => {
    const named = { ...freshTurn([]), understood: { ...NOTHING_SAID, start: "Дом", end: "Парк" } };
    expect(pinnedEnds(named, spec("p2p", FINISH))).toEqual(named);
    const clicked = freshTurn([HERE, THERE]);
    expect(pinnedEnds(clicked, spec("p2p", FINISH)).points).toEqual([HERE, THERE]);
  });

  it("pins the loop shape", () => {
    const turn = pinnedEnds(freshTurn([HERE, THERE]), spec("loop", null));
    expect(turn.understood.shape).toBe("loop");
    expect(turn.points).toEqual([HERE, THERE]);
    expect(pinnedEnds(freshTurn([]), spec("loop", null)).points).toEqual([{ ...HERE, label: "Дом" }]);
  });

  it("appends the finish when no clicked point or word is it", () => {
    const named = { ...freshTurn([THERE]), understood: { ...NOTHING_SAID, start: "Дом" } };
    // the only clicked point would be the finish: nothing to add
    expect(pinnedEnds(named, spec("p2p", FINISH)).points).toEqual([THERE]);
    const turn = pinnedEnds(freshTurn([HERE]), spec("p2p", FINISH));
    expect(turn.points).toEqual([HERE, FINISH]);
    const said = { ...freshTurn([HERE]), understood: { ...NOTHING_SAID, end: "Парк" } };
    expect(pinnedEnds(said, spec("p2p", FINISH)).points).toEqual([HERE]);
  });

  it("inserts a point at an index", () => {
    const turn = freshTurn([HERE, FAR]);
    expect(withPointAt(turn, 1, THERE).points).toEqual([HERE, THERE, FAR]);
    expect(withPointAt(turn, 2, THERE).points).toEqual([HERE, FAR, THERE]);
    expect(turn.points).toEqual([HERE, FAR]);
  });
});

describe("frameAfter", () => {
  const line: [number, number][] = [
    [60, 56],
    [61, 57],
  ];

  it("frames the routes found, except after an edit of the points on the map", () => {
    expect(frameAfter(false, [line])).toEqual([
      [60, 56],
      [61, 57],
    ]);
    expect(frameAfter(true, [line])).toBeNull();
  });

  it("has nothing to frame without routes", () => {
    expect(frameAfter(false, [])).toBeNull();
  });
});

describe("appended", () => {
  const line = (id: number, found = false) => ({ id, found });

  it("replaces a found line by the next one from a point edit", () => {
    expect(appended([line(1), line(2, true)], line(3, true), true)).toEqual([line(1), line(3, true)]);
  });

  it("keeps the lines of typed requests, other answers and a different previous line", () => {
    expect(appended([line(1, true)], line(2, true), false)).toEqual([line(1, true), line(2, true)]);
    expect(appended([line(1, true)], line(2), true)).toEqual([line(1, true), line(2)]);
    expect(appended([line(1)], line(2, true), true)).toEqual([line(1), line(2, true)]);
    expect(appended([], line(1, true), true)).toEqual([line(1, true)]);
  });
});
