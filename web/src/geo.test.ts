import { describe, expect, it } from "vitest";

import type { LonLat } from "./api";
import { along, bounds, distanceM, insertIndex, pointAt, walkLines } from "./geo";

// One degree of latitude, then about as far east along the 57th parallel.
const line: LonLat[] = [
  [60, 56],
  [60, 57],
  [62, 57],
];

describe("geo", () => {
  it("measures along a line", () => {
    expect(distanceM([60, 56], [60, 57])).toBeCloseTo(111_195, -2);
    const metres = along(line);
    expect(metres[0]).toBe(0);
    expect(metres[1]).toBeCloseTo(111_195, -2);
    expect(metres[2]! - metres[1]!).toBeCloseTo(121_100, -3);
    expect(along([])).toEqual([]);
  });

  it("finds the point at a share of the length", () => {
    const metres = along(line);
    expect(pointAt(line, metres, 0)).toEqual([60, 56]);
    expect(pointAt(line, metres, 1)).toEqual([62, 57]);
    expect(pointAt(line, metres, 2)).toEqual([62, 57]);
    const quarter = pointAt(line, metres, (0.5 * metres[1]!) / metres[2]!)!;
    expect(quarter[0]).toBe(60);
    expect(quarter[1]).toBeCloseTo(56.5, 6);
    expect(pointAt([], [], 0.5)).toBeNull();
    expect(pointAt([[60, 56]], [0], 0.5)).toEqual([60, 56]);
  });

  it("frames lines", () => {
    expect(bounds([line, [[59, 58]]])).toEqual([[59, 56], [62, 58]]);
    expect(bounds([[]])).toBeNull();
  });

  it("cuts walked stretches out of a route", () => {
    const route: LonLat[] = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]];
    expect(walkLines(route, [[1, 2]])).toEqual([[[1, 0], [2, 0]]]);
    expect(walkLines(route, [[0, 1], [3, 4]])).toEqual([[[0, 0], [1, 0]], [[3, 0], [4, 0]]]);
    expect(walkLines(route, [])).toEqual([]);
    expect(walkLines(route, [[3, 7], [-1, 2], [1, 2]])).toEqual([[[1, 0], [2, 0]]]);
  });

  describe("insertIndex", () => {
    // A straight line east, one coordinate per 0.1 degree.
    const way: LonLat[] = Array.from({ length: 11 }, (_, i) => [60 + i / 10, 56]);

    it("goes between the start and the first intermediate point", () => {
      expect(insertIndex(way, [[60, 56], [60.5, 56], [61, 56]], [60.2, 56.001])).toBe(1);
    });

    it("goes between two intermediate points", () => {
      expect(insertIndex(way, [[60, 56], [60.3, 56], [60.7, 56], [61, 56]], [60.5, 56])).toBe(2);
    });

    it("goes before the finish", () => {
      expect(insertIndex(way, [[60, 56], [60.3, 56], [61, 56]], [60.8, 56])).toBe(2);
    });

    it("on a loop, a grab on the way home lands after the last intermediate point", () => {
      const there = way.slice(0, 6);
      const loop: LonLat[] = [...there, ...there.slice(0, -1).reverse()];
      const points: LonLat[] = [[60, 56], [60.5, 56]]; // the start sits at both ends of the line
      expect(insertIndex(loop, points, [60.2, 56])).toBe(1);
      expect(insertIndex(loop, points, [60.1, 56.0002])).toBe(1);
      expect(insertIndex(loop, [[60, 56], [60.3, 56]], [60.4, 56])).toBe(2);
    });

    it("with no intermediate points, goes between the ends", () => {
      expect(insertIndex(way, [[60, 56], [61, 56]], [60.5, 56])).toBe(1);
      expect(insertIndex(way, [], [60.5, 56])).toBe(0);
    });
  });
});
