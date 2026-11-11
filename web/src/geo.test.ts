import { describe, expect, it } from "vitest";

import type { LonLat } from "./api";
import { along, bounds, distanceM, pointAt } from "./geo";

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
});
