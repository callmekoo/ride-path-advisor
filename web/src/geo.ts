// Geometry on route lines; coordinates are [lon, lat].

import type { LonLat } from "./api";

const EARTH_M = 6_371_008.8;
const RAD = Math.PI / 180;

export function distanceM(a: LonLat, b: LonLat): number {
  const dLat = (b[1] - a[1]) * RAD;
  const dLon = (b[0] - a[0]) * RAD;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[1] * RAD) * Math.cos(b[1] * RAD) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_M * Math.asin(Math.sqrt(h));
}

/** Metres from the start of the line to each of its points. */
export function along(line: LonLat[]): number[] {
  const metres = [0];
  for (let i = 1; i < line.length; i++) metres.push(metres[i - 1]! + distanceM(line[i - 1]!, line[i]!));
  return metres.slice(0, line.length);
}

/** The point at `share` (0…1) of the line's length; `metres` is `along(line)`. */
export function pointAt(line: LonLat[], metres: number[], share: number): LonLat | null {
  if (line.length === 0) return null;
  const target = Math.min(1, Math.max(0, share)) * metres[metres.length - 1]!;
  let low = 0;
  let high = line.length - 1;
  while (high - low > 1) {
    const middle = (low + high) >> 1;
    if (metres[middle]! <= target) low = middle;
    else high = middle;
  }
  const [a, b] = [line[low]!, line[high]!];
  const span = metres[high]! - metres[low]!;
  const t = span > 0 ? (target - metres[low]!) / span : 0;
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

/** [[west, south], [east, north]] around all the lines, or null if there are no points. */
export function bounds(lines: LonLat[][]): [LonLat, LonLat] | null {
  let [west, south, east, north] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const line of lines) {
    for (const [lon, lat] of line) {
      west = Math.min(west, lon);
      east = Math.max(east, lon);
      south = Math.min(south, lat);
      north = Math.max(north, lat);
    }
  }
  return west <= east ? [[west, south], [east, north]] : null;
}
