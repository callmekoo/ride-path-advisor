import { type PointerEvent, useMemo, useState } from "react";

import type { Profile } from "../api";
import { formatKm } from "./CandidateCard";

export interface ElevationProfileProps {
  profile: Profile;
  colour: string;
  /** The pointer is at this share (0…1) of the route's length; null when it leaves the chart. */
  onHover(share: number | null): void;
}

const WIDTH = 1000;
const HEIGHT = 110;
const MAX_POINTS = 500;
const MIN_RANGE_M = 40; // a flat route must not look like a cliff

/** Elevation at `distance`, interpolated between the two surrounding samples. */
function elevationAt(distance_m: number[], elevation_m: number[], distance: number): number {
  let low = 0;
  let high = distance_m.length - 1;
  while (high - low > 1) {
    const middle = (low + high) >> 1;
    if (distance_m[middle]! <= distance) low = middle;
    else high = middle;
  }
  const from = distance_m[low]!;
  const span = distance_m[high]! - from;
  const t = span > 0 ? Math.min(Math.max((distance - from) / span, 0), 1) : 0;
  return elevation_m[low]! + t * (elevation_m[high]! - elevation_m[low]!);
}

export function ElevationProfile({ profile, colour, onHover }: ElevationProfileProps) {
  const [share, setShare] = useState<number | null>(null);
  const { distance_m, elevation_m } = profile;
  const count = Math.min(distance_m.length, elevation_m.length);

  const shape = useMemo(() => {
    if (count < 2) return null;
    const total = distance_m[count - 1]!;
    let low = Infinity;
    let high = -Infinity;
    for (let i = 0; i < count; i++) {
      low = Math.min(low, elevation_m[i]!);
      high = Math.max(high, elevation_m[i]!);
    }
    const pad = Math.max(0, MIN_RANGE_M - (high - low)) / 2;
    const bottom = low - pad;
    const range = high - low + 2 * pad;
    const step = Math.max(1, Math.ceil(count / MAX_POINTS));
    const points: string[] = [];
    for (let i = 0; i < count; i += step) points.push(point(i));
    if ((count - 1) % step !== 0) points.push(point(count - 1));

    function point(i: number): string {
      const x = total > 0 ? (distance_m[i]! / total) * WIDTH : 0;
      const y = HEIGHT - ((elevation_m[i]! - bottom) / range) * HEIGHT;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }
    const line = points.join(" ");
    return { total, low, high, line, area: `0,${HEIGHT} ${line} ${WIDTH},${HEIGHT}` };
  }, [distance_m, elevation_m, count]);

  if (!shape) return null;

  const move = (event: PointerEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    if (box.width <= 0) return;
    const next = Math.min(Math.max((event.clientX - box.left) / box.width, 0), 1);
    setShare(next);
    onHover(next);
  };
  const leave = () => {
    setShare(null);
    onHover(null);
  };

  const readout =
    share === null
      ? null
      : `${formatKm((share * shape.total) / 1000)} · ${Math.round(elevationAt(distance_m, elevation_m, share * shape.total))} м`;

  return (
    <div className="profile">
      <div className="profile-labels">
        <span>мин {Math.round(shape.low)} м</span>
        <span className="profile-readout">{readout}</span>
        <span>макс {Math.round(shape.high)} м</span>
      </div>
      <div className="profile-plot" onPointerMove={move} onPointerDown={move} onPointerLeave={leave} onPointerCancel={leave}>
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" role="img" aria-label="Профиль высоты">
          <polygon points={shape.area} fill={colour} fillOpacity={0.15} />
          <polyline points={shape.line} fill="none" stroke={colour} strokeWidth={2} vectorEffect="non-scaling-stroke" />
        </svg>
        {share !== null && <div className="profile-cursor" style={{ left: `${share * 100}%` }} />}
      </div>
      <div className="profile-labels">
        <span>0 км</span>
        <span>{formatKm(shape.total / 1000)}</span>
      </div>
    </div>
  );
}
