"""Render the personal heatmap: every ridden edge, coloured by how often it was ridden."""

import json
from pathlib import Path

import numpy as np

from rpa.graph.store import GraphData

# (fewest rides, colour, legend label), from rarely to often ridden.
_BUCKETS = [(1, "#4575b4", "1"), (2, "#74add1", "2–4"), (5, "#fdae61", "5–14"), (15, "#f46d43", "15–49"), (50, "#a50026", "50+")]

_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rpa: карта заездов</title>
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<style>
  html, body, #map { height: 100%; margin: 0; }
  #legend { position: absolute; top: 10px; left: 10px; z-index: 1; background: #fff; padding: 8px 12px;
            border-radius: 6px; font: 13px/1.6 system-ui, sans-serif; box-shadow: 0 1px 4px rgba(0, 0, 0, .3); }
  #legend i { display: inline-block; width: 18px; height: 4px; margin-right: 8px; vertical-align: middle; }
</style>
</head>
<body>
<div id="map"></div>
<div id="legend"><b>Заездов по дороге</b></div>
<script>
const data = __DATA__;
const map = new maplibregl.Map({
  container: "map", style: "https://tiles.openfreemap.org/styles/positron",
  bounds: data.bounds, fitBoundsOptions: { padding: 40 },
  customAttribution: "© участники OpenStreetMap",
});
map.addControl(new maplibregl.NavigationControl());
const legend = document.getElementById("legend");
for (const feature of data.features) {
  const row = document.createElement("div");
  const swatch = document.createElement("i");
  swatch.style.background = feature.properties.colour;
  row.append(swatch, feature.properties.label);
  legend.append(row);
}
map.on("load", () => {
  map.addSource("rides", { type: "geojson", data: { type: "FeatureCollection", features: data.features } });
  map.addLayer({ id: "rides", type: "line", source: "rides",
                 paint: { "line-color": ["get", "colour"], "line-width": ["interpolate", ["linear"], ["zoom"], 8, 1, 14, 3] } });
});
</script>
</body>
</html>
"""


def render_heatmap(g: GraphData, rides: np.ndarray, out: Path) -> None:
    """Write an HTML map of all edges with at least one ride."""
    features = []
    for i, (low, colour, label) in enumerate(_BUCKETS):
        high = _BUCKETS[i + 1][0] if i + 1 < len(_BUCKETS) else np.inf
        edges = np.flatnonzero((rides >= low) & (rides < high))
        lines = [np.round(g.edge_geometry(int(e)), 5).tolist() for e in edges]
        features.append(
            {
                "type": "Feature",
                "properties": {"colour": colour, "label": label},
                "geometry": {"type": "MultiLineString", "coordinates": lines},
            }
        )
    ridden = np.asarray(g.node_lonlat)[np.unique(np.asarray(g.edge_u)[rides > 0])]
    low, high = (ridden.min(axis=0), ridden.max(axis=0)) if len(ridden) else (np.zeros(2), np.zeros(2))
    data = {"features": features, "bounds": [low.tolist(), high.tolist()]}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_PAGE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
