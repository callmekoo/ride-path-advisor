"""Render routes into a single self-contained HTML page for eyeballing."""

import json
from pathlib import Path

from rpa.routing.path import RoutePath

_COLOURS = ["#d7263d", "#1b6ca8", "#2e933c", "#f49d37", "#7b2d8e"]

_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ride-path-advisor</title>
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<style>
  html, body, #map { height: 100%; margin: 0; }
  #legend { position: absolute; top: 10px; left: 10px; z-index: 1; background: #fff;
            padding: 8px 12px; border-radius: 6px; font: 14px/1.5 system-ui, sans-serif;
            box-shadow: 0 1px 4px rgba(0, 0, 0, .3); }
  #legend i { display: inline-block; width: 18px; height: 4px; margin-right: 8px; vertical-align: middle; }
</style>
</head>
<body>
<div id="map"></div>
<div id="legend"></div>
<script>
const routes = __ROUTES__;
const map = new maplibregl.Map({
  container: "map",
  style: "https://tiles.openfreemap.org/styles/liberty",
  customAttribution: "© участники OpenStreetMap",
});
map.addControl(new maplibregl.NavigationControl());
map.on("load", () => {
  map.addSource("routes", { type: "geojson", data: routes });
  map.addLayer({ id: "routes-casing", type: "line", source: "routes",
                 paint: { "line-color": "#fff", "line-width": 7, "line-opacity": 0.8 } });
  map.addLayer({ id: "routes", type: "line", source: "routes",
                 paint: { "line-color": ["get", "colour"], "line-width": 4 } });
  const bounds = new maplibregl.LngLatBounds();
  const legend = document.getElementById("legend");
  for (const feature of routes.features) {
    feature.geometry.coordinates.forEach((c) => bounds.extend(c));
    const row = document.createElement("div");
    const swatch = document.createElement("i");
    swatch.style.background = feature.properties.colour;
    row.append(swatch, feature.properties.label);
    legend.append(row);
  }
  if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 60, duration: 0 });
});
</script>
</body>
</html>
"""


def render_map(routes: list[RoutePath], labels: list[str], out: Path) -> None:
    """Write an HTML page showing `routes` on an OpenFreeMap base map."""
    features = [
        {
            "type": "Feature",
            "properties": {"label": label, "colour": _COLOURS[i % len(_COLOURS)]},
            "geometry": {"type": "LineString", "coordinates": route.coords.tolist()},
        }
        for i, (route, label) in enumerate(zip(routes, labels, strict=True))
    ]
    collection = json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False)
    # Labels are free text; keep them from closing the <script> element.
    collection = collection.replace("</", "<\\/")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_PAGE.replace("__ROUTES__", collection), encoding="utf-8")
