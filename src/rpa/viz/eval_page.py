"""Render eval results into one HTML page: a list of cases beside a map."""

import json
from pathlib import Path

import numpy as np

from rpa.eval.run import CaseResult

_COLOURS = ["#d7263d", "#1b6ca8", "#2e933c", "#f49d37", "#7b2d8e"]

_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rpa eval</title>
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<style>
  html, body { height: 100%; margin: 0; font: 13px/1.4 system-ui, sans-serif; color: #222; }
  body { display: flex; }
  #cases { width: 640px; overflow-y: auto; border-right: 1px solid #ccc; }
  #map { flex: 1; }
  section { padding: 10px 12px; border-bottom: 1px solid #ddd; cursor: pointer; }
  section.active { background: #eef4fb; }
  h2 { font-size: 14px; margin: 0 0 2px; }
  .text { color: #555; }
  .ok { color: #2e933c; } .fail { color: #d7263d; }
  table { border-collapse: collapse; margin-top: 6px; width: 100%; }
  th, td { text-align: right; padding: 2px 5px; white-space: nowrap; }
  th { font-weight: 600; color: #555; border-bottom: 1px solid #ccc; }
  td i { display: inline-block; width: 14px; height: 4px; vertical-align: middle; }
  .note { color: #8a5a00; margin-top: 4px; }
</style>
</head>
<body>
<div id="cases"></div>
<div id="map"></div>
<script>
const results = __RESULTS__;
const columns = ["", "км", "Δ длины", "повторы", "цена", "поворотов/10 км", "крупные", "грунт", "знакомо", "лес", "набор, м", "оценка"];
const map = new maplibregl.Map({
  container: "map",
  style: "https://tiles.openfreemap.org/styles/liberty",
  center: [60.6, 56.84], zoom: 9,
  customAttribution: "© участники OpenStreetMap",
});
map.addControl(new maplibregl.NavigationControl());

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function show(index) {
  document.querySelectorAll("section").forEach((s, i) => s.classList.toggle("active", i === index));
  const features = results[index].candidates.map((c) => ({
    type: "Feature", properties: { colour: c.colour },
    geometry: { type: "LineString", coordinates: c.coords },
  }));
  map.getSource("routes").setData({ type: "FeatureCollection", features });
  const bounds = new maplibregl.LngLatBounds();
  features.forEach((f) => f.geometry.coordinates.forEach((c) => bounds.extend(c)));
  if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 50, duration: 0 });
}

const list = document.getElementById("cases");
results.forEach((result, index) => {
  const section = el("section");
  const title = el("h2", result.name + " ");
  title.append(el("span", result.problems.length ? "✗" : "✓", result.problems.length ? "fail" : "ok"));
  section.append(title, el("div", `${result.text} · ${result.seconds.toFixed(1)} с`, "text"));
  result.problems.forEach((p) => section.append(el("div", p, "fail")));
  const table = el("table");
  const head = el("tr");
  columns.forEach((c) => head.append(el("th", c)));
  table.append(head);
  result.candidates.forEach((c) => {
    const row = el("tr");
    const swatch = el("td");
    const line = el("i");
    line.style.background = c.colour;
    swatch.append(line);
    row.append(swatch);
    c.cells.forEach((cell) => row.append(el("td", cell)));
    table.append(row);
  });
  section.append(table);
  result.candidates.filter((c) => c.note).forEach((c) => section.append(el("div", c.note, "note")));
  section.addEventListener("click", () => show(index));
  list.append(section);
});

map.on("load", () => {
  map.addSource("routes", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addLayer({ id: "casing", type: "line", source: "routes",
                 paint: { "line-color": "#fff", "line-width": 7, "line-opacity": 0.8 } });
  map.addLayer({ id: "routes", type: "line", source: "routes",
                 paint: { "line-color": ["get", "colour"], "line-width": 4, "line-opacity": 0.85 } });
  if (results.length) show(0);
});
</script>
</body>
</html>
"""


def _result_json(result: CaseResult) -> dict:
    candidates = []
    for i, candidate in enumerate(result.candidates):
        f = candidate.features
        candidates.append(
            {
                "colour": _COLOURS[i % len(_COLOURS)],
                "coords": np.round(candidate.path.coords, 6).tolist(),
                "note": " · ".join(filter(None, [candidate.via_label, candidate.note])) or None,
                "cells": [
                    f"{f.length_m / 1000:.1f}",
                    f"{f.length_error:.0%}",
                    f"{f.repeat_share:.1%}",
                    f"×{f.cost_ratio:.2f}",
                    f"{f.turns_per_10km:.0f}",
                    f"{f.major_share:.0%}",
                    f"{f.unpaved_share:.0%}",
                    f"{f.ridden_share:.0%}",
                    f"{f.forest_share:.0%}",
                    f"{f.ascent_m:.0f}",
                    f"{candidate.score:.0f}",
                ],
            }
        )
    return {
        "name": result.case.name,
        "text": result.case.text,
        "seconds": result.seconds,
        "problems": result.problems,
        "candidates": candidates,
    }


def render_eval(results: list[CaseResult], out: Path) -> None:
    """Write the eval page: every case with its candidates' numbers, and a map for the selected one."""
    data = json.dumps([_result_json(result) for result in results], ensure_ascii=False)
    # Case texts are free text; keep them from closing the <script> element.
    data = data.replace("</", "<\\/")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_PAGE.replace("__RESULTS__", data), encoding="utf-8")
