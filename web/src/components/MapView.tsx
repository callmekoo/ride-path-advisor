import { Map as MapLibre, Marker, NavigationControl, ScaleControl, setWorkerUrl } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import type { ExpressionSpecification, FilterSpecification, GeoJSONSource, MapMouseEvent } from "maplibre-gl";
import type { FeatureCollection } from "geojson";
import { useEffect, useRef, useState, type ReactNode } from "react";

import type { Heatmap, LonLat, Point } from "../api";
import { HEAT_COLOURS, routeColour } from "../palette";
import "./map.css";

export interface MapViewProps {
  /** Fit the view to this box ([[west, south], [east, north]]) whenever it changes. */
  frame: [LonLat, LonLat] | null;
  /** The start marker. */
  start: Point | null;
  /** Where a point-to-point route ends. */
  finish: Point | null;
  /** Lines of the shown routes; colours come from `routeColour(index)`. */
  routes: LonLat[][];
  /** Index of the chosen route: drawn on top and thicker. */
  selected: number;
  onSelect(index: number): void;
  /** A click on the map that hit neither a route nor an option. */
  onClick(point: Point): void;
  /** Places offered as answers to a question: numbered markers. */
  options: Point[];
  onOption(index: number): void;
  /** The point under the cursor of the elevation profile. */
  hover: LonLat | null;
  /** Ridden roads; shown, with a legend, whenever not null. */
  heatmap: Heatmap | null;
  /** Controls laid over the map. */
  children?: ReactNode;
}

// MapLibre looks for its worker next to its own module, where the bundler leaves nothing: point it at the copy
// that the bundler makes for this import.
setWorkerUrl(workerUrl);

const STYLE = "https://tiles.openfreemap.org/styles/liberty";
const ATTRIBUTION = "Высоты: Copernicus DEM GLO-30 · Лес: ESA WorldCover";
const HIT_PX = 6;
const ROUTE_LAYERS = ["map-routes-other", "map-routes-casing", "map-routes-selected"];
const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };

const filters = (selected: number) => ({
  chosen: ["==", ["get", "index"], selected] as FilterSpecification,
  others: ["!=", ["get", "index"], selected] as FilterSpecification,
});

export function MapView(props: MapViewProps) {
  const { frame, start, finish, routes, selected, options, hover, heatmap, children } = props;
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibre | null>(null);
  const [loaded, setLoaded] = useState(false);
  const startMarker = useRef<Marker | null>(null);
  const finishMarker = useRef<Marker | null>(null);
  const optionMarkers = useRef<Marker[]>([]);

  // Handlers are read through this ref so that map listeners never call stale callbacks.
  const latest = useRef(props);
  latest.current = props;

  useEffect(() => {
    if (!container.current) return;
    const map = new MapLibre({
      container: container.current,
      style: STYLE,
      center: [60.6, 56.84],
      zoom: 8,
      attributionControl: { customAttribution: ATTRIBUTION },
    });
    map.addControl(new NavigationControl(), "top-right");
    map.addControl(new ScaleControl({ unit: "metric" }), "bottom-left");

    map.on("load", () => {
      map.addSource("map-routes", { type: "geojson", data: EMPTY });
      map.addSource("map-hover", { type: "geojson", data: EMPTY });
      const line = { "line-join": "round", "line-cap": "round" } as const;
      map.addLayer({
        id: "map-routes-other",
        type: "line",
        source: "map-routes",
        layout: line,
        paint: { "line-color": ["get", "colour"], "line-width": 4, "line-opacity": 0.55 },
      });
      map.addLayer({
        id: "map-routes-casing",
        type: "line",
        source: "map-routes",
        layout: line,
        paint: { "line-color": "#ffffff", "line-width": 9 },
      });
      map.addLayer({
        id: "map-routes-selected",
        type: "line",
        source: "map-routes",
        layout: line,
        paint: { "line-color": ["get", "colour"], "line-width": 6, "line-opacity": 0.95 },
      });
      map.addLayer({
        id: "map-hover",
        type: "circle",
        source: "map-hover",
        paint: { "circle-radius": 6, "circle-color": "#ffffff", "circle-stroke-color": "#1f2937", "circle-stroke-width": 2 },
      });
      setLoaded(true);
    });

    map.on("click", (event: MapMouseEvent) => {
      const { x, y } = event.point;
      const hits = map.queryRenderedFeatures(
        [
          [x - HIT_PX, y - HIT_PX],
          [x + HIT_PX, y + HIT_PX],
        ],
        { layers: ROUTE_LAYERS.filter((id) => map.getLayer(id)) },
      );
      const indexes = hits.map((hit: { properties: Record<string, unknown> }) => Number(hit.properties["index"]));
      if (indexes.length > 0) {
        const chosen = latest.current.selected;
        latest.current.onSelect(indexes.includes(chosen) ? chosen : indexes[0]!);
      } else {
        latest.current.onClick({ lon: event.lngLat.lng, lat: event.lngLat.lat });
      }
    });
    for (const id of ROUTE_LAYERS) {
      map.on("mouseenter", id, () => (map.getCanvas().style.cursor = "pointer"));
      map.on("mouseleave", id, () => (map.getCanvas().style.cursor = ""));
    }

    mapRef.current = map;
    return () => {
      startMarker.current = null;
      finishMarker.current = null;
      optionMarkers.current = [];
      mapRef.current = null;
      setLoaded(false);
      map.remove();
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loaded) return;
    const data: FeatureCollection = {
      type: "FeatureCollection",
      features: routes.map((coordinates, index) => ({
        type: "Feature",
        properties: { index, colour: routeColour(index) },
        geometry: { type: "LineString", coordinates },
      })),
    };
    (map.getSource("map-routes") as GeoJSONSource).setData(data);
    const { chosen, others } = filters(selected);
    map.setFilter("map-routes-other", others);
    map.setFilter("map-routes-casing", chosen);
    map.setFilter("map-routes-selected", chosen);
  }, [loaded, routes, selected]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loaded) return;
    const data: FeatureCollection = hover
      ? { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: hover } }] }
      : EMPTY;
    (map.getSource("map-hover") as GeoJSONSource).setData(data);
  }, [loaded, hover]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loaded) return;
    if (heatmap) {
      const source = map.getSource("map-heat") as GeoJSONSource | undefined;
      if (source) source.setData(heatmap as unknown as FeatureCollection); // Heatmap is GeoJSON by construction
      else {
        map.addSource("map-heat", { type: "geojson", data: heatmap as unknown as FeatureCollection });
        map.addLayer(
          {
            id: "map-heat",
            type: "line",
            source: "map-heat",
            layout: { "line-join": "round", "line-cap": "round" },
            paint: {
              "line-color": ["match", ["get", "level"], ...HEAT_COLOURS.flatMap((colour, level) => [level, colour]), "#888888"] as unknown as ExpressionSpecification, // variadic match is not expressible in the tuple type
              "line-width": ["interpolate", ["linear"], ["zoom"], 8, 1, 14, 3],
              "line-opacity": 0.8,
            },
          },
          "map-routes-other",
        );
      }
      map.setLayoutProperty("map-heat", "visibility", "visible");
    } else if (map.getLayer("map-heat")) {
      map.setLayoutProperty("map-heat", "visibility", "none");
    }
  }, [loaded, heatmap]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loaded || !frame) return;
    map.fitBounds(frame, { padding: 60, maxZoom: 14, duration: 600 });
  }, [loaded, frame]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loaded) return;
    startMarker.current?.remove();
    startMarker.current = null;
    if (!start) return;
    const element = document.createElement("div");
    element.className = "map-start";
    element.title = start.label ?? "Старт";
    startMarker.current = new Marker({ element }).setLngLat([start.lon, start.lat]).addTo(map);
  }, [loaded, start]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loaded) return;
    finishMarker.current?.remove();
    finishMarker.current = null;
    if (!finish) return;
    const element = document.createElement("div");
    element.className = "map-finish";
    element.title = finish.label ?? "Финиш";
    finishMarker.current = new Marker({ element }).setLngLat([finish.lon, finish.lat]).addTo(map);
  }, [loaded, finish]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loaded) return;
    for (const marker of optionMarkers.current) marker.remove();
    optionMarkers.current = options.map((option, index) => {
      const element = document.createElement("button");
      element.type = "button";
      element.className = "map-option";
      element.textContent = String(index + 1);
      element.title = option.label ?? `Вариант ${index + 1}`;
      element.addEventListener("click", (event) => {
        event.stopPropagation(); // otherwise the map would also report a click on empty ground
        latest.current.onOption(index);
      });
      return new Marker({ element }).setLngLat([option.lon, option.lat]).addTo(map);
    });
  }, [loaded, options]);

  return (
    <div className="map">
      <div className="map-canvas" ref={container} />
      {children}
      {heatmap && (
        <div className="map-legend">
          <div className="map-legend-title">Заездов по дороге</div>
          {heatmap.features.map((feature) => (
            <div className="map-legend-row" key={feature.properties.level}>
              <span className="map-legend-swatch" style={{ background: HEAT_COLOURS[feature.properties.level] }} />
              {feature.properties.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
