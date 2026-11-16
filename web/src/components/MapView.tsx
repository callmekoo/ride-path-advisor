import { Map as MapLibre, Marker, NavigationControl, ScaleControl, setWorkerUrl } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import type { ExpressionSpecification, FilterSpecification, GeoJSONSource, MapMouseEvent, MapTouchEvent } from "maplibre-gl";
import type { FeatureCollection } from "geojson";
import { useEffect, useRef, useState, type ReactNode } from "react";

import type { Heatmap, LonLat, Point } from "../api";
import { HEAT_COLOURS, routeColour } from "../palette";
import type { Role } from "../session";
import "./map.css";

export interface MapViewProps {
  /** Fit the view to this box ([[west, south], [east, north]]) whenever it changes. */
  frame: [LonLat, LonLat] | null;
  /** The route's start when it is not one of the clicked points (the default place, a named one). */
  start: Point | null;
  /** Where a point-to-point route ends, when it is not a clicked point. */
  finish: Point | null;
  /** The clicked points, in order: numbered, draggable markers. */
  points: Point[];
  /** What each clicked point is for (`pointRoles`). */
  roles: Role[];
  /** A request is on its way: the markers cannot be dragged. */
  locked: boolean;
  onMovePoint(index: number, point: Point): void;
  onPointClick(index: number): void;
  /** The selected route's line was grabbed at one place and dropped at another: a new point goes there. */
  onDragLine(grabbed: LonLat, dropped: LonLat): void;
  /** Where a new saved place would go (the «Места» tab): a plain marker, no part of the route. */
  place: Point | null;
  /** Lines of the shown routes; colours come from `routeColour(index)`. */
  routes: LonLat[][];
  /** Index of the chosen route: drawn on top and thicker. */
  selected: number;
  /** Walked stretches of the chosen route: a light dashed line over it. */
  walk: LonLat[][];
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
const DRAG_CLICK_MS = 300;
const GRAB_PX = 5; // a press that moves less than this is a click
const GESTURE_EVENTS = ["mousemove", "mouseup", "touchstart", "touchmove", "touchend", "touchcancel", "blur"];
const LONG_PRESS_MS = 350; // a touch grabs the line only after this long, so that scrolling the map still works
const ROUTE_LAYERS = ["map-routes-other", "map-routes-casing", "map-routes-selected"];
const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };

const filters = (selected: number) => ({
  chosen: ["==", ["get", "index"], selected] as FilterSpecification,
  others: ["!=", ["get", "index"], selected] as FilterSpecification,
});

export function MapView(props: MapViewProps) {
  const { frame, start, finish, points, roles, locked, place, routes, walk, selected, options, hover, heatmap, children } = props;
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibre | null>(null);
  const [loaded, setLoaded] = useState(false);
  const startMarker = useRef<Marker | null>(null);
  const finishMarker = useRef<Marker | null>(null);
  const optionMarkers = useRef<Marker[]>([]);
  const pointMarkers = useRef<Marker[]>([]);
  const placeMarker = useRef<Marker | null>(null);
  const draggedAt = useRef(-Infinity); // a drag ends with a click on the marker, which is not one
  const lineDraggedAt = useRef(-Infinity); // likewise a line drag ends with a click on the map

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
      map.addSource("map-walk", { type: "geojson", data: EMPTY });
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
        id: "map-routes-walk",
        type: "line",
        source: "map-walk",
        layout: { "line-join": "round" },
        paint: { "line-color": "#ffffff", "line-width": 3, "line-dasharray": [1.5, 1.5] },
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
      if (performance.now() - lineDraggedAt.current < DRAG_CLICK_MS) return;
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

    // Grabbing the selected route's line: a mouse press that moves, or a touch held still for a moment. The new
    // point is a temporary marker under the pointer; nothing is rebuilt until the line is let go.
    let release: (() => void) | null = null; // ends a gesture in progress, whatever state it is in
    const onLine = (event: MapMouseEvent | MapTouchEvent) => {
      const { x, y } = event.point;
      return map.queryRenderedFeatures(
        [
          [x - HIT_PX, y - HIT_PX],
          [x + HIT_PX, y + HIT_PX],
        ],
        { layers: map.getLayer("map-routes-selected") ? ["map-routes-selected"] : [] },
      ).length > 0;
    };
    const grab = (event: MapMouseEvent | MapTouchEvent, touch: boolean) => {
      const original = event.originalEvent;
      if (release || latest.current.locked || original.target !== map.getCanvas() || !onLine(event)) return;
      if (touch ? (original as TouchEvent).touches.length !== 1 : (original as MouseEvent).button !== 0) return;
      const grabbed: LonLat = [event.lngLat.lng, event.lngLat.lat];
      const pressed = touch ? (original as TouchEvent).touches[0]! : (original as MouseEvent);
      const from = { x: pressed.clientX, y: pressed.clientY };
      const canvas = map.getCanvas();
      let marker: Marker | null = null;
      let timer: number | undefined;
      let farthest = 0; // how far the pointer has been from where it pressed
      const place = (clientX: number, clientY: number) => {
        const box = canvas.getBoundingClientRect();
        const at = map.unproject([clientX - box.left, clientY - box.top]);
        marker?.setLngLat(at);
        return [at.lng, at.lat] as LonLat;
      };
      const start = () => {
        map.dragPan.disable();
        const element = document.createElement("div");
        element.className = "map-point map-point-via";
        element.textContent = "+";
        marker = new Marker({ element }).setLngLat(grabbed).addTo(map);
      };
      const end = (dropped: LonLat | null) => {
        window.clearTimeout(timer);
        for (const name of GESTURE_EVENTS) {
          window.removeEventListener(name, handle);
        }
        const dragged = marker !== null;
        marker?.remove();
        marker = null;
        if (dragged) {
          map.dragPan.enable();
          lineDraggedAt.current = performance.now();
        }
        release = null;
        if (dragged && dropped) latest.current.onDragLine(grabbed, dropped);
      };
      const handle = (e: Event) => {
        if (e.type === "blur" || e.type === "touchcancel") return end(null);
        if (e.type === "touchstart") return e === original || !touch ? undefined : end(null); // a second finger: a pinch, not a grab
        const point = e.type.startsWith("touch") ? (e as TouchEvent).changedTouches[0] : (e as MouseEvent);
        if (!point) return end(null);
        const moved = Math.hypot(point.clientX - from.x, point.clientY - from.y);
        farthest = Math.max(farthest, moved);
        if (e.type === "mousemove" || e.type === "touchmove") {
          if (e.type === "mousemove" && (e as MouseEvent).buttons === 0) return end(null); // let go outside the window
          if (!marker && moved >= GRAB_PX) {
            if (touch) return end(null); // the finger is scrolling the map
            start();
          }
          if (marker) {
            if (e.cancelable) e.preventDefault();
            place(point.clientX, point.clientY);
          }
        } else {
          // A long press held still and lifted is not a drop; the mouse needs no such rule (it moves to grab).
          end(marker && (!touch || farthest >= GRAB_PX) ? place(point.clientX, point.clientY) : null);
        }
      };
      release = () => end(null);
      for (const name of GESTURE_EVENTS) {
        window.addEventListener(name, handle, { passive: false });
      }
      if (touch) timer = window.setTimeout(start, LONG_PRESS_MS);
    };
    map.on("mousedown", (event) => grab(event, false));
    map.on("touchstart", (event) => grab(event, true));

    mapRef.current = map;
    return () => {
      release?.();
      startMarker.current = null;
      finishMarker.current = null;
      optionMarkers.current = [];
      pointMarkers.current = [];
      placeMarker.current = null;
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
    const data: FeatureCollection = {
      type: "FeatureCollection",
      features: walk.map((coordinates) => ({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates } })),
    };
    (map.getSource("map-walk") as GeoJSONSource).setData(data);
  }, [loaded, walk]);

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
    for (const marker of pointMarkers.current) marker.remove();
    pointMarkers.current = points.map((point, index) => {
      const role = roles[index];
      const element = document.createElement("div");
      element.className = `map-point map-point-${role === "start" ? "start" : role === "finish" ? "finish" : "via"}`;
      element.textContent = role === "start" ? "С" : role === "finish" ? "Ф" : String(role);
      element.title =
        role === "start"
          ? points.length > 1
            ? "Старт: клик замкнёт маршрут в кольцо"
            : "Старт: клик уберёт точку"
          : `${role === "finish" ? "Финиш" : `Точка ${role}`}: клик уберёт точку, можно перетащить`;
      element.addEventListener("click", (event) => {
        event.stopPropagation(); // otherwise the map would also report a click on empty ground
        if (performance.now() - draggedAt.current > DRAG_CLICK_MS) latest.current.onPointClick(index);
      });
      const marker = new Marker({ element, draggable: !locked }).setLngLat([point.lon, point.lat]).addTo(map);
      marker.on("dragend", () => {
        draggedAt.current = performance.now();
        const { lng, lat } = marker.getLngLat();
        latest.current.onMovePoint(index, { lon: lng, lat });
      });
      return marker;
    });
  }, [loaded, points, roles, locked]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loaded) return;
    placeMarker.current?.remove();
    placeMarker.current = null;
    if (!place) return;
    const element = document.createElement("div");
    element.className = "map-place";
    element.title = "Здесь будет сохранённое место";
    placeMarker.current = new Marker({ element }).setLngLat([place.lon, place.lat]).addTo(map);
  }, [loaded, place]);

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
