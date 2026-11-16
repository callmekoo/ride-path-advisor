import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  type CandidateBody,
  type Heatmap,
  type LonLat,
  type Meta,
  type ParseBody,
  type ParsedRequest,
  type Place,
  type Point,
  type RouteResponse,
  type SavedRoute,
  type SavedRouteFull,
  type TracksStatus,
  api,
} from "./api";
import { chips, pointOf, without } from "./chips";
import { CandidateCard } from "./components/CandidateCard";
import { Chat, type Entry } from "./components/Chat";
import { Chips } from "./components/Chips";
import { ElevationProfile } from "./components/ElevationProfile";
import { MapView } from "./components/MapView";
import { PlacesPanel } from "./components/PlacesPanel";
import { RequestEditor } from "./components/RequestEditor";
import { SavedRoutesPanel } from "./components/SavedRoutesPanel";
import { TracksPanel } from "./components/TracksPanel";
import { along, bounds, insertIndex, pointAt, walkLines } from "./geo";
import { routeColour } from "./palette";
import {
  MAX_POINTS,
  type Pending,
  type Role,
  type Turn,
  answerWithOption,
  answerWithPoint,
  answerWithText,
  appended,
  closedLoop,
  edited,
  freshTurn,
  frameAfter,
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

type Tab = "route" | "places" | "saved" | "tracks";
type Frame = [LonLat, LonLat];

const TABS: [Tab, string][] = [
  ["route", "Маршрут"],
  ["places", "Места"],
  ["saved", "Сохранённые"],
  ["tracks", "Треки"],
];
const TRACKS_POLL_MS = 2000;
const NO_POINTS: Point[] = []; // one array for "none", so the map does not redraw markers on every render
const NO_OPTIONS: Point[] = [];
const NO_ROLES: Role[] = [];
const AROUND_HOME_DEG = 0.25; // how much of the map to show around the default place at first

/** A route on the screen: offered just now, or opened from the saved ones. */
interface Shown {
  key: string;
  body: CandidateBody;
  gpxUrl: string;
  candidateId: number | null; // null for a saved route: it cannot be saved again
}

function found(count: number): string {
  const few = count % 10 >= 2 && count % 10 <= 4 && (count % 100 < 12 || count % 100 > 14);
  const word = count % 10 === 1 && count % 100 !== 11 ? "вариант" : few ? "варианта" : "вариантов";
  return `Нашёл ${count} ${word}.`;
}

const explain = (error: unknown) => (error instanceof ApiError ? error.message : "Что-то пошло не так. Попробуйте ещё раз.");

export function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [places, setPlaces] = useState<Place[]>([]);
  const [placeError, setPlaceError] = useState<string | null>(null);
  const [savedRoutes, setSavedRoutes] = useState<SavedRoute[]>([]);
  const [opened, setOpened] = useState<SavedRouteFull | null>(null); // the saved route on the screen
  const [savedAs, setSavedAs] = useState<Record<number, number>>({}); // candidate id → saved route id

  const [tab, setTab] = useState<Tab>("route");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<Pending | null>(null); // a question waiting for the rider
  const [turn, setTurn] = useState<Turn | null>(null); // the turn behind the shown result
  const [result, setResult] = useState<RouteResponse | null>(null);
  const [editing, setEditing] = useState(false);
  const [selected, setSelected] = useState(0);
  const [points, setPoints] = useState<Point[]>([]); // clicked on the map, in order: sent with every request
  const [place, setPlace] = useState<Point | null>(null); // where a new saved place would go
  const [hoverShare, setHoverShare] = useState<number | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [heatmap, setHeatmap] = useState<Heatmap | null>(null);
  const [heatmapOn, setHeatmapOn] = useState(false);
  const [tracksStatus, setTracksStatus] = useState<TracksStatus | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const [tracksError, setTracksError] = useState<string | null>(null);
  const tracksFinished = useRef<string | null>(null); // when the last run known here ended, to notice the next one
  const heatmapOnNow = useRef(heatmapOn);
  heatmapOnNow.current = heatmapOn;
  const entryId = useRef(0);
  const pointsNow = useRef(points);
  pointsNow.current = points;
  const inFlight = useRef(false); // one request at a time: the later answer must not overwrite the earlier

  const say = useCallback((role: Entry["role"], text: string, notes: string[] = [], found?: { fromEdit: boolean }) => {
    const entry: Entry = { id: ++entryId.current, role, text, notes, found: !!found };
    setEntries((log) => appended(log, entry, found?.fromEdit ?? false));
  }, []);

  useEffect(() => {
    // The first view: around the default place if there is one, otherwise the whole region.
    Promise.all([api.meta(), api.places(), api.savedRoutes()])
      .then(([about, saved, routes]) => {
        setMeta(about);
        setPlaces(saved);
        setSavedRoutes(routes);
        const home = saved.find((place) => place.is_default);
        const [west, south, east, north] = about.bbox;
        setFrame(
          home
            ? [
                [home.lon - 2 * AROUND_HOME_DEG, home.lat - AROUND_HOME_DEG],
                [home.lon + 2 * AROUND_HOME_DEG, home.lat + AROUND_HOME_DEG],
              ]
            : [
                [west, south],
                [east, north],
              ],
        );
      })
      .catch((error) => say("error", explain(error)));
  }, [say]);

  const showTracks = useCallback((status: TracksStatus) => {
    // By the time of its end, not by seeing it run: a short run can be over before the first look at it.
    const finished = status.state === "done" && status.finished !== tracksFinished.current;
    setTracksStatus(status);
    if (!finished) return;
    tracksFinished.current = status.finished;
    // New rides change the numbers and the heatmap: take both again.
    api.meta().then(setMeta, () => undefined);
    setHeatmap(null);
    if (heatmapOnNow.current) api.heatmap().then(setHeatmap, () => undefined);
  }, []);

  useEffect(() => {
    if (tab === "tracks") api.tracks().then(showTracks, (error) => setTracksError(explain(error)));
  }, [tab, showTracks]);

  const polling = tracksStatus?.state === "running";
  useEffect(() => {
    if (!polling) return;
    const timer = setInterval(() => void api.tracks().then(showTracks, () => undefined), TRACKS_POLL_MS);
    return () => clearInterval(timer);
  }, [polling, showTracks]);

  const ask = useCallback(
    async (body: ParseBody, fromMap = false) => {
      if (inFlight.current) return;
      inFlight.current = true;
      const before = pointsNow.current;
      setBusy(true);
      setEditing(false);
      setPoints(body.points);
      try {
        const parsed = await api.parse(body);
        const asked: Turn = {
          text: body.text,
          requestId: body.request_id,
          points: body.points,
          picks: body.picks,
          understood: parsed.understood,
        };
        if (parsed.question) {
          setPending({ ...asked, question: parsed.question });
          say("service", parsed.question.text, parsed.warnings);
          const offered = parsed.question.options.map((option): LonLat => [option.lon, option.lat]);
          const view = frameAfter(fromMap, [offered]);
          if (view) setFrame(view);
          return;
        }
        const routed = await api.route(parsed.spec!);
        setPending(null);
        setTurn(asked);
        setResult(routed);
        setSelected(0);
        setHoverShare(null);
        const lines = routed.candidates.map((candidate) => candidate.geometry.coordinates);
        const view = frameAfter(fromMap, lines);
        if (view) setFrame(view);
        const notes = [...parsed.warnings, ...routed.warnings];
        say(
          "service",
          lines.length > 0 ? found(lines.length) : (routed.message ?? "Маршрутов не нашлось."),
          notes,
          // Only a plain count after a point edit (no words) takes the place of the previous one.
          lines.length > 0 && notes.length === 0 ? { fromEdit: body.text === "" } : undefined,
        );
      } catch (error) {
        setPoints(before); // the shown result still belongs to the points it was built from
        say("error", explain(error));
      } finally {
        inFlight.current = false;
        setBusy(false);
      }
    },
    [say],
  );

  const onSay = (text: string) => {
    say("rider", text);
    void ask(pending ? answerWithText(pending, text) : utterance(text, result?.request_id ?? null, points));
  };

  const onOption = (index: number) => {
    const option = pending?.question.options[index];
    if (!pending || !option) return;
    say("rider", option.label ?? "этот вариант");
    void ask(answerWithOption(pending, option));
  };

  /** The setting that point edits are resolved in: the turn behind the result or the question, if any. */
  const base = (): Turn => ({ ...(pending ?? turn ?? freshTurn([])), points });

  /**
   * A change of the points rebuilds the route without the model; with no turn yet, once there are two.
   * `fromMap`: the edit was made on the map itself, which then stays where it is.
   */
  const changePoints = (next: Turn, fromMap: boolean) => {
    if (next.points.length > MAX_POINTS) return say("error", `Точек на карте не больше ${MAX_POINTS}.`);
    if (rebuilds(!!(pending || turn), next.points.length)) void ask(edited(next, next.understood), fromMap);
    else setPoints(next.points);
  };

  const onMapClick = (point: Point) => {
    if (tab === "places") return setPlace(point);
    if (tab !== "route" || busy) return;
    if (pending) {
      say("rider", "точка на карте");
      const answer = answerWithPoint(pending, point);
      if (answer.points.length > MAX_POINTS) return say("error", `Точек на карте не больше ${MAX_POINTS}.`);
      return void ask(answer);
    }
    if (points.length >= MAX_POINTS) return say("error", `Точек на карте не больше ${MAX_POINTS}.`);
    changePoints(withPoint(base(), point), true);
  };

  const onPointClick = (index: number) => {
    if (busy) return;
    const now = base();
    // With other points around, the start marker closes and opens the loop; alone it is removed like the rest.
    if (pointRoles(points.length, now.understood)[index] === "start" && points.length >= 2) {
      changePoints(closedLoop(now, now.understood.shape !== "loop"), true);
    } else {
      changePoints(withoutPoint(now, index), true);
    }
  };

  const onMovePoint = (index: number, point: Point) => {
    if (!busy) changePoints(movedPoint(base(), index, point), true);
  };

  /** The selected route's line was dragged: a new intermediate point where it was dropped, between its neighbours. */
  const onDragLine = (grabbed: LonLat, dropped: LonLat) => {
    const line = chosen?.body.geometry.coordinates;
    if (busy || pending || !turn || !result || showingSaved || !line) return;
    const now = pinnedEnds(base(), result.spec);
    const index = insertIndex(line, now.points.map((point): LonLat => [point.lon, point.lat]), grabbed);
    changePoints(withPointAt(now, index, { lon: dropped[0], lat: dropped[1] }), true);
  };

  const onChange = (request: ParsedRequest) => {
    if (turn) void ask(edited({ ...turn, points }, request));
  };

  const onRemoveChip = (key: string) => {
    const index = pointOf(key);
    if (index !== null) changePoints(withoutPoint(base(), index), false);
    else if (understood) onChange(without(understood, key));
  };

  const onReset = () => {
    // A new route starts from scratch: the next request takes the default place or a fresh click.
    setPoints([]);
    setSelected(0);
    setPending(null);
    setTurn(null);
    setResult(null);
    setEntries([]);
    setEditing(false);
    setHoverShare(null);
  };

  const onSave = async (candidateId: number) => {
    try {
      const route = await api.saveRoute(candidateId);
      setSavedAs((known) => ({ ...known, [candidateId]: route.id }));
      setSavedRoutes(await api.savedRoutes());
    } catch (error) {
      say("error", explain(error));
    }
  };

  const onOpenSaved = async (id: number) => {
    try {
      const route = await api.savedRoute(id);
      setOpened(route);
      setHoverShare(null);
      setFrame(bounds([route.candidate.geometry.coordinates]));
    } catch (error) {
      say("error", explain(error));
    }
  };

  const onDeleteSaved = async (id: number) => {
    await api.deleteSavedRoute(id).catch(() => undefined); // already gone is as good as deleted
    if (opened?.id === id) setOpened(null);
    setSavedAs((known) => Object.fromEntries(Object.entries(known).filter(([, saved]) => saved !== id)));
    setSavedRoutes(await api.savedRoutes().catch(() => []));
  };

  const changePlaces = async (change: () => Promise<unknown>) => {
    try {
      await change();
      setPlaceError(null);
    } catch (error) {
      setPlaceError(explain(error));
    }
    setPlaces(await api.places().catch(() => places));
  };

  const onHeatmap = async (on: boolean) => {
    setHeatmapOn(on);
    if (!on || heatmap) return;
    try {
      setHeatmap(await api.heatmap());
    } catch (error) {
      setHeatmapOn(false);
      say("error", explain(error));
    }
  };

  const onUpload = async (files: File[]) => {
    setTracksError(null);
    try {
      for (const [index, file] of files.entries()) {
        setUploading(`${index + 1} из ${files.length}: ${file.name}`);
        await api.uploadTrack(file);
      }
      showTracks(await api.importTracks());
    } catch (error) {
      setTracksError(explain(error));
      await api.tracks().then(showTracks, () => undefined);
    } finally {
      setUploading(null);
    }
  };

  const onImport = async () => {
    setTracksError(null);
    try {
      showTracks(await api.importTracks());
    } catch (error) {
      setTracksError(explain(error));
    }
  };

  const showingSaved = tab === "saved";
  const shown: Shown[] = useMemo(() => {
    if (showingSaved) {
      return opened
        ? [{ key: `saved-${opened.id}`, body: opened.candidate, gpxUrl: api.savedRouteGpx(opened.id), candidateId: null }]
        : [];
    }
    return (result?.candidates ?? []).map((candidate) => ({
      key: `candidate-${candidate.id}`,
      body: candidate,
      gpxUrl: api.candidateGpx(candidate.id),
      candidateId: candidate.id,
    }));
  }, [showingSaved, opened, result]);
  const chosen = shown[showingSaved ? 0 : selected];
  const chosenIndex = showingSaved ? 0 : selected;
  /** Choose a route and bring the whole of it into view; choosing it again re-centres the map. */
  const select = (index: number) => {
    setSelected(index);
    const line = shown[index]?.body.geometry.coordinates;
    if (line && line.length > 0) setFrame(bounds([line]));
  };

  const lines = useMemo(() => shown.map((route) => route.body.geometry.coordinates), [shown]);
  const walk = useMemo(
    () => (chosen ? walkLines(chosen.body.geometry.coordinates, chosen.body.walk_spans ?? []) : []),
    [chosen],
  );
  const hover = useMemo(() => {
    const line = chosen?.body.geometry.coordinates;
    return line && hoverShare !== null ? pointAt(line, along(line), hoverShare) : null;
  }, [chosen, hoverShare]);

  const options = tab === "route" && pending ? pending.question.options : NO_OPTIONS;
  const understood = pending?.understood ?? turn?.understood ?? null;
  const roles = useMemo(() => pointRoles(points.length, understood), [points, understood]);
  const labels = understood ? chips(understood, pending ? null : (result?.spec ?? null), roles) : [];
  // The route's own start and finish are drawn as plain dots only where no clicked point stands for them.
  const start = showingSaved ? (opened?.spec.start ?? null) : roles.includes("start") ? null : (result?.spec.start ?? null);
  const finish = (showingSaved ? opened?.spec.end : roles.includes("finish") ? null : result?.spec.end) ?? null;

  const cards = shown.map((route, index) => (
    <CandidateCard
      key={route.key}
      heading={showingSaved ? (opened?.name ?? "") : `Вариант ${index + 1}`}
      route={route.body}
      colour={routeColour(showingSaved ? 0 : index)}
      selected={index === chosenIndex}
      gpxUrl={route.gpxUrl}
      canSave={route.candidateId !== null}
      saved={route.candidateId !== null && route.candidateId in savedAs}
      onSelect={() => select(index)}
      onSave={() => route.candidateId !== null && void onSave(route.candidateId)}
    />
  ));
  const profile = chosen?.body.profile && (
    <ElevationProfile key={chosen.key} profile={chosen.body.profile} colour={routeColour(chosenIndex)} onHover={setHoverShare} />
  );

  return (
    <div className="app">
      <aside className="panel">
        <header className="panel-header">
          <nav className="tabs">
            {TABS.filter(([name]) => name !== "tracks" || meta?.can_import_tracks).map(([name, title]) => (
              <button key={name} className={name === tab ? "tab tab-active" : "tab"} onClick={() => setTab(name)}>
                {title}
              </button>
            ))}
          </nav>
        </header>

        {tab === "route" && (
          <Chat
            entries={entries}
            question={pending?.question ?? null}
            busy={busy}
            canParse={meta?.has_llm ?? true}
            hasResult={result !== null || pending !== null}
            onSay={onSay}
            onOption={onOption}
            onReset={onReset}
          >
            {labels.length > 0 && (
              <Chips
                chips={labels}
                disabled={busy || pending !== null}
                onRemove={onRemoveChip}
                onEdit={() => setEditing(!editing)}
              />
            )}
            {editing && turn && <RequestEditor request={turn.understood} onApply={onChange} onCancel={() => setEditing(false)} />}
            {cards}
            {profile}
          </Chat>
        )}

        {tab === "places" && (
          <PlacesPanel
            places={places}
            marker={place}
            error={placeError}
            onAdd={(name) => place && void changePlaces(() => api.addPlace({ name, lon: place.lon, lat: place.lat }))}
            onMakeDefault={(id) => void changePlaces(() => api.changePlace(id, { is_default: true }))}
            onStartFrom={(chosen) => {
              if (busy) return;
              setTab("route");
              changePoints(startingAt(base(), { lon: chosen.lon, lat: chosen.lat, label: chosen.name }), false);
            }}
            onDelete={(id) => void changePlaces(() => api.deletePlace(id))}
          />
        )}

        {tab === "saved" && (
          <SavedRoutesPanel
            routes={savedRoutes}
            openId={opened?.id ?? null}
            onOpen={(id) => void onOpenSaved(id)}
            onDelete={(id) => void onDeleteSaved(id)}
          >
            {cards}
            {profile}
          </SavedRoutesPanel>
        )}

        {tab === "tracks" && (
          <TracksPanel
            status={tracksStatus}
            riddenKm={meta?.has_rides ? meta.ridden_km : null}
            uploading={uploading}
            error={tracksError}
            onUpload={(files) => void onUpload(files)}
            onImport={() => void onImport()}
          />
        )}
      </aside>

      <MapView
        frame={frame}
        start={start}
        finish={finish}
        points={showingSaved ? NO_POINTS : points}
        roles={showingSaved ? NO_ROLES : roles}
        locked={busy}
        onMovePoint={onMovePoint}
        onPointClick={onPointClick}
        onDragLine={onDragLine}
        place={tab === "places" ? place : null}
        routes={lines}
        walk={walk}
        selected={chosenIndex}
        onSelect={select}
        onClick={onMapClick}
        options={options}
        onOption={onOption}
        hover={hover}
        heatmap={heatmapOn ? heatmap : null}
      >
        {meta?.has_rides && (
          <label className="map-toggle" title={`Проехано ${Math.round(meta.ridden_km)} км дорог`}>
            <input type="checkbox" checked={heatmapOn} onChange={(event) => void onHeatmap(event.target.checked)} />
            Мои заезды
          </label>
        )}
      </MapView>
    </div>
  );
}
