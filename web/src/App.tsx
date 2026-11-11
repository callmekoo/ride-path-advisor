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
import { chips, without } from "./chips";
import { CandidateCard } from "./components/CandidateCard";
import { Chat, type Entry } from "./components/Chat";
import { Chips } from "./components/Chips";
import { ElevationProfile } from "./components/ElevationProfile";
import { MapView } from "./components/MapView";
import { PlacesPanel } from "./components/PlacesPanel";
import { RequestEditor } from "./components/RequestEditor";
import { SavedRoutesPanel } from "./components/SavedRoutesPanel";
import { TracksPanel } from "./components/TracksPanel";
import { along, bounds, pointAt } from "./geo";
import { routeColour } from "./palette";
import { type Pending, type Turn, answerWithOption, answerWithPoint, answerWithText, edited, utterance } from "./session";

type Tab = "route" | "places" | "saved" | "tracks";
type Frame = [LonLat, LonLat];

const TABS: [Tab, string][] = [
  ["route", "Маршрут"],
  ["places", "Места"],
  ["saved", "Сохранённые"],
  ["tracks", "Треки"],
];
const TRACKS_POLL_MS = 2000;
const NO_OPTIONS: Point[] = []; // one array for "none", so the map does not redraw markers on every render
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
  const [click, setClick] = useState<Point | null>(null); // the start marker, if moved since the last result
  const [hoverShare, setHoverShare] = useState<number | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [heatmap, setHeatmap] = useState<Heatmap | null>(null);
  const [heatmapOn, setHeatmapOn] = useState(false);
  const [tracksStatus, setTracksStatus] = useState<TracksStatus | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const [tracksError, setTracksError] = useState<string | null>(null);
  const tracksState = useRef<TracksStatus["state"] | null>(null); // the state last seen, to notice a run ending
  const heatmapOnNow = useRef(heatmapOn);
  heatmapOnNow.current = heatmapOn;
  const entryId = useRef(0);

  const say = useCallback((role: Entry["role"], text: string, notes: string[] = []) => {
    setEntries((log) => [...log, { id: ++entryId.current, role, text, notes }]);
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
    const finished = tracksState.current === "running" && status.state === "done";
    tracksState.current = status.state;
    setTracksStatus(status);
    if (!finished) return;
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
    async (body: ParseBody) => {
      setBusy(true);
      setEditing(false);
      try {
        const parsed = await api.parse(body);
        const asked: Turn = {
          text: body.text,
          requestId: body.request_id,
          click: body.click,
          picks: body.picks,
          understood: parsed.understood,
        };
        if (parsed.question) {
          setPending({ ...asked, question: parsed.question });
          say("service", parsed.question.text, parsed.warnings);
          const offered = parsed.question.options.map((option): LonLat => [option.lon, option.lat]);
          if (offered.length > 0) setFrame(bounds([offered]));
          return;
        }
        const routed = await api.route(parsed.spec!);
        setPending(null);
        setTurn(asked);
        setResult(routed);
        setSelected(0);
        setHoverShare(null);
        setClick(null);
        const lines = routed.candidates.map((candidate) => candidate.geometry.coordinates);
        if (lines.length > 0) setFrame(bounds(lines));
        say(
          "service",
          lines.length > 0 ? found(lines.length) : (routed.message ?? "Маршрутов не нашлось."),
          [...parsed.warnings, ...routed.warnings],
        );
      } catch (error) {
        say("error", explain(error));
      } finally {
        setBusy(false);
      }
    },
    [say],
  );

  const onSay = (text: string) => {
    say("rider", text);
    void ask(pending ? answerWithText(pending, text) : utterance(text, result?.request_id ?? null, click));
  };

  const onOption = (index: number) => {
    const option = pending?.question.options[index];
    if (!pending || !option) return;
    say("rider", option.label ?? "этот вариант");
    void ask(answerWithOption(pending, option));
  };

  const onMapClick = (point: Point) => {
    const answer = tab === "route" && pending && !busy ? answerWithPoint(pending, point) : null;
    if (answer) {
      say("rider", "точка на карте");
      if (pending?.question.about === "start") setClick(point);
      void ask(answer);
    } else {
      setClick(point);
    }
  };

  const onChange = (request: ParsedRequest) => {
    if (turn) void ask(edited(turn, request, click));
  };

  const onReset = () => {
    // The marker stays where the last route started, so the next request starts there too.
    setClick(click ?? result?.spec.start ?? null);
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

  const lines = useMemo(() => shown.map((route) => route.body.geometry.coordinates), [shown]);
  const hover = useMemo(() => {
    const line = chosen?.body.geometry.coordinates;
    return line && hoverShare !== null ? pointAt(line, along(line), hoverShare) : null;
  }, [chosen, hoverShare]);

  const start = showingSaved ? (opened?.spec.start ?? null) : (click ?? pending?.click ?? result?.spec.start ?? null);
  const finish = (showingSaved ? opened?.spec.end : result?.spec.end) ?? null;
  const options = tab === "route" && pending ? pending.question.options : NO_OPTIONS;
  const understood = pending?.understood ?? turn?.understood ?? null;
  const labels = understood ? chips(understood, pending ? null : (result?.spec ?? null)) : [];

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
      onSelect={() => setSelected(index)}
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
                onRemove={(key) => understood && onChange(without(understood, key))}
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
            marker={click}
            error={placeError}
            onAdd={(name) => click && void changePlaces(() => api.addPlace({ name, lon: click.lon, lat: click.lat }))}
            onMakeDefault={(id) => void changePlaces(() => api.changePlace(id, { is_default: true }))}
            onStartFrom={(place) => setClick({ lon: place.lon, lat: place.lat, label: place.name })}
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
        routes={lines}
        selected={chosenIndex}
        onSelect={setSelected}
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
