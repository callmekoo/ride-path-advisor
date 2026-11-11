"""Command line interface: `rpa fetch-region`, `rpa build-graph`, `rpa route`."""

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

import rpa_kernel
from rpa import config
from rpa.graph.snap import Snapper
from rpa.graph.store import FORMAT_VERSION, GraphData, load_graph, save_graph
from rpa.osm.build import build_graph
from rpa.osm.fetch import fetch_region
from rpa.osm.tags import Highway, Surface
from rpa.routing.cost import Prefs, arc_weights
from rpa.routing.path import build_path
from rpa.viz.map import render_map

app = typer.Typer(no_args_is_help=True, add_completion=False)

# Beyond this distance a snapped point probably is not where the user meant.
FAR_SNAP_M = 500.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _share_by_class(classes: np.ndarray, length_m: np.ndarray, enum: type) -> str:
    km = np.bincount(classes, weights=length_m, minlength=len(enum)) / 1000
    total = km.sum()
    parts = [f"{member.name.lower()} {km[member] / total:.0%}" for member in enum if km[member] > 0]
    return ", ".join(parts)


def _describe_graph(g: GraphData) -> str:
    length = np.asarray(g.edge_len_m, dtype=np.float64)
    return "\n".join(
        [
            f"Узлов: {g.n_nodes}, рёбер: {g.n_edges}, всего {length.sum() / 1000:.1f} км",
            f"Классы дорог: {_share_by_class(np.asarray(g.edge_highway), length, Highway)}",
            f"Покрытие: {_share_by_class(np.asarray(g.edge_surface), length, Surface)}",
        ]
    )


@app.command("fetch-region")
def fetch_region_command(
    buffer_km: Annotated[float, typer.Option(min=0.0, help="Буфер вокруг границы региона, км")] = (
        config.REGION_BUFFER_KM
    ),
) -> None:
    """Скачать выгрузки OSM и вырезать регион с буфером."""
    path = fetch_region(
        config.REGION_EXTRACT_URLS, config.REGION_BOUNDARY_POLY_URL, buffer_km, config.osm_dir(), log=typer.echo
    )
    typer.echo(f"Готово: {path}")


@app.command("build-graph")
def build_graph_command(
    source: Annotated[Path, typer.Argument(help="Файл OSM (.pbf, .osm, .opl)", exists=True, dir_okay=False)],
    out: Annotated[Path | None, typer.Option(help="Каталог графа; по умолчанию data/graph/<sha>-v<N>")] = None,
) -> None:
    """Собрать велопроезжий граф из файла OSM."""
    sha = _sha256(source)
    graph = build_graph(source)
    target = out or config.graph_root() / f"{sha[:12]}-v{FORMAT_VERSION}"
    meta = {
        "source_file": source.name,
        "source_sha256": sha,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    save_graph(graph, target, meta)
    if config.match_file().exists():
        # Rides are matched to OSM ways, not to graph edges, so they carry over to a rebuilt graph.
        _write_rides(target, graph)
    if out is None:
        link = config.current_graph_dir()
        temp_link = link.with_name("current.tmp")
        temp_link.unlink(missing_ok=True)
        temp_link.symlink_to(target.name)
        os.replace(temp_link, link)
    typer.echo(_describe_graph(graph))
    typer.echo(f"Граф записан в {target}")


def _parse_point(value: str) -> tuple[float, float]:
    """Parse "lat,lon" into (lon, lat)."""
    try:
        lat, lon = (float(part) for part in value.split(","))
    except ValueError:
        raise typer.BadParameter(f"ожидается ШИРОТА,ДОЛГОТА, например 56.84,60.60; получено {value!r}") from None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise typer.BadParameter(f"координаты вне допустимого диапазона: {value!r}")
    return lon, lat


def _snap(snapper: Snapper, point: tuple[float, float], name: str) -> int:
    node, distance = snapper.nearest_node(*point)
    if distance > FAR_SNAP_M:
        typer.echo(f"Внимание: точка «{name}» далеко от дорог графа ({distance:.0f} м до ближайшей).", err=True)
    return node


@app.command("route")
def route_command(
    start: Annotated[str, typer.Option("--from", help="Старт: ШИРОТА,ДОЛГОТА")],
    end: Annotated[str, typer.Option("--to", help="Финиш: ШИРОТА,ДОЛГОТА")],
    avoid_major: Annotated[float, typer.Option(min=0.0, max=1.0, help="Избегать крупных дорог, 0..1")] = 0.5,
    surface: Annotated[str, typer.Option(help="any | paved | prefer_unpaved")] = "any",
    graph_dir: Annotated[Path | None, typer.Option("--graph", help="Каталог графа")] = None,
    html: Annotated[Path | None, typer.Option(help="Записать карту маршрута в HTML-файл")] = None,
) -> None:
    """Построить маршрут из точки в точку по предпочтениям."""
    start_point, end_point = _parse_point(start), _parse_point(end)
    try:
        prefs = Prefs(avoid_major=avoid_major, surface=surface)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None

    directory = graph_dir or config.current_graph_dir()
    if not (directory / "meta.json").exists():
        typer.echo(f"Граф не найден в {directory}. Сначала соберите его: rpa build-graph <файл.pbf>", err=True)
        raise typer.Exit(1)
    graph, _ = load_graph(directory)

    snapper = Snapper(graph)
    src = _snap(snapper, start_point, "старт")
    dst = _snap(snapper, end_point, "финиш")

    kernel = rpa_kernel.Graph(graph.edge_u, graph.edge_v, graph.node_lonlat)
    found = kernel.route(src, dst, arc_weights(graph, prefs))
    if found is None:
        typer.echo("Маршрут не найден: между точками нет проезда при заданных ограничениях.", err=True)
        raise typer.Exit(1)
    arcs, cost = found
    path = build_path(graph, arcs)

    typer.echo(f"Длина: {path.length_m / 1000:.1f} км (стоимость ×{cost / max(path.length_m, 1e-9):.2f} от длины)")
    typer.echo("Дороги: " + ", ".join(f"{name} {km:.1f} км" for name, km in path.km_by_highway.items()))
    typer.echo("Покрытие: " + ", ".join(f"{name} {km:.1f} км" for name, km in path.km_by_surface.items()))
    if html is not None:
        label = f"{path.length_m / 1000:.1f} км · avoid_major={avoid_major:g} · {surface}"
        render_map([path], [label], html)
        typer.echo(f"Карта: {html}")


@app.command("eval")
def eval_command(
    out: Annotated[Path, typer.Option(help="Куда записать HTML-страницу")] = Path("out/eval.html"),
    graph_dir: Annotated[Path | None, typer.Option("--graph", help="Каталог графа")] = None,
    only: Annotated[list[str] | None, typer.Option(help="Имена кейсов; по умолчанию все")] = None,
) -> None:
    """Прогнать эталонные запросы и собрать страницу для оценки качества маршрутов."""
    # Imported here: route generation is not needed by the other commands.
    from rpa.eval.cases import CASES
    from rpa.eval.run import run_case
    from rpa.routing.engine import load_engine
    from rpa.viz.eval_page import render_eval

    directory = graph_dir or config.current_graph_dir()
    if not (directory / "meta.json").exists():
        typer.echo(f"Граф не найден в {directory}. Сначала соберите его: rpa build-graph <файл.pbf>", err=True)
        raise typer.Exit(1)
    cases = [case for case in CASES if not only or case.name in only]
    if not cases:
        raise typer.BadParameter(f"нет таких кейсов; доступны: {', '.join(case.name for case in CASES)}")

    engine = load_engine(directory)
    results = []
    for case in cases:
        result = run_case(engine, case)
        results.append(result)
        lengths = ", ".join(f"{c.path.length_m / 1000:.0f}" for c in result.candidates) or "—"
        repeats = max((c.features.repeat_share for c in result.candidates), default=0.0)
        verdict = "ok" if not result.problems else "ПРОБЛЕМА: " + "; ".join(result.problems)
        typer.echo(f"{case.name:20} {result.seconds:5.1f} с  км: {lengths:14} повторы до {repeats:4.0%}  {verdict}")
    render_eval(results, out)
    typer.echo(f"Страница: {out}")
    if any(result.problems for result in results):
        raise typer.Exit(1)


def _load_current_graph(graph_dir: Path | None) -> GraphData:
    directory = graph_dir or config.current_graph_dir()
    if not (directory / "meta.json").exists():
        typer.echo(f"Граф не найден в {directory}. Сначала соберите его: rpa build-graph <файл.pbf>", err=True)
        raise typer.Exit(1)
    return load_graph(directory)[0]


@app.command("import-tracks")
def import_tracks_command(
    graph_dir: Annotated[Path | None, typer.Option("--graph", help="Каталог графа")] = None,
) -> None:
    """Импортировать заезды из data/inbox: архивы Strava и отдельные GPX/FIT/TCX."""
    from rpa.tracks.ingest import import_inbox
    from rpa.tracks.store import save_tracks

    inbox = config.inbox_dir()
    if not inbox.is_dir():
        typer.echo(f"Нет каталога {inbox}: положите туда архив Strava или файлы треков.", err=True)
        raise typer.Exit(1)
    lonlat = np.asarray(_load_current_graph(graph_dir).node_lonlat)
    bbox = (*lonlat.min(axis=0).tolist(), *lonlat.max(axis=0).tolist())
    tracks, report = import_inbox(inbox, bbox, log=typer.echo)
    save_tracks(tracks, config.tracks_dir())

    typer.echo(f"Импортировано заездов: {report.imported}, {report.km:.0f} км, сегментов: {tracks.n_segments}")
    typer.echo(f"Целиком вне региона или слишком короткие: {report.outside_region}")
    if report.skipped_types:
        skipped = ", ".join(f"{name or '(без типа)'} — {count}" for name, count in report.skipped_types.most_common())
        typer.echo(f"Пропущены по типу активности: {skipped}")
    for error in report.unreadable:
        typer.echo(f"Не удалось прочитать: {error}", err=True)
    typer.echo(f"Треки записаны в {config.tracks_dir()}")


def _write_rides(directory: Path, graph: GraphData) -> np.ndarray:
    """Count rides per edge of `graph` from the matched tracks and store them next to the graph."""
    from rpa.match.matcher import load_matched
    from rpa.match.project import project
    from rpa.routing.engine import RIDES_FILE

    rides = project(graph, load_matched(config.match_file()))
    temp = directory / "rides.tmp.npy"
    np.save(temp, rides)
    temp.replace(directory / RIDES_FILE)
    ridden = rides > 0
    km = float(np.asarray(graph.edge_len_m, dtype=np.float64)[ridden].sum()) / 1000
    typer.echo(f"Проеханных рёбер: {int(ridden.sum())} ({km:.0f} км дорог), максимум заездов по ребру: {int(rides.max())}")
    return rides


@app.command("match")
def match_command(
    jobs: Annotated[int, typer.Option(min=1, help="Параллельных процессов")] = min(12, os.cpu_count() or 1),
    graph_dir: Annotated[Path | None, typer.Option("--graph", help="Каталог графа")] = None,
    pbf: Annotated[Path | None, typer.Option(help="Файл OSM, из которого собран граф")] = None,
) -> None:
    """Привязать импортированные треки к дорогам и посчитать заезды по каждому ребру графа."""
    from rpa.match.matcher import build_tiles, match_tracks, save_matched
    from rpa.tracks.store import load_tracks

    if not (config.tracks_dir() / "tracks.json").exists():
        typer.echo("Треки не импортированы. Сначала: rpa import-tracks", err=True)
        raise typer.Exit(1)
    directory = graph_dir or config.current_graph_dir()
    graph = _load_current_graph(graph_dir)
    source = pbf or config.region_pbf()

    # Tiles must come from the same OSM file as the graph, or way ids and geometry may disagree.
    sha = _sha256(source)
    stamp = config.valhalla_dir() / "source_sha256"
    valhalla_config = config.valhalla_dir() / "valhalla.json"
    if not valhalla_config.exists() or not stamp.exists() or stamp.read_text() != sha:
        typer.echo("Собираю тайлы Valhalla (около минуты)")
        valhalla_config = build_tiles(source, config.valhalla_dir())
        stamp.write_text(sha)
    if load_graph(directory)[1].get("source_sha256") != sha:
        typer.echo("Внимание: граф собран из другого файла OSM; пересоберите его: rpa build-graph", err=True)

    tracks = load_tracks(config.tracks_dir())
    typer.echo(f"Привязываю сегментов: {tracks.n_segments}")
    matched, report = match_tracks(tracks, valhalla_config, jobs, log=typer.echo)
    save_matched(matched, config.match_file())
    share = report.matched_km / report.track_km if report.track_km else 0.0
    typer.echo(f"Привязано {report.matched_km:.0f} из {report.track_km:.0f} км ({share:.0%}), не удалось: {report.failed} сегм.")
    _write_rides(directory.resolve(), graph)


@app.command("pop-map")
def pop_map_command(
    out: Annotated[Path, typer.Option(help="Куда записать HTML-карту")] = Path("out/heatmap.html"),
    graph_dir: Annotated[Path | None, typer.Option("--graph", help="Каталог графа")] = None,
) -> None:
    """Нарисовать карту проеханных дорог."""
    from rpa.routing.engine import load_engine
    from rpa.viz.heat_page import render_heatmap

    engine = load_engine(graph_dir or config.current_graph_dir())
    if engine.rides is None:
        typer.echo("Заезды ещё не привязаны. Сначала: rpa match", err=True)
        raise typer.Exit(1)
    render_heatmap(engine.g, engine.rides, out)
    typer.echo(f"Карта: {out}")


@app.command("enrich")
def enrich_command(
    graph_dir: Annotated[Path | None, typer.Option("--graph", help="Каталог графа")] = None,
    pbf: Annotated[Path | None, typer.Option(help="Файл OSM, из которого собран граф")] = None,
    rasters: Annotated[bool, typer.Option(help="Скачивать лес и высоты (нужна сеть)")] = True,
) -> None:
    """Дополнить граф сведениями об окружении дорог: выходы к воде, лес, высоты."""
    from rpa.enrich import rasters as r
    from rpa.enrich.water import read_water, save_water, water_access

    directory = (graph_dir or config.current_graph_dir()).resolve()
    graph = _load_current_graph(graph_dir)
    typer.echo("Читаю водоёмы и реки из OSM")
    nodes, bodies = water_access(graph, read_water(pbf or config.region_pbf()))
    save_water(directory, nodes, bodies)
    rivers = sum(body["kind"] == "river" for body in bodies)
    typer.echo(
        f"Водоёмов с подъездом: {len(bodies) - rivers}, участков рек: {rivers}, "
        f"узлов у воды: {len(np.unique(nodes[:, 0]))}"
    )
    if not rasters:
        return

    lonlat, off = r.edge_samples(graph)
    typer.echo(f"Лес: ESA WorldCover, плиток: {r.tile_count(lonlat, 3)}")
    forest = r.forest_share(lonlat, off, log=typer.echo)
    typer.echo(f"Высоты: Copernicus DEM GLO-30, плиток: {r.tile_count(lonlat, 1)}")
    ele = r.elevation(lonlat, log=typer.echo)
    ascent = np.stack(r.edge_ascent(off, ele), axis=1)
    for name, array in (("forest.npy", forest), ("prof_off.npy", off), ("prof_ele.npy", ele), ("ascent.npy", ascent)):
        np.save(directory / name, array)
    length = np.asarray(graph.edge_len_m, dtype=np.float64)
    typer.echo(
        f"Доля дорог в лесу: {(forest / 255 * length).sum() / length.sum():.0%}; "
        f"высоты от {ele.min() / 10:.0f} до {ele.max() / 10:.0f} м"
    )
