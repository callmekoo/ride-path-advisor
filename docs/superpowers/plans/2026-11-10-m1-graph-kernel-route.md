# M1: граф + ядро + маршрут A→B — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** из OSM-выгрузки региона собрать велопроезжий граф и строить маршрут A→B по весам предпочтений из CLI с HTML-картой.

**Architecture:** Python-пакет `rpa` собирает граф в каталог `.npy`-массивов; Rust-ядро `rpa_kernel` (PyO3) ищет путь по вектору весов дуг, который считает Python. Дуга `a = 2·e + d` (`d=0` — по направлению OSM-пути), `reverse(a) = a ^ 1`, `edge(a) = a >> 1`.

**Tech Stack:** Python 3.14, uv workspace, maturin 1.16, PyO3 0.30 (`abi3-py312`), rust-numpy 0.30, pyosmium 4.3, numpy 2.5, scipy 1.18, typer, pytest, hypothesis.

## Global Constraints

- Спека: `docs/superpowers/specs/2026-11-10-ride-path-advisor-design.md`.
- Код и комментарии — на английском; документация — на русском.
- Контракт ядра: `w[a]` равен `+inf` (запрет) либо `>= edge_len_m[a >> 1]`.
- Ядро ничего не знает о тегах, штрафах и предпочтениях.
- В репозиторий не попадают: `data/`, `out/`, треки, `.env`, личные координаты.
- За HTTPS-прокси cargo нужен `CARGO_HTTP_CAINFO=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem`.
- Ядро тестируется из Python (свойства против scipy); `cargo test` не используется — без `python3-devel` тестовый бинарник не линкуется.
- Каждая задача: сначала падающий тест, потом код, потом коммит.

## Отложено относительно спеки (осознанно)

- **Барьеры на узлах** — в M1 не обрабатываются; вернёмся, если на странице `eval` (M2) маршруты пойдут через закрытые ворота.
- **Скелет Containerfile** — перенесён в M8, чтобы не держать несобираемый артефакт.
- **Закоммиченная реальная вырезка 1–3 МБ** — добавится в M2 вместе с `eval`; в M1 хватает рукописной микрокарты.

---

### Task 1: Каркас workspace + ядро поиска

**Files:**
- Create: `pyproject.toml`, `kernel/pyproject.toml`, `kernel/Cargo.toml`, `kernel/src/lib.rs`, `kernel/src/search.rs`, `src/rpa/__init__.py`, `README.md`
- Test: `tests/test_kernel.py`

**Interfaces — Produces** (модуль `rpa_kernel`):

```python
class Graph:
    def __init__(self, edge_u: u32[E], edge_v: u32[E], node_lonlat: f64[N, 2]): ...
    n_nodes: int
    n_edges: int
    def route(self, src: int, dst: int, w: f32[2E], h_scale: float = 0.999
              ) -> tuple[u32[k], float] | None      # arc ids src→dst, cost; None if unreachable
    def sssp(self, src: int, w: f32[2E], aux: f32[E],
             max_cost: float = inf, reverse: bool = False
             ) -> tuple[f32[N], f32[N], u32[N]]     # cost, summed aux, pred_arc (0xFFFFFFFF = none)
```

`sssp(reverse=True)` даёт стоимость пути **из** каждого узла **в** `src`; `pred_arc[n]` — дуга, выходящая из `n` по этому пути. Недостижимые узлы: `cost = inf`, `aux = inf`. `ValueError` при неверных длинах массивов или номерах узлов.

**Тесты** (`tests/test_kernel.py`, hypothesis: случайные графы 2–40 узлов, веса `len·[1..5]`, часть дуг `inf`):
- `route` на графе-линии возвращает дуги по порядку и сумму весов;
- стоимость `route` == `scipy.sparse.csgraph.dijkstra` (допуск 1e-4 относительный);
- дуги `route` непрерывны: `head(a_i) == tail(a_{i+1})`, начинаются в `src`, кончаются в `dst`, сумма `w` == cost;
- `sssp(...)[0]` == scipy для всех узлов, включая недостижимые; `reverse=True` == scipy на транспонированной матрице;
- `sssp` aux равен сумме `aux` вдоль цепочки `pred_arc`;
- `max_cost` обрезает: узлы дальше порога — `inf`;
- запрещённая (`inf`) дуга не используется; `src == dst` → пустой путь, cost 0;
- неверная длина `w` → `ValueError`.

**Verify:** `uv sync && uv run pytest tests/test_kernel.py -q` → все проходят.

### Task 2: Классификация тегов

**Files:** Create `src/rpa/osm/__init__.py`, `src/rpa/osm/tags.py`; Test `tests/test_tags.py`

**Interfaces — Produces:**

```python
class Highway(IntEnum):   # CYCLEWAY, UNCLASSIFIED, TRACK, RESIDENTIAL, LIVING_STREET, SERVICE,
                          # TERTIARY, PATH, PEDESTRIAN, FOOTWAY, SECONDARY, PRIMARY, TRUNK
class Surface(IntEnum):   # PAVED, LIKELY_PAVED, UNKNOWN, LIKELY_UNPAVED, UNPAVED, BAD
FWD_OK = 1; BACK_OK = 2; SOFT_RESTRICTED = 4; FORD = 8; BRIDGE = 16; TUNNEL = 32; CYCLE_INFRA = 64

@dataclass(frozen=True)
class EdgeAttrs:
    highway: Highway
    surface: Surface
    flags: int

def classify_way(tags: Mapping[str, str]) -> EdgeAttrs | None   # None = not bike-routable
```

**Правила** (параметризованный тест, ~60 случаев):
- `*_link` → класс родителя; `road` → UNCLASSIFIED; `bridleway` → PATH.
- `None`: неизвестный/отсутствующий `highway`, motorway, construction, proposed, steps, `motorroad=yes`, `winter_road=yes`, `ice_road=yes`, `access=no|military` без `bicycle=yes|designated|permissive`.
- `SOFT_RESTRICTED`: `access=private|destination|delivery|customers`, `vehicle=no|private`, `bicycle=no|use_sidepath|dismount`, footway без разрешающего `bicycle`. Разрешающий `bicycle` снимает ограничение.
- Односторонность: `oneway=yes|1|true` → только FWD; `-1|reverse` → только BACK; `junction=roundabout|circular` → FWD, если нет `oneway=no`. Игнорируется при `oneway:bicycle=no`, `cycleway*=opposite*` и для footway/path/pedestrian.
- Покрытие: тег `surface` (таблицы paved / unpaved / bad) → `smoothness=horrible|very_horrible|impassable` → BAD → `tracktype` (grade1 → LIKELY_PAVED, grade2–3 → UNPAVED, grade4–5 → BAD) → класс (trunk/primary → PAVED; secondary/tertiary/cycleway → LIKELY_PAVED; track/path → LIKELY_UNPAVED; остальное → UNKNOWN).
- Флаги: `ford=yes`, `bridge`, `tunnel` (кроме `no`), вело-инфраструктура (`cycleway*=lane|track|…`, `bicycle=designated`).

**Verify:** `uv run pytest tests/test_tags.py -q`.

### Task 3: Хранилище графа

**Files:** Create `src/rpa/graph/__init__.py`, `src/rpa/graph/store.py`; Test `tests/test_store.py`

**Interfaces — Produces:**

```python
FORMAT_VERSION = 1

@dataclass
class GraphData:
    node_lonlat: f64[N, 2]; node_osm_id: i64[N]
    edge_u: u32[E]; edge_v: u32[E]; edge_len_m: f32[E]
    edge_way_id: i64[E]; edge_way_pos: u16[E, 2]
    edge_highway: u8[E]; edge_surface: u8[E]; edge_flags: u16[E]
    geom_off: u32[E + 1]; geom_xy: i32[P, 2]        # lon/lat ×1e7, u→v order, endpoints included
    n_nodes: int; n_edges: int                       # properties
    def edge_geometry(self, e: int) -> f64[k, 2]     # lon/lat degrees
    def validate(self) -> None                       # raises ValueError on shape/dtype/range mismatch

def save_graph(g: GraphData, directory: Path, meta: dict) -> None   # writes *.npy + meta.json atomically
def load_graph(directory: Path, mmap: bool = True) -> tuple[GraphData, dict]
```

**Тесты:** round-trip сохраняет dtype и значения; `load` с неверной `format_version` → `ValueError`; `validate` ловит несогласованные длины и `edge_u >= N`; запись идёт во временный каталог с переименованием (нет полузаписанного графа).

**Verify:** `uv run pytest tests/test_store.py -q`.

### Task 4: Сборка графа из OSM

**Files:** Create `src/rpa/osm/build.py`, `tests/fixtures/micro.opl`; Test `tests/test_build.py`

**Interfaces — Consumes:** `classify_way`, `GraphData`. **Produces:**

```python
@dataclass
class RawWays:                       # flat arrays collected in one pyosmium pass
    refs: i64[R]; xy: i32[R, 2]; way_off: u32[W + 1]
    way_id: i64[W]; way_highway: u8[W]; way_surface: u8[W]; way_flags: u16[W]

def read_ways(path: Path) -> RawWays                 # pyosmium, skips nodes without location
def assemble(raw: RawWays) -> GraphData              # pure numpy: split, geometry, lengths, giant component
def build_graph(path: Path) -> GraphData             # read_ways + assemble
```

**Алгоритм `assemble`:** узел разреза = конец пути или узел, встречающийся ≥2 раз среди узлов оставленных путей; ребро = отрезок пути между соседними узлами разреза; длина — сумма гаверсинусов (f64, R = 6 371 008.8 м, минимум 0.1 м); петли `u == v` выбрасываются; остаётся наибольшая слабосвязная компонента с перенумерацией узлов.

**Микрокарта** `tests/fixtures/micro.opl`: T-перекрёсток, путь с промежуточными узлами, односторонняя улица, зимник (исчезает), здание (игнорируется), изолированный островок (отбрасывается), кольцевой путь.

**Тесты:** ожидаемые N и E; путь режется на перекрёстке на два ребра; геометрия ребра включает промежуточные узлы в порядке u→v; длина ребра 100 м по широте ≈ 100 ± 0.5 м; флаги односторонки доехали; зимника и островка нет; `g.validate()` проходит.

**Verify:** `uv run pytest tests/test_build.py -q`.

### Task 5: Веса, привязка, сборка пути

**Files:** Create `src/rpa/routing/__init__.py`, `src/rpa/routing/cost.py`, `src/rpa/routing/path.py`, `src/rpa/graph/snap.py`; Test `tests/test_cost.py`, `tests/test_path.py`, `tests/test_snap.py`

**Interfaces — Produces:**

```python
@dataclass(frozen=True)
class Prefs:
    avoid_major: float = 0.5                      # 0..1
    surface: Literal["any", "paved", "prefer_unpaved"] = "any"

def arc_weights(g: GraphData, prefs: Prefs) -> f32[2E]
# w = len · (1 + m̂·b_hw) · S_mode[surface] · R,  m̂ = 0.3 + 1.7·avoid_major; таблицы — из спеки.
# Запрещённое направление (нет FWD_OK/BACK_OK) и S = inf → +inf. Итоговый множитель ≤ 50.

class Snapper:
    def __init__(self, g: GraphData): ...
    def nearest_node(self, lon: float, lat: float) -> tuple[int, float]   # node, distance in metres

@dataclass
class RoutePath:
    arcs: u32[k]; coords: f64[P, 2]; length_m: float
    km_by_highway: dict[str, float]; km_by_surface: dict[str, float]

def build_path(g: GraphData, arcs: u32[k]) -> RoutePath   # backward arcs reverse the geometry; joints deduplicated
```

**Тесты:** все конечные веса ≥ длины ребра (инвариант ядра); против односторонки — `inf`; в режиме `paved` грунт дороже асфальта, `bad` — `inf`; рост `avoid_major` увеличивает вес primary и не меняет cycleway; `nearest_node` находит ближайший узел и расстояние ±1 м; `build_path` стыкует геометрию без дублей и с разворотом обратных дуг; сумма `km_by_highway` == длина.

**Verify:** `uv run pytest tests/test_cost.py tests/test_path.py tests/test_snap.py -q`.

### Task 6: CLI и HTML-карта

**Files:** Create `src/rpa/viz/__init__.py`, `src/rpa/viz/map.py`, `src/rpa/cli.py`, `src/rpa/config.py`; Test `tests/test_cli.py`

**Interfaces — Produces:**

```python
def render_map(routes: list[RoutePath], labels: list[str], out: Path) -> None
# self-contained HTML: MapLibre GL JS from CDN, OpenFreeMap style, routes as inline GeoJSON, attribution

# CLI (typer app `rpa`):
#   rpa build-graph PBF [--out data/graph]
#   rpa route --from LAT,LON --to LAT,LON [--avoid-major 0.5] [--surface any] [--graph DIR] [--html FILE]
```

`config.py`: `data_dir()` из переменной `RPA_DATA_DIR`, по умолчанию `./data`; каталог графа — `data/graph/<sha12 файла>-v<FORMAT_VERSION>`, симлинк `data/graph/current`.

**Тесты** (`typer.testing.CliRunner`, микрокарта): `build-graph` создаёт каталог с `meta.json`; `route` печатает длину и раскладку по классам, пишет HTML с GeoJSON; точка дальше 500 м от графа → предупреждение; недостижимая цель → код возврата 1 и понятное сообщение.

**Verify:** `uv run pytest -q` (весь набор).

### Task 7: Реальный регион + бенчмарк (спайк S2)

**Files:** Create `src/rpa/osm/fetch.py`, `spikes/bench_kernel.py`; Modify `src/rpa/cli.py` (команда `fetch-region`), `README.md`

**Interfaces — Produces:**

```python
def fetch_region(urls: list[str], boundary_relation: str, buffer_km: float, out: Path) -> Path
# download (resume, httpx) → osmium merge → buffered boundary polygon (shapely) → osmium extract → region.pbf
```

`osmium` берётся из PATH, иначе — из контейнера podman. Конфиг региона (`config.py`): список пообластных выгрузок download.openstreetmap.fr (Свердловская, Челябинская, Курганская, Тюменская области, Пермский край, Башкортостан), буфер 100 км.

**Verify (на реальных данных, артефакты в `out/`):**
1. `uv run rpa fetch-region && uv run rpa build-graph data/osm/region.pbf` — печатает N, E, распределение классов дорог и покрытия.
2. `uv run python spikes/bench_kernel.py` — критерии S2: полный `sssp` < 400 мс, `route` на ~50 км < 30 мс (медиана по 200 случайным парам).
3. `uv run rpa route --from 56.84,60.60 --to 56.80,59.93 --html out/ab.html` — маршрут на карте выглядит разумно при `--avoid-major 0` и `1`.

Результаты бенчмарка и статистика покрытия записываются в README (раздел «Состояние»).
