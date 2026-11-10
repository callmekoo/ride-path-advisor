# M2: генератор маршрутов + оценка + eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** по структурированному запросу (`RouteSpec`) выдавать 2–3 заметно разных маршрута — кольцо заданной длины или A→B с целевой длиной — и страницу `eval` для просмотра качества на реальном графе.

**Architecture:** `Engine` держит граф, ядро и индексы. Генератор перебирает ~36 попыток (число опорных точек × азимут), каждая — обход через опорные узлы со штрафом за повтор рёбер и подгонкой длины; затем обрезка хвостов, оценка и отбор разнообразных кандидатов. Попытки идут в пуле потоков (ядро отпускает GIL).

**Tech Stack:** как в M1 + pydantic 2 для `RouteSpec`.

## Global Constraints

- Спека: `docs/superpowers/specs/2026-11-10-ride-path-advisor-design.md`, разделы «Алгоритмы» и S3.
- Критерии S3 на реальном графе: ошибка длины ≤10%, доля повторов ≤5%, не меньше 2 кандидатов на запрос.
- Контракт ядра сохраняется: любой вес, включая штрафованный, ≥ длины ребра либо `+inf`.
- В eval-кейсах только публичные точки (центр города, станции); личные координаты не хардкодятся.
- Числа в оценке и штрафах — стартовые, подбираются по странице `eval`.

## Отложено (появится вместе с данными)

- Опорные точки по категории («к воде»), лес, знакомость, набор высоты — M3/M4.
- `avoid`-зоны и `anchor_candidate` — M6, вместе с разбором текста.

---

### Task 1: RouteSpec и Engine

**Files:** Create `src/rpa/routing/spec.py`, `src/rpa/routing/engine.py`, `src/rpa/routing/geo.py`; Modify `src/rpa/routing/cost.py` (Prefs → pydantic), `src/rpa/graph/snap.py` (`nodes_within`); Test `tests/test_spec.py`, `tests/test_engine.py`

```python
class Point(BaseModel): lon: float; lat: float; label: str | None = None
class LengthKm(BaseModel):
    target: float; min: float; max: float            # min <= target <= max, all > 0
    @classmethod
    def around(cls, km: float, tolerance: float = 0.1) -> "LengthKm"
    @classmethod
    def between(cls, low: float, high: float) -> "LengthKm"   # target = midpoint
class RouteSpec(BaseModel):
    start: Point; end: Point | None = None
    shape: Literal["loop", "p2p"]                    # p2p requires end; loop requires length_km
    length_km: LengthKm | None = None
    prefs: Prefs = Prefs(); n_candidates: int = 3

class Engine:
    def __init__(self, g: GraphData): ...
    g; kernel; snapper; length: f32[E]; degree: i32[N]; arc_tail: u32[2E]; arc_head: u32[2E]
    def weights(self, prefs: Prefs) -> f32[2E]
    def node_niceness(self, w: f32[2E]) -> f32[N]    # min multiplier over arcs leaving the node; inf if none

def destination(lon, lat, bearing_deg, distance_m) -> tuple[float, float]
def bearing_deg(lon1, lat1, lon2, lat2) -> float
Snapper.nodes_within(lon, lat, radius_m) -> i64[k]
```

### Task 2: Обход через опорные узлы и обрезка хвостов

**Files:** Create `src/rpa/routing/walk.py`, `src/rpa/routing/trim.py`; Test `tests/test_walk.py`, `tests/test_trim.py`

```python
USED_EDGE_FACTOR = 20.0
def route_through(engine: Engine, w: f32[2E], nodes: list[int]) -> u32[k] | None
# Legs nodes[i] -> nodes[i+1] on a private copy of w: after each leg its edges cost x20 in both
# directions and the reverse of its last arc is forbidden (no U-turn at a via node).
def trim_spurs(arcs: u32[k]) -> u32[m]     # removes every "arc, then its reverse" pair, repeatedly
```

### Task 3: Оценка и разнообразие

**Files:** Create `src/rpa/routing/score.py`, `src/rpa/routing/diversity.py`; Test `tests/test_score.py`, `tests/test_diversity.py`

```python
@dataclass
class Features:
    length_m: float; length_error: float      # relative distance to target ("около") or to the range
    repeat_share: float                        # metres ridden on an edge beyond its first traversal / total
    cost_ratio: float                          # preference cost / length; 1.0 = ideal roads
    turns_per_10km: float; major_share: float; unpaved_share: float
def features(engine, w, arcs, length: LengthKm | None) -> Features
def score(f: Features) -> float                # 100 = perfect
def rejection(f: Features, spec: RouteSpec) -> str | None   # reason, or None if acceptable
def select_diverse(candidates, n, edge_len, max_overlap=0.4, skip_m=2000.0) -> list
# greedy by score; overlap = length-weighted Jaccard of edge sets, ignoring the first and last 2 km
```

### Task 4: Генератор

**Files:** Create `src/rpa/routing/generate.py`; Test `tests/test_generate.py` (синтетическая сетка 21×21, шаг 500 м)

```python
@dataclass
class Candidate: path: RoutePath; features: Features; score: float; note: str | None
def generate(engine: Engine, spec: RouteSpec) -> list[Candidate]
```

- Кольцо: правильный (k+1)-угольник с вершиной в старте, k ∈ {1, 2, 3}, 12 азимутов; периметр по прямой = L / 1.3; вершина привязывается к самому «приятному» узлу степени ≥3 в круге 0.25·сторона; до 5 итераций масштаба `scale *= L / факт`.
- A→B: если цели нет или она меньше кратчайшего — маршрут по предпочтениям + альтернативы методом штрафов (×3), с пометкой «минимум N км»; при `L < 1.15·d` — только альтернативы; иначе опорный узел там, где `metres_от_A + metres_до_B ≈ L`, по одному на сектор азимута от середины AB.
- Тесты-свойства: кольцо начинается и кончается в старте; дуги непрерывны; длина в диапазоне; нет пары «дуга, обратная дуга» подряд; все дуги разрешены базовыми весами; кандидаты попарно различны (перекрытие < 0.4); детерминизм.

### Task 5: eval

**Files:** Create `src/rpa/eval/cases.py`, `src/rpa/eval/run.py`, `src/rpa/viz/eval_page.py`; Modify `src/rpa/cli.py` (команда `eval`); Test `tests/test_eval.py`

`rpa eval --out out/eval.html`: по каждому кейсу — таблица (длина, ошибка, повторы, cost_ratio, повороты, время) с отметкой порогов S3 и карта кандидатов. Кейсы: кольца 45/80/120 км от центра Екатеринбурга с разными предпочтениями, кольцо 100–150 км `paved`, A→B Ревда → Екатеринбург на 80 км.

**Verify:** `uv run pytest -q`; `uv run rpa eval --out out/eval.html` — все кейсы проходят пороги S3, время на запрос ≤ 5 с.
