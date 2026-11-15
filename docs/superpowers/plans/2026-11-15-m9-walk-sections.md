# M9. Пешие участки — план реализации

> **Для исполнителя:** скилл `superpowers:subagent-driven-development`, тест раньше кода.
> План содержит задачи, интерфейсы, тесты и команды проверки, без кода реализации.
> Дизайн: `docs/superpowers/specs/2026-11-15-walk-via-around-design.md`.

**Цель:** лестницы и другие участки, где велосипед ведут рядом, входят в граф; маршруты могут
через них идти со штрафом; заезды по ним видны на карте заездов.

**Архитектура:** `classify_way` перестаёт выбрасывать пешие пути и помечает их флагом
`DISMOUNT`; острова, связанные с сетью только лестницами, сами входят в главную компоненту.
Веса берут с таких рёбер множитель вместо запрета. Заезды пересчитываются из уже готового
`data/match/matched.npz` при пересборке графа — Valhalla не запускается.

**Стек:** без новых зависимостей. Python (numpy, pydantic), TypeScript (MapLibre).

## Общие ограничения

- Инвариант ядра: вес дуги равен `+inf` либо не меньше длины ребра. Все новые множители ≥ 1.
- Не читать `.env`; не печатать `/api/heatmap` и содержимое `data/match`, `data/tracks` —
  только счётчики и коды ответа.
- В тестах и эталонных запросах — только публичные точки.
- Числа штрафов — стартовые, как остальные веса в `routing/cost.py`.
- Код, комментарии, сообщения коммитов — на английском; коммит заканчивается строкой
  `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- `tests/fixtures/micro.opl` не менять: на её точных составах узлов и рёбер стоят тесты
  `test_build.py`, `test_cli.py`, `test_gazetteer.py`, `test_water.py`.

## Отличие от дизайна

`bicycle=no` на footway / path / pedestrian **остаётся** `SOFT_RESTRICTED`, а не становится
`DISMOUNT`: такая дорожка физически проезжаема, и сейчас два заезда снимают с неё штраф
полностью. Перевод в «пешком» сделал бы такие дорожки дороже, чем сегодня, — обратное тому,
о чём просили.

## Файлы

| Файл | Ответственность |
|---|---|
| `src/rpa/osm/tags.py` | `Highway.STEPS`, флаг `DISMOUNT`, правила пеших путей |
| `src/rpa/graph/store.py` | `FORMAT_VERSION = 2` |
| `src/rpa/routing/cost.py` | штраф за пешие рёбра |
| `src/rpa/routing/path.py`, `score.py` | пешие отрезки и метры пешком в описании маршрута |
| `src/rpa/api/schemas.py`, `app.py` | `walk_spans` в ответе |
| `web/src/geo.ts`, `api.ts`, `App.tsx`, `components/CandidateCard.tsx`, `components/MapView.tsx` | «пешком N м», пунктир на карте |
| `docs/roadmap.md`, `README.md`, `CLAUDE.md` | состояние после этапа |

---

### Задача 1. Пешие пути в графе — *Sonnet*

**Файлы:** `src/rpa/osm/tags.py`, `src/rpa/graph/store.py`, `tests/test_tags.py`,
`tests/test_build.py`.

**Интерфейс (его используют задачи 2 и 3):**
- `Highway.STEPS = 13` — только `highway=steps`.
- `DISMOUNT = 128` — флаг ребра: велосипед здесь ведут рядом.
- `FORMAT_VERSION = 2` в `graph/store.py`.

**Правила `classify_way`:**

| Теги | Результат |
|---|---|
| `highway=steps` | `Highway.STEPS`, `DISMOUNT` — при любых `bicycle` / `access`, кроме закрытого доступа ниже |
| `highway=corridor`, `highway=platform` | `Highway.FOOTWAY`, `DISMOUNT` |
| `bicycle=dismount` на любом классе | `DISMOUNT` вместо `SOFT_RESTRICTED` |
| `access=no\|military` и `foot=yes\|designated\|permissive`, `bicycle` не разрешён | путь остаётся, `DISMOUNT` (сейчас выбрасывается) |
| `access=no\|military` без разрешённого `foot` и `bicycle` | выбрасывается, как сейчас — в том числе лестница |
| `bicycle=no`, `use_sidepath`, `access=private…`, `vehicle=no` | без изменений: `SOFT_RESTRICTED` |

- `DISMOUNT` и `SOFT_RESTRICTED` вместе не ставятся: пешее ребро получает только `DISMOUNT`.
- `oneway` на `STEPS` игнорируется (добавить в `_ONEWAY_IGNORED_ON`).
- `CYCLE_INFRA` на ребре с `DISMOUNT` не ставится.
- Покрытие лестницы — по общим правилам (`Surface.UNKNOWN` без тегов).
- `winter_road`, `motorroad`, `construction`, `motorway` по-прежнему выбрасываются.

**Тесты (`tests/test_tags.py`):**
- из `NOT_ROUTABLE` уходят `steps` и `platform`; добавляются `{"highway": "steps", "access": "no"}`
  и `{"highway": "footway", "access": "no"}` (нет `foot`);
- `steps` → `Highway.STEPS`, флаги содержат `DISMOUNT` и оба направления, не содержат
  `SOFT_RESTRICTED`; то же при `bicycle=yes` и при `oneway=yes`;
- `corridor`, `platform` → `Highway.FOOTWAY` с `DISMOUNT`;
- `footway` + `bicycle=dismount` → `DISMOUNT` без `SOFT_RESTRICTED`;
- `footway` + `access=no` + `foot=yes` → остаётся, `DISMOUNT`;
- `footway` + `bicycle=no` → по-прежнему `SOFT_RESTRICTED`, без `DISMOUNT`;
- обычный `footway` → `SOFT_RESTRICTED`, без `DISMOUNT` (не изменилось).

**Тест (`tests/test_build.py`)** — на карте, записанной в `tmp_path` (формат `.opl`, как в
`tests/fixtures/micro.opl`), а не на `micro.opl`: улица `residential` A–B, лестница B–C,
мост `footway` + `bridge=yes` C–D, лестница D–E, улица `residential` E–F. Ожидается: все шесть
узлов в графе (один компонент), рёбра лестниц несут `Highway.STEPS` и `DISMOUNT`, ребро моста —
`BRIDGE`. Второй вариант той же карты без лестниц (пути B–C и D–E убраны): моста в графе нет.

**Проверка:** `uv run pytest -q tests/test_tags.py tests/test_build.py tests/test_store.py tests/test_cli.py`
— зелёные; затем `uv run pytest -q` — падает только импорт/построение `routing/cost.py`
(`KeyError` на `Highway.STEPS` в `_PENALTY_BY_CLASS`), это чинит задача 2. Чтобы коммит задачи 1
не ломал остальные тесты, в этой же задаче добавить в `_ROAD_CLASS_PENALTY` запись
`Highway.STEPS: 0.3` (как у `FOOTWAY`) — и тогда `uv run pytest -q` зелёный целиком.

### Задача 1б. Железнодорожные платформы — *Sonnet*

Добавлена по итогам проверки на реальных данных: лестницы моста у Перегона упираются в
платформы, размеченные `railway=platform` + `public_transport=platform` без тега `highway`.
Сборка читает только пути с `highway`, поэтому мост оставался островом. В регионе таких
платформ 1694, из них 544 — замкнутые контуры.

**Файлы:** `src/rpa/osm/tags.py`, `src/rpa/osm/build.py`, `tests/test_tags.py`,
`tests/test_build.py`.

**Правило `classify_way`:** путь без тега `highway`, у которого `railway=platform` или
`public_transport=platform`, классифицируется так же, как `highway=platform`:
`Highway.FOOTWAY`, `DISMOUNT`, оба направления; остальные правила (закрытый доступ, покрытие,
мост, тоннель) действуют как для любого пути. Если `highway` есть, он главный: `railway` и
`public_transport` не смотрятся. Прочие пути без `highway` (`railway=rail` и т. п.)
по-прежнему дают `None`.

**`read_ways` (`osm/build.py`):** фильтр ключей пропускает пути с `highway`, `railway` или
`public_transport`; решение по-прежнему принимает `classify_way`. Замкнутый контур платформы
обрабатывается как любой замкнутый путь — отдельной логики для площадей нет.

**Тесты (`tests/test_tags.py`):**
- `{"railway": "platform"}`, `{"public_transport": "platform"}`, оба тега вместе →
  `Highway.FOOTWAY`, `DISMOUNT`, оба направления, без `SOFT_RESTRICTED` и `CYCLE_INFRA`;
- `{"railway": "platform", "surface": "paved"}` → `Surface.PAVED`;
- `{"railway": "platform", "access": "no"}` → `None`; с `foot=yes` → остаётся;
- `{"railway": "rail"}`, `{"railway": "station"}`, `{"public_transport": "stop_position"}` →
  `None`;
- `{"highway": "residential", "public_transport": "platform"}` → `Highway.RESIDENTIAL` без
  `DISMOUNT`.

**Тест (`tests/test_build.py`)** на карте в `tmp_path`: улица A–B–B2, лестница B–C,
платформа C–D (`railway=platform`, без `highway`), лестница D–E, улица E–F–F2; рядом рельсы
`railway=rail` через собственные узлы. Ожидается: узлы платформы в графе, её ребро несёт
`Highway.FOOTWAY` и `DISMOUNT`, рельсов в графе нет. Заодно сделать улицы в существующем
тесте «мост без лестниц — остров» разной длины (на одной — лишний узел и ребро) и проверять,
что узлы более длинной улицы остались: сейчас тест зависит от того, какую из равных компонент
выберет сборка.

**Проверка:** `uv run pytest -q tests/test_tags.py tests/test_build.py tests/test_cli.py tests/test_gazetteer.py tests/test_water.py`,
затем `uv run pytest -q`.

### Задача 2. Штраф за пешие рёбра — *Sonnet*

**Файлы:** `src/rpa/routing/cost.py`, `tests/test_cost.py`.

**Интерфейс:**
- `DISMOUNT_FACTOR = 8.0` — множитель ограничения для ребра с `DISMOUNT`.
- `WALKED_DISMOUNT_FACTOR = 3.0` — он же, когда `rides[e] ≥ RIDES_LIFTING_RESTRICTION`.
- Множитель стоит на месте `SOFT_RESTRICTION_FACTOR` в формуле (умножается на покрытие и
  брод); остальные слагаемые (класс дороги, незнакомость, лес, подъём) действуют как обычно;
  потолок `MAX_MULTIPLIER` прежний. Без `rides` всегда `DISMOUNT_FACTOR`.
- Докстринг модуля: упомянуть пешие участки в `restriction_factor`.

**Тесты (`tests/test_cost.py`, приём `with_edge` + `multiplier` из этого файла):**
- ребро с `DISMOUNT` на `Highway.CYCLEWAY` / `Surface.PAVED`, предпочтения без незнакомости и
  леса: множитель ровно `DISMOUNT_FACTOR`;
- то же при двух заездах по ребру: ровно `WALKED_DISMOUNT_FACTOR`; при одном — `DISMOUNT_FACTOR`;
- `DISMOUNT` вместе с `FORD`: произведение множителей;
- вес пешего ребра конечен в обоих направлениях и не меньше длины ребра во всех режимах
  покрытия, кроме `paved` + `Surface.BAD` (там `+inf`, как для любого ребра);
- при всех штрафах сразу множитель не превышает `MAX_MULTIPLIER`.

**Проверка:** `uv run pytest -q tests/test_cost.py tests/test_kernel.py`.

### Задача 3. Метры пешком и пешие отрезки в описании маршрута — *Sonnet*

**Файлы:** `src/rpa/routing/path.py`, `src/rpa/routing/score.py`, `src/rpa/api/schemas.py`,
`src/rpa/api/app.py`, `tests/test_path.py`, `tests/test_score.py`, `tests/test_api.py`.

**Интерфейс (его использует задача 4):**
- `RoutePath.walk_spans: list[tuple[int, int]]` — пары индексов `(первая, последняя)` в
  `coords`, по одной на каждую непрерывную череду рёбер с `DISMOUNT`; пусто, если таких нет.
  Соседние пешие рёбра сливаются в один отрезок.
- `Features.walk_m: float = 0.0` — сумма длин рёбер с `DISMOUNT` на маршруте (ребро,
  пройденное дважды, считается дважды).
- `CandidateBody.walk_spans: list[tuple[int, int]] = []`. `walk_m` попадает в
  `CandidateBody.features` сам — там `asdict(Features)`.
- Оценка (`score`) и отбраковка (`rejection`) не меняются: пешие метры уже учтены в
  `cost_ratio`.
- `_candidate_line` в `api/app.py` (строка для модели) не меняется.

**Тесты:**
- `tests/test_path.py`: на копии графа с `DISMOUNT` на одном ребре в середине маршрута —
  один отрезок, индексы указывают ровно на координаты этого ребра (в том числе при проходе
  ребра в обратном направлении); два соседних пеших ребра дают один отрезок; пешее ребро в
  начале маршрута начинается с индекса 0; без пеших рёбер — пусто.
- `tests/test_score.py`: `walk_m` равен длине пешего ребра; 0 без пеших рёбер.
- `tests/test_api.py`: `POST /api/route` на графе с пешим ребром на пути — у кандидата
  `features.walk_m > 0` и непустой `walk_spans`, каждый индекс меньше числа координат
  геометрии; на обычном графе `walk_spans == []`.

**Проверка:** `uv run pytest -q tests/test_path.py tests/test_score.py tests/test_api.py tests/test_gpx.py tests/test_db.py`.

### Задача 4. Страница: «пешком N м» и пунктир — *Sonnet*

**Файлы:** `web/src/api.ts`, `web/src/geo.ts`, `web/src/geo.test.ts`, `web/src/App.tsx`,
`web/src/components/CandidateCard.tsx`, `web/src/components/MapView.tsx`.

**Интерфейс:**
- `api.ts`: у кандидата `walk_spans: [number, number][]` (в старых сохранённых маршрутах поля
  может не быть — читать как пустой список).
- `geo.ts`: `walkLines(coordinates, spans) → [number, number][][]` — по ломаной на отрезок;
  отрезок с индексами за пределами координат пропускается.
- `CandidateCard.tsx`: при `features.walk_m ≥ 1` — отдельная строка «пешком N м» (целые
  метры; от 1000 м — через `formatKm`).
- `MapView.tsx`: новое свойство `walk` (ломаные выбранного маршрута), источник `map-walk` и
  слой `map-routes-walk` — пунктир светлым поверх линии выбранного маршрута; пустой список
  очищает слой. `App.tsx` передаёт `walkLines` выбранного кандидата, в том числе для открытого
  сохранённого маршрута.

**Тесты (`web/src/geo.test.ts`):** один отрезок; два отрезка; пустой список; отрезок за
пределами координат пропущен.

**Проверка:** `cd web && npm test && npm run build`.

### Задача 5. Реальные данные — *основная сессия*

Делает основная сессия, без субагента: работа с `data/` и правила приватности.

1. `uv run rpa update --skip-fetch` — новый каталог `data/graph/<sha12>-v2`, указатель,
   вода, лес, высоты. Заезды пересчитываются из `data/match/matched.npz` сами.
2. Сверка (скрипт без записи в файлы; печатает только счётчики):
   - путь моста у Перегона (OSM 35914954) и три лестницы (35914543, 35914956, 830411912)
     есть в `edge_way_id.npy` нового графа;
   - число привязанных путей вне графа: было 423, ожидается порядка десятков
     (остаются `access=no` и настоящие острова) — напечатать разбивку по классу;
   - число рёбер с `DISMOUNT`, их суммарная длина, сколько из них с заездами ≥ 1 и ≥ 2.
3. `uv run rpa eval` — код возврата 0; сравнить времена с прежними (0,5–0,8 с на запрос).
4. Маршрут через пути у Перегона: выбрать по OSM два публичных узла дорог по разные стороны
   путей в 200–400 м от моста; `uv run rpa route` между ними — в сводке должны быть
   километры `steps`. Если генератор даёт здесь ≥ 2 варианта и проходит пороги — добавить
   эталонный запрос `p2p-peregon-bridge` в `src/rpa/eval/cases.py`; если нет — записать
   команду проверки в `docs/roadmap.md`, а в `CASES` не добавлять.
5. `uv run rpa serve`, затем `curl --noproxy '*'`: `/api/meta` — `ridden_km` вырос
   относительно прежнего; `/api/heatmap` — код 200 и размер ответа (тело не печатать);
   `POST /api/route` с точками из п. 4 — `features.walk_m > 0`, `walk_spans` непуст.
6. Страница в headless Chromium (как в M7): карточка показывает «пешком N м», пунктир виден
   на снимке **с выключенным слоем «Мои заезды»**.

Если маршруты начали без нужды срезать через лестницы (эталоны `loop-*` получили пешие метры
там, где раньше их не было, и стали хуже на вид) — поднять `DISMOUNT_FACTOR`, записать новое
число в roadmap.

### Задача 5а. Без заданной длины маршруты сравниваются по полной стоимости — *Sonnet*

Добавлена по итогам проверки на реальных данных. От Палкинского Торфяника до Каменных палаток
прямой маршрут — 3,35 км (108 м пешком через мост у Перегона), без моста — 15,3 км; но
`generate` отдавал 17–30 км, а прямой вариант терял. Причина: `score` судит только о качестве
на километр, и при незаданной длине короткий маршрут с лестницей (дорогие метры, много
поворотов на 10 км) проигрывает длинным объездам. Для запроса без длины лучший маршрут —
самый дешёвый по сумме весов: его и выбрал поиск.

**Файлы:** `src/rpa/routing/score.py`, `src/rpa/routing/generate.py`, `src/rpa/eval/cases.py`,
`src/rpa/eval/run.py`, `tests/test_score.py`, `tests/test_generate.py`, `tests/test_eval.py`.

**Интерфейс:**
- `Features.cost_m` — свойство: `cost_ratio * length_m` (сумма весов дуг маршрута).
- `score(f, length, cheapest_cost_m: float | None = None)`. Когда `length is None` и
  `cheapest_cost_m` задан, из оценки вычитается
  `EXTRA_COST_PENALTY * (1 - cheapest_cost_m / f.cost_m)`, `EXTRA_COST_PENALTY = 60.0`.
  У самого дешёвого вычет нулевой. При заданной длине аргумент не действует. Маршрут нулевой
  длины вычета не получает.
- Штраф за повороты ограничен сверху: `MAX_TURN_PENALTY = 20.0` (формула та же, сверху —
  потолок). Иначе короткий маршрут с двумя поворотами получает сотни «поворотов на 10 км».
- `generate`: сначала измеряются и проходят отбраковку все маршруты, затем — если
  `spec.length_km is None` (в том числе когда длину снял `note`) — берётся наименьшая
  `cost_m` среди прошедших и передаётся в `score`. Остальное (`select_diverse`, профиль)
  без изменений.
- `eval.cases.Case.expect_walk: bool = False`; `eval.run.check`: при `expect_walk` и
  непустом списке кандидатов — проблема «№1: нет пешего участка», если у первого кандидата
  `features.walk_m == 0`.
- Новый эталонный запрос в `CASES`: имя `p2p-peregon-bridge`, текст «от Палкинского
  Торфяника до Палкинских каменных палаток», `RouteSpec(shape="p2p")` без длины, старт
  `Point(lon=60.3839, lat=56.8496, label="Палкинский Торфяник")`, финиш
  `Point(lon=60.3865, lat=56.8700, label="Палкинские каменные палатки")`, `expect_walk=True`.
  Обе точки — публичные объекты OSM.

**Тесты:**
- `tests/test_score.py`: без длины маршрут вдвое дороже самого дешёвого получает на
  `EXTRA_COST_PENALTY / 2` меньше, чем без аргумента; самый дешёвый — без вычета; при заданной
  длине аргумент ничего не меняет; штраф за повороты не превышает `MAX_TURN_PENALTY`
  (`Features` с очень большим `turns_per_10km`); `cost_m` равен сумме весов дуг.
- `tests/test_generate.py`, на сетке: копия сетки, где все рёбра, пересекающие одну
  вертикаль, кроме нескольких у самого края, несут `DISMOUNT` (`helpers.with_edge` меняет одно
  ребро — для набора рёбер сделать копию массива флагов тем же приёмом); запрос A→B без длины
  между соседними узлами по разные стороны этой вертикали в середине сетки. Ожидается: первый
  кандидат — самый короткий из отданных и с `walk_m > 0`. До правки тест должен падать.
- `tests/test_eval.py`: `check` сообщает о проблеме, когда `expect_walk` и у первого кандидата
  нет пеших метров; молчит, когда они есть; имена эталонных запросов по-прежнему уникальны.

**Проверка:** `uv run pytest -q tests/test_score.py tests/test_generate.py tests/test_eval.py`,
затем `uv run pytest -q`. `uv run rpa eval` на реальном графе запускает основная сессия.

### Задача 6. Документы и слияние — *основная сессия*

- `docs/roadmap.md`: строка M9 в таблице этапов; «Что построено» (граф: лестницы больше не
  выбрасываются, флаг `DISMOUNT`; веса: множители 8 и 3; формат графа v2); «Данные»
  (`-v2`); числа из задачи 5; «Отличия кода от дизайна» — про `bicycle=no`; раздел «Что
  дальше» — M10, M11 со ссылкой на дизайн.
- `README.md`, раздел «Состояние».
- `CLAUDE.md`: ничего, если команды не менялись.
- `superpowers:finishing-a-development-branch`: слияние fast-forward в `main`, удаление ветки.
- Остановка и отчёт.

## Порядок и параллельность

Задача 1 — первой. Затем 2, 3 и 4 можно вести одновременно: файлы не пересекаются, интерфейсы
заданы выше (`DISMOUNT`, `Highway.STEPS`, `walk_spans`, `walk_m`). Задачи 5 и 6 — после всех.
