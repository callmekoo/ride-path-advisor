# ride-path-advisor

Личный сервис, который строит велосипедные маршруты по текстовому запросу
(«заезд около 45 км к воде, желательно по лесу, меньше больших дорог»).
Маршрут идёт по дорожному графу OpenStreetMap, а дороги из собственной истории
заездов получают бонус.

Дизайн: [`docs/superpowers/specs/2026-11-10-ride-path-advisor-design.md`](docs/superpowers/specs/2026-11-10-ride-path-advisor-design.md).

## Состояние

Готов этап M1: сборка графа, ядро поиска пути, маршрут A→B из командной строки.
Дальше — M2: кольцевые маршруты заданной длины и страница оценки качества.

```bash
uv run rpa fetch-region                        # выгрузки OSM → data/osm/region.osm.pbf
uv run rpa build-graph data/osm/region.osm.pbf # граф → data/graph/current
uv run rpa route --from 56.84,60.60 --to 56.80,59.93 --html out/ab.html
```

Замеры на графе региона (Свердловская область + 100 км, выгрузки OSM от 09.11.2026;
16 ядер, один поток):

| | |
|---|---|
| Исходный `region.osm.pbf` | 210 МБ |
| Граф | 803 тыс. узлов, 1,06 млн рёбер, 243 тыс. км; 89 МБ на диске |
| Сборка графа | 12 с, пик памяти 0,85 ГБ |
| Поиск от точки до всех узлов | 123 мс (цель < 400 мс) |
| Маршрут 40–60 км | медиана 16 мс, p95 23 мс (цель < 30 мс) |
| Маршрут 120–180 км | медиана 29 мс, p95 40 мс |

Покрытие по длине дорог: paved 11%, likely_paved 4%, unknown 28%, likely_unpaved 38%,
unpaved 15%, bad 3% — тег `surface` в регионе заполнен редко, поэтому классы
«likely» и «unknown» выводятся из типа дороги.

## Разработка

Нужны [uv](https://docs.astral.sh/uv/) и Rust (через [rustup](https://rustup.rs/)).

```bash
uv sync            # ставит зависимости и собирает Rust-ядро
uv run pytest -q
```

Если cargo за HTTPS-прокси не может проверить сертификат crates.io
(`unable to get local issuer certificate`), укажите ему системный набор сертификатов:

```bash
export CARGO_HTTP_CAINFO=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
```

## Устройство

- `kernel/` — Rust-ядро (`rpa_kernel`): поиск пути по вектору весов дуг.
- `src/rpa/` — Python: сборка графа из OSM, веса по предпочтениям, генерация маршрутов.
- `tests/` — тесты; ядро проверяется сравнением с Dijkstra из scipy.

## Данные

© участники OpenStreetMap (ODbL).
