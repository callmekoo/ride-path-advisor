# ride-path-advisor

Личный сервис, который строит велосипедные маршруты по текстовому запросу
(«заезд около 45 км к воде, желательно по лесу, меньше больших дорог»).
Маршрут идёт по дорожному графу OpenStreetMap, а дороги из собственной истории
заездов получают бонус.

Дизайн: [`docs/superpowers/specs/2026-11-10-ride-path-advisor-design.md`](docs/superpowers/specs/2026-11-10-ride-path-advisor-design.md).

## Состояние

Готовы этапы M1 (граф, ядро поиска пути, маршрут A→B), M2 (кольца заданной длины,
A→B с целевой длиной, 2–3 разных варианта, страница оценки качества) M3 (импорт треков,
привязка к дорогам, предпочтение знакомых дорог) и M4 (маршруты к воде, предпочтение леса,
высоты и набор).
Дальше — M5: веб-API.

```bash
uv run rpa fetch-region                        # выгрузки OSM → data/osm/region.osm.pbf
uv run rpa build-graph data/osm/region.osm.pbf # граф → data/graph/current
uv run rpa route --from 56.84,60.60 --to 56.80,59.93 --html out/ab.html
uv run rpa eval --out out/eval.html             # эталонные запросы → страница с картой и метриками

# Треки: положите архив «Скачать свои данные» из Strava (или GPX/FIT/TCX) в data/inbox/
uv run rpa import-tracks                       # чистка и нарезка → data/tracks
uv run rpa match                               # привязка к дорогам → заезды по рёбрам графа
uv run rpa pop-map                             # карта проеханных дорог → out/heatmap.html
uv run rpa enrich                              # вода, лес, высоты (после каждой пересборки графа; ~2 мин, нужна сеть)
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
| Запрос на 3 варианта кольца (36 попыток, 8 потоков) | 0,5–0,6 с (цель ≤ 5 с) |

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

- Дороги, водоёмы: © участники OpenStreetMap (ODbL).
- Высоты: Copernicus DEM GLO-30 © DLR e.V. 2010–2014 и © Airbus Defence and Space GmbH 2014–2018,
  предоставлено по программе COPERNICUS Европейским союзом и ЕКА.
- Лес: © ESA WorldCover project 2021, содержит модифицированные данные Copernicus Sentinel (CC BY 4.0).
