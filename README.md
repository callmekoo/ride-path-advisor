# ride-path-advisor

Личный сервис, который строит велосипедные маршруты по текстовому запросу
(«заезд около 45 км к воде, желательно по лесу, меньше больших дорог»).
Маршрут идёт по дорожному графу OpenStreetMap, а дороги из собственной истории
заездов получают бонус.

Состояние и планы: [`docs/roadmap.md`](docs/roadmap.md). Исходный дизайн:
[`docs/superpowers/specs/2026-11-10-ride-path-advisor-design.md`](docs/superpowers/specs/2026-11-10-ride-path-advisor-design.md).

## Состояние

Готовы все этапы плана (M1–M8): граф и ядро поиска пути, кольца и маршруты заданной длины,
учёт собственных заездов, вода, лес и высоты, разбор текстовых запросов, страница в браузере,
контейнер. Что известно о слабых местах — в [`docs/roadmap.md`](docs/roadmap.md).

## Установка в контейнере

Нужен только podman (или docker — команды те же). Логинов в сервисе нет, поэтому порт открыт
лишь на `127.0.0.1`; наружу его выставляйте только через VPN или обратный прокси с авторизацией.

```bash
podman build -t rpa .                  # образ: сервер и все пакетные команды

cp .env.example .env                   # адрес OpenAI-совместимого API, модель, ключ, при необходимости прокси
mkdir -p data

# Карта: выгрузки OSM, граф, указатель названий, вода, лес, высоты.
podman run --rm --userns=keep-id -v ./data:/data:Z rpa update

# Сервер: страница на http://127.0.0.1:8000/
podman run -d --name rpa --userns=keep-id -v ./data:/data:Z -p 127.0.0.1:8000:8000 --env-file .env rpa
```

Дальше всё делается на странице: вкладка «Места» — поставить точку и сохранить её как место
по умолчанию; вкладка «Треки» — загрузить архив Strava («Скачать свои данные») или отдельные
GPX/FIT/TCX. Заезды обрабатываются отдельным процессом минуту-две и учитываются без перезапуска.

- **Обновить карту** (раз в несколько месяцев): `podman run --rm --userns=keep-id -v ./data:/data:Z rpa update`,
  затем `podman restart rpa` — работающий сервер держит старый граф в памяти. Заезды переносятся
  на новый граф сами.
- **Любая команда** запускается так же: `podman run --rm --userns=keep-id -v ./data:/data:Z rpa eval`.
- Если прокси для модели слушает на самой машине (`127.0.0.1`), из контейнера он не виден:
  запускайте сервер с `--network=host` (тогда `-p` не нужен, а адрес задаёт `rpa serve --host`).
- Без `.env` сервис работает, но текст не разбирает: маршруты строятся только через API.

## Установка без контейнера

Нужны [uv](https://docs.astral.sh/uv/), Rust (через [rustup](https://rustup.rs/)), Node 24 и
`osmium-tool` (или podman — тогда `fetch-region` запустит osmium в контейнере сам).

```bash
uv sync                                  # зависимости и сборка Rust-ядра
(cd web && npm install && npm run build) # страница → web/dist
cp .env.example .env                     # настройки модели
uv run rpa update                        # карта и всё, что из неё строится
uv run rpa serve                         # http://127.0.0.1:8000/, описание API — на /docs
```

## Команды

`rpa update` и вкладка «Треки» — это цепочки из команд ниже; по отдельности они нужны при
разработке и разборе проблем.

```bash
uv run rpa fetch-region                        # выгрузки OSM → data/osm/region.osm.pbf
uv run rpa build-graph data/osm/region.osm.pbf # граф → data/graph/current
uv run rpa route --from 56.84,60.60 --to 56.80,59.93 --html out/ab.html
uv run rpa eval --out out/eval.html             # эталонные запросы → страница с картой и метриками

uv run rpa gazetteer                           # указатель названий мест для разбора запросов
uv run rpa nlu-eval                            # эталонные формулировки на настроенной модели

# Треки: положите архив «Скачать свои данные» из Strava (или GPX/FIT/TCX) в data/inbox/
uv run rpa sync-tracks                         # обе следующие команды подряд
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

```bash
uv run pytest -q                   # тесты Python, ~6 с
(cd web && npm test)               # тесты логики страницы
(cd web && npm run dev)            # страница с горячей заменой; /api проксируется на запущенный rpa serve
```

Если cargo за HTTPS-прокси не может проверить сертификат crates.io
(`unable to get local issuer certificate`), укажите ему системный набор сертификатов:

```bash
export CARGO_HTTP_CAINFO=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
```

## Устройство

- `kernel/` — Rust-ядро (`rpa_kernel`): поиск пути по вектору весов дуг.
- `src/rpa/` — Python: сборка графа из OSM, веса по предпочтениям, генерация маршрутов.
- `web/` — страница (Vite, React, TypeScript, MapLibre): `npm test`, `npm run build`;
  при разработке `npm run dev` рядом с запущенным `rpa serve` (запросы `/api` проксируются на порт 8000).
- `tests/` — тесты; ядро проверяется сравнением с Dijkstra из scipy.

## Данные

- Дороги, водоёмы: © участники OpenStreetMap (ODbL).
- Высоты: Copernicus DEM GLO-30 © DLR e.V. 2010–2014 и © Airbus Defence and Space GmbH 2014–2018,
  предоставлено по программе COPERNICUS Европейским союзом и ЕКА.
- Лес: © ESA WorldCover project 2021, содержит модифицированные данные Copernicus Sentinel (CC BY 4.0).
