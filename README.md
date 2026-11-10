# ride-path-advisor

Личный сервис, который строит велосипедные маршруты по текстовому запросу
(«заезд около 45 км к воде, желательно по лесу, меньше больших дорог»).
Маршрут идёт по дорожному графу OpenStreetMap, а дороги из собственной истории
заездов получают бонус.

Дизайн: [`docs/superpowers/specs/2026-11-10-ride-path-advisor-design.md`](docs/superpowers/specs/2026-11-10-ride-path-advisor-design.md).

## Состояние

В разработке, этап M1: сборка графа, ядро поиска пути, маршрут A→B из командной строки.

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
