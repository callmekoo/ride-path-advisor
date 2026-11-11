"""What the language model is asked and what it answers.

The model turns a rider's words into a ParsedRequest: a flat structure in which
places are still names. It never sees coordinates or ride history; finding the
places and building the RouteSpec is the server's job (see resolve.py).
"""

from typing import Literal

from pydantic import BaseModel, Field

from rpa.routing.spec import RouteSpec

Direction = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIRECTIONS: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")  # 45 degrees apart, from north clockwise

# How a start taken from a click on the map is referred to in words.
CLICKED_POINT = "точка на карте"


class ParsedRequest(BaseModel):
    """A route request as understood from text. None everywhere means "not said"."""

    shape: Literal["loop", "p2p"] | None = None
    start: str | None = None
    end: str | None = None
    via: str | None = None
    via_water: bool | None = None
    length_km: float | None = Field(default=None, gt=0)
    length_min_km: float | None = Field(default=None, gt=0)
    length_max_km: float | None = Field(default=None, gt=0)
    avoid_major: float | None = Field(default=None, ge=0, le=1)
    surface: Literal["any", "paved", "prefer_unpaved"] | None = None
    forest: float | None = Field(default=None, ge=0, le=1)
    familiarity: float | None = Field(default=None, ge=0, le=1)
    hills: Literal["any", "avoid", "seek"] | None = None
    avoid: list[str] = []
    direction: Direction | None = None
    toward: str | None = None
    candidate: int | None = Field(default=None, ge=1)


SYSTEM = f"""\
Ты разбираешь запросы велосипедиста на построение маршрута и заполняешь структуру. \
Маршрут строит другая программа; твоя задача — только понять, что сказано. \
Поле, о котором в запросе ничего не сказано, оставляй null (для avoid — пустой список). Ничего не выдумывай.

Поля:
- shape: "loop" — кольцо, круг, заезд с возвращением в точку старта; "p2p" — из одной точки в другую («от … до …»). \
Если назван финиш, отличный от старта, это "p2p". «До X и обратно» — это "loop" с via = X.
- start, end, via, toward и элементы avoid — названия мест. Пиши их в именительном падеже, как на карте, \
без предлогов: «от Дидинского тоннеля» → "Дидинский тоннель", «до Ревды» → "Ревда". Слово, говорящее, что это \
за объект (озеро, река, станция, гора), сохраняй: "озеро Шарташ". Сохранённые места пользователя («дом», «дача» — \
их список дан в запросе) пиши так, как они названы в списке. Если пользователь говорит «отсюда» или «от точки на \
карте», пиши "{CLICKED_POINT}".
- start: только если старт назван явно. «Из дома» → "дом".
- via: место, через которое нужно проехать или до которого нужно доехать в кольце.
- via_water: true, если хочется к воде вообще («к воде», «к озеру», «искупаться»), а конкретный водоём не назван. \
Если водоём назван, это via, а via_water — null.
- length_km: желаемая длина в километрах («около 45 км», «на 80»). Диапазон «100–150 км» — это length_min_km и \
length_max_km, а length_km — null. «Не больше 60» — только length_max_km; «хотя бы 80» — только length_min_km. \
Время переводи в километры из расчёта 20 км/ч (25 км/ч для шоссейного заезда).
- avoid_major (0…1) — насколько избегать крупных дорог: «поменьше больших дорог», «меньше машин», «подальше от \
трасс» → 0.8; «совсем без трасс» → 1.0; «можно по трассе», «дороги не важны» → 0.2.
- surface: "paved" — шоссейный заезд, по асфальту; "prefer_unpaved" — гравийный, по грунтам, по лесным дорогам; \
"any" — если сказано, что покрытие не важно.
- forest (0…1) — насколько хочется ехать по лесу: «желательно по лесу», «через лес» → 0.7; «только лесом» → 1.0.
- familiarity (0…1) — насколько держаться знакомых дорог: «по знакомым местам» → 0.8; «что-нибудь новое», \
«где я ещё не был» → 0.0.
- hills: "avoid" — «без горок», «поровнее», «плоский»; "seek" — «с набором», «погорнее», «с подъёмами».
- avoid: места, которых нужно избегать: «не через Ревду», «в объезд Дегтярска» → ["Ревда"], ["Дегтярск"].
- direction: сторона света, куда ехать («на север» → "N", «куда-нибудь на юго-запад» → "SW").
- toward: место, в сторону которого ехать, не обязательно до него («в сторону Сысерти»).
- candidate: номер ранее предложенного варианта, если пользователь на него ссылается («вариант 2, но короче» → 2).

Уточнение. Если в запросе есть предыдущий запрос и предложенные варианты, новая реплика его уточняет. \
Верни полный новый запрос: поля, которых реплика не касается, перенеси из предыдущего без изменений, остальные \
измени. Поле start переноси только если оно было в предыдущем запросе. «Короче» или «длиннее» без числа — измени \
длину примерно на четверть относительно предыдущей (а если назван вариант — относительно длины этого варианта), \
запиши её в length_km, а length_min_km и length_max_km обнули. «Не обязательно к воде» и подобное — сними \
соответствующее условие. Если реплика явно начинает новый маршрут («а теперь кольцо на 100 км по шоссе»), \
старые условия не переноси.\
"""


def messages(text: str, previous: ParsedRequest | None, candidates: list[str], saved_places: list[str]) -> tuple[str, str]:
    """(system, user) messages for one request; `previous` and `candidates` make it a refinement."""
    parts = []
    if saved_places:
        parts.append("Сохранённые места: " + ", ".join(saved_places))
    if previous is not None:
        parts.append("Предыдущий запрос: " + previous.model_dump_json())
        if candidates:
            parts.append("Предложенные варианты:\n" + "\n".join(candidates))
        parts.append("Новая реплика: " + text)
    else:
        parts.append("Запрос: " + text)
    return SYSTEM, "\n\n".join(parts)


def describe(spec: RouteSpec) -> ParsedRequest:
    """A RouteSpec in the model's vocabulary, so that it can carry a request over into its refinement.

    The start is left out on purpose: it stays as it was unless the new words or a new click change it.
    """
    length = spec.length_km
    about = length is not None and abs(length.min - 0.9 * length.target) < 0.01 * length.target
    about = about and abs(length.max - 1.1 * length.target) < 0.01 * length.target
    return ParsedRequest(
        shape=spec.shape,
        end=spec.end.label if spec.end else None,
        via=spec.via.label if spec.via else None,
        via_water=True if spec.via_category == "water" else None,
        length_km=round(length.target, 1) if length and about else None,
        length_min_km=round(length.min, 1) if length and not about else None,
        length_max_km=round(length.max, 1) if length and not about else None,
        avoid_major=spec.prefs.avoid_major,
        surface=spec.prefs.surface,
        forest=spec.prefs.forest,
        familiarity=spec.prefs.familiarity,
        hills=spec.prefs.hills,
        avoid=[area.point.label for area in spec.avoid if area.point.label],
        direction=DIRECTIONS[round(spec.bearing_deg / 45) % 8] if spec.bearing_deg is not None else None,
    )
