"""Check the configured model against the reference phrasings."""

from dataclasses import dataclass

from rpa.nlu.cases import SAVED_PLACES, Case
from rpa.nlu.client import LlmError
from rpa.nlu.prompt import ParsedRequest, messages
from rpa.osm.gazetteer import same_name

_PLACE_FIELDS = {"start", "end", "via", "toward"}
PASS_SHARE = 0.9


@dataclass
class Outcome:
    case: Case
    problems: list[str]  # empty when the model understood the case as expected


def mismatches(parsed: ParsedRequest, expect: dict) -> list[str]:
    """Fields of `parsed` that are not what `expect` asks for, described for a person."""
    problems = []
    for name, wanted in expect.items():
        got = getattr(parsed, name)
        if not _matches(name, got, wanted):
            problems.append(f"{name}: ожидалось {wanted!r}, получено {got!r}")
    return problems


def _matches(name: str, got: object, wanted: object) -> bool:
    if isinstance(wanted, tuple) and wanted[0] == "in":
        return any(_matches(name, got, option) for option in wanted[1])
    if isinstance(wanted, tuple) and wanted[0] == "range":
        return isinstance(got, (int, float)) and not isinstance(got, bool) and wanted[1] <= got <= wanted[2]
    if name == "avoid":
        return len(got) == len(wanted) and all(any(same_name(g, w) for g in got) for w in wanted)
    if name in _PLACE_FIELDS and isinstance(wanted, str):
        return isinstance(got, str) and same_name(got, wanted)
    return got == wanted


def run_cases(llm, cases: list[Case]) -> list[Outcome]:
    outcomes = []
    for case in cases:
        try:
            parsed = llm.parse(*messages(case.text, case.previous, case.candidates, SAVED_PLACES), ParsedRequest)
            outcomes.append(Outcome(case, mismatches(parsed, case.expect)))
        except LlmError as error:
            outcomes.append(Outcome(case, [str(error)]))
    return outcomes
