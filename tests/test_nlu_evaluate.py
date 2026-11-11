from rpa.nlu.cases import CASES, Case
from rpa.nlu.client import LlmError
from rpa.nlu.evaluate import mismatches, run_cases
from rpa.nlu.prompt import ParsedRequest


def test_expectations_compare_values_ranges_alternatives_and_place_names():
    parsed = ParsedRequest(shape="p2p", start="Дидинский Тоннель", length_km=44, avoid=["город Ревда"], via_water=None)

    assert mismatches(parsed, {"shape": "p2p", "start": "Дидинский тоннель", "length_km": ("range", 40, 50)}) == []
    assert mismatches(parsed, {"avoid": ["Ревда"], "via_water": ("in", [None, False]), "end": None}) == []
    problems = mismatches(parsed, {"shape": "loop", "start": "Ревда", "length_km": ("range", 50, 60), "avoid": []})
    assert [problem.split(":")[0] for problem in problems] == ["shape", "start", "length_km", "avoid"]
    assert mismatches(ParsedRequest(), {"length_km": ("range", 1, 2), "via_water": True}) != []


def test_reference_cases_only_expect_fields_that_exist():
    assert len(CASES) >= 40
    for case in CASES:
        assert set(case.expect) <= set(ParsedRequest.model_fields), case.text
        ParsedRequest(**{k: v for k, v in case.expect.items() if not isinstance(v, tuple)})  # values are valid


def test_cases_run_through_the_model_and_failures_are_reported():
    class Llm:
        def __init__(self) -> None:
            self.asked = []

        def parse(self, system, user, model):
            self.asked.append(user)
            if "третий" in user:
                raise LlmError("Модель недоступна")
            return model(length_km=45)

    llm = Llm()
    previous = ParsedRequest(length_km=60)
    cases = [
        Case("кольцо 45", {"length_km": 45}),
        Case("короче", {"length_km": 30}, previous, ["1. Кольцо 60 км"]),
        Case("третий", {}),
    ]
    first, second, third = run_cases(llm, cases)

    assert first.problems == [] and len(second.problems) == 1 and third.problems == ["Модель недоступна"]
    assert "Сохранённые места: дом, дача" in llm.asked[0] and "Новая реплика: короче" in llm.asked[1]
