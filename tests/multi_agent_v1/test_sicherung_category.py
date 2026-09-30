"""Sicherung Kategorie 1-5: SicherungsAgent NEVER guesses a category. It
only reads one already written explicitly on the plan (deterministic regex,
zero model calls for the category decision itself), and RueckflussAgent
always reports NOT_ASSESSABLE with an honest reason rather than fabricating
an SVGW compliance verdict this codebase has no rule for."""
from __future__ import annotations

from app.multi_agent_v1.agents.sicherungs_agent import extract_explicit_category
from app.multi_agent_v1.agents.rueckfluss_agent import RueckflussAgent
from app.multi_agent_v1.provider import FixtureAgentModelProvider


def test_explicit_category_text_is_read_verbatim():
    cat, source = extract_explicit_category(["Sicherheitsventil Kategorie 3"])
    assert cat == 3
    assert source == "explicit_text_on_plan"


def test_explicit_category_matches_flk_abbreviation_and_umlaut_variants():
    assert extract_explicit_category(["FLK 4"]) == (4, "explicit_text_on_plan")
    assert extract_explicit_category(["Flüssigkeitskategorie 1"]) == (1, "explicit_text_on_plan")


def test_no_category_text_never_guessed():
    cat, source = extract_explicit_category(["Sicherheitsventil", "1/2 Zoll"])
    assert cat is None
    assert source == "no_explicit_category_text"


def test_out_of_range_numbers_are_not_mistaken_for_a_category():
    # "Kategorie 7" is not a valid liquid category (1-5) -- must not match
    # and silently produce a fabricated out-of-range category.
    cat, source = extract_explicit_category(["Kategorie 7"])
    assert cat is None


def test_rueckfluss_not_assessable_when_no_explicit_category():
    agent = RueckflussAgent(FixtureAgentModelProvider())
    obs = agent.assess(
        context=None, subject_id="INV1",
        sicherungseinrichtung={"liquid_category": None, "category_source": "no_explicit_category_text", "device_type": "sicherheitsventil"},
    )
    assert obs.value["result"] == "NOT_ASSESSABLE"
    assert obs.value["reason"] == "no_explicit_category_evidence"
    assert obs.model is None  # never called a model to guess a verdict


def test_rueckfluss_not_assessable_even_with_explicit_category_absent_a_rule():
    agent = RueckflussAgent(FixtureAgentModelProvider())
    obs = agent.assess(
        context=None, subject_id="INV1",
        sicherungseinrichtung={"liquid_category": 3, "category_source": "explicit_text_on_plan", "device_type": "systemtrenner"},
    )
    # A category IS documented, but this never becomes a fabricated
    # COMPLIANT/NON_COMPLIANT verdict -- no existing rule maps category ->
    # required device in this codebase.
    assert obs.value["result"] == "NOT_ASSESSABLE"
    assert obs.value["reason"] == "no_existing_category_requirement_rule"
    assert obs.value["documented_category"] == 3


def test_rueckfluss_agent_never_calls_the_model():
    provider = FixtureAgentModelProvider(responses={"anything": {"result": "COMPLIANT"}})
    agent = RueckflussAgent(provider)
    agent.assess(context=None, subject_id="INV1", sicherungseinrichtung={"liquid_category": 2, "category_source": "explicit_text_on_plan", "device_type": "x"})
    assert provider.calls == []
