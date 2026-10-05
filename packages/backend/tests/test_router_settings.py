import pytest
from pydantic import ValidationError

from app.config import Settings
from app.routing.from_settings import static_strategy_from_settings, tiers_from_settings
from app.routing.router import RoutingRequest, TaskType

TIERS = [
    {"name": "small", "model": "m-7b", "base_url": "http://gw/v1", "cost_per_1k_in": 0.1},
    {"name": "large", "model": "m-70b", "base_url": "http://gw/v1", "temperature": 0.2},
]


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def test_routing_is_off_by_default():
    s = _settings()
    assert s.router_tiers == [] and s.router_fallback_tier is None
    assert tiers_from_settings(s) == []


def test_tiers_load_and_convert():
    s = _settings(router_tiers=TIERS, router_fallback_tier="large",
                  router_task_tiers={"extraction": "small"})
    small, large = tiers_from_settings(s)
    assert (small.name, small.model, small.cost_per_1k_in) == ("small", "m-7b", 0.1)
    assert large.temperature == 0.2 and large.api_key is None


def test_static_strategy_uses_table_then_fallback():
    s = _settings(router_tiers=TIERS, router_fallback_tier="large",
                  router_task_tiers={"extraction": "small"})
    strategy = static_strategy_from_settings(s)
    req = lambda task: RoutingRequest(task, [])
    assert strategy.choose(req(TaskType.EXTRACTION), {}).tier == "small"
    assert strategy.choose(req(TaskType.SYNTHESIS), {}).tier == "large"


def test_tiers_parse_from_env_json(monkeypatch):
    monkeypatch.setenv(
        "ROUTER_TIERS", '[{"name":"only","model":"m","base_url":"http://gw/v1"}]'
    )
    monkeypatch.setenv("ROUTER_FALLBACK_TIER", "only")
    s = _settings()
    assert [t.name for t in s.router_tiers] == ["only"]


def test_fallback_must_name_a_tier():
    with pytest.raises(ValidationError):
        _settings(router_tiers=TIERS, router_fallback_tier="ghost")
    with pytest.raises(ValidationError):
        _settings(router_tiers=TIERS)


def test_duplicate_tier_names_rejected():
    with pytest.raises(ValidationError):
        _settings(router_tiers=[TIERS[0], TIERS[0]], router_fallback_tier="small")


def test_task_table_must_reference_known_tiers():
    with pytest.raises(ValidationError):
        _settings(router_tiers=TIERS, router_fallback_tier="large",
                  router_task_tiers={"extraction": "ghost"})


def test_unknown_task_type_rejected_when_building_strategy():
    s = _settings(router_tiers=TIERS, router_fallback_tier="large",
                  router_task_tiers={"summarising": "small"})
    with pytest.raises(ValueError, match="unknown task types"):
        static_strategy_from_settings(s)
