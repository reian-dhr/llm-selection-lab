"""Tests for model_registry.py: the cost formula and the choice of models to run."""

from types import SimpleNamespace

import pytest

from model_selection_lab.models.model_registry import (
    MODELS,
    calculate_cost,
    display_name,
    select_models,
)

FAKE_MODELS = {
    "cheap-one": SimpleNamespace(tier="standard"),
    "big-one": SimpleNamespace(tier="premium"),
    "cheap-two": SimpleNamespace(tier="standard"),
    "helper-one": SimpleNamespace(tier="helper"),
}


def test_calculate_cost_example():
    cost = calculate_cost(
        input_tokens=2_000,
        output_tokens=500,
        input_price_per_m=0.10,
        output_price_per_m=0.50,
    )
    assert cost == pytest.approx(0.00045)


def test_select_models_leaves_premium_out_by_default():
    assert select_models(models=FAKE_MODELS) == ["cheap-one", "cheap-two"]


def test_select_models_includes_premium_when_asked():
    selected = select_models(include_premium=True, models=FAKE_MODELS)

    assert selected == ["cheap-one", "big-one", "cheap-two"]


def test_a_default_run_uses_standard_models_only():
    # Guards the API budget: nothing expensive may run unless it is asked for.
    for model_key in select_models():
        assert MODELS[model_key].tier == "standard", model_key


def test_expensive_models_are_marked_premium():
    # Any model above the price of the dearest standard model today must be premium.
    for model_key, model in MODELS.items():
        if model.output_price_per_mtok > 10.00:
            assert model.tier == "premium", model_key


def test_every_registry_model_has_a_known_tier():
    for model_key, model in MODELS.items():
        assert model.tier in ("standard", "premium", "helper"), model_key


def test_helper_models_are_never_compared():
    assert "helper-one" not in select_models(models=FAKE_MODELS)
    assert "helper-one" not in select_models(include_premium=True, models=FAKE_MODELS)


def test_display_name_falls_back_to_the_key():
    assert display_name("haiku-5-5") == "Claude Haiku 5.5"
    assert display_name("unknown-model") == "unknown-model"
