"""Pytest unit tests for the actuarial risk engine.

Validates the severity-index maths ``S_claim = sum w_i * r_i ** gamma``,
edge cases, and the triage thresholds. Run with::

    pytest -q
"""

from __future__ import annotations

import math

import pytest

from src.actuarial.risk_engine import (
    ActuarialConfig,
    ClaimAssessment,
    DamageComponent,
    RiskEngine,
    TriageDecision,
    load_actuarial_config,
    severity_index,
)


@pytest.fixture()
def engine() -> RiskEngine:
    # Weights sum to 1 already, so normalisation is a no-op here.
    return RiskEngine(ActuarialConfig(weights={"roof": 0.5, "wall": 0.3, "other": 0.2}, gamma=1.5))


# --- severity maths -------------------------------------------------------

def test_severity_single_component(engine: RiskEngine) -> None:
    severity, contrib = engine.compute_severity([DamageComponent("roof", 0.25)])
    assert severity == pytest.approx(0.5 * 0.25 ** 1.5)
    assert contrib["roof"] == pytest.approx(severity)


def test_severity_is_sum_of_weighted_powers(engine: RiskEngine) -> None:
    comps = [DamageComponent("roof", 0.4), DamageComponent("wall", 0.1)]
    severity, _ = engine.compute_severity(comps)
    expected = 0.5 * 0.4 ** 1.5 + 0.3 * 0.1 ** 1.5
    assert severity == pytest.approx(expected)


def test_severity_standalone_matches_engine() -> None:
    assert severity_index([0.4, 0.1], [0.5, 0.3], gamma=1.5) == pytest.approx(
        0.5 * 0.4 ** 1.5 + 0.3 * 0.1 ** 1.5
    )


def test_gamma_penalises_large_damage() -> None:
    # Same total area: one big component scores higher than two small ones.
    big = severity_index([0.8], [1.0], gamma=2.0)
    split = severity_index([0.4, 0.4], [0.5, 0.5], gamma=2.0)
    assert big > split
    assert big == pytest.approx(0.64)
    assert split == pytest.approx(0.16)


def test_severity_empty_components_is_zero(engine: RiskEngine) -> None:
    severity, contrib = engine.compute_severity([])
    assert severity == 0.0
    assert contrib == {}


def test_severity_zero_ratios_is_zero(engine: RiskEngine) -> None:
    severity, _ = engine.compute_severity([DamageComponent("roof", 0.0)])
    assert severity == 0.0


def test_severity_full_damage_bounded_by_weights(engine: RiskEngine) -> None:
    # Every r_i = 1 and weights sum to 1 -> S_claim = 1.
    comps = [DamageComponent(name, 1.0) for name in ("roof", "wall", "other")]
    severity, _ = engine.compute_severity(comps)
    assert severity == pytest.approx(1.0)


def test_unknown_component_falls_back_to_other(engine: RiskEngine) -> None:
    severity, contrib = engine.compute_severity([DamageComponent("vehicle_body", 0.5)])
    assert severity == pytest.approx(0.2 * 0.5 ** 1.5)
    assert "vehicle_body" in contrib


def test_per_component_cap_is_applied() -> None:
    eng = RiskEngine(ActuarialConfig(weights={"roof": 1.0}, gamma=1.0, caps={"roof": 0.3}))
    severity, _ = eng.compute_severity([DamageComponent("roof", 0.9)])
    assert severity == pytest.approx(0.3)


def test_weights_are_normalised() -> None:
    eng = RiskEngine(ActuarialConfig(weights={"roof": 2.0, "wall": 2.0}, gamma=1.0))
    assert eng.weights == pytest.approx({"roof": 0.5, "wall": 0.5})


# --- validation / edge cases ----------------------------------------------

def test_area_ratio_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        DamageComponent("roof", 1.2)
    with pytest.raises(ValueError):
        DamageComponent("roof", -0.1)


def test_severity_index_length_mismatch() -> None:
    with pytest.raises(ValueError):
        severity_index([0.5], [0.5, 0.5])


def test_severity_index_rejects_negative_weight_and_ratio() -> None:
    with pytest.raises(ValueError):
        severity_index([0.5], [-1.0])
    with pytest.raises(ValueError):
        severity_index([1.5], [1.0])


def test_config_rejects_bad_gamma_and_thresholds() -> None:
    with pytest.raises(ValueError):
        ActuarialConfig(weights={"a": 1.0}, gamma=0)
    with pytest.raises(ValueError):
        ActuarialConfig(weights={"a": 1.0}, tau_fraud=1.5)
    with pytest.raises(ValueError):
        ActuarialConfig(weights={})


def test_load_actuarial_config_from_repo_yaml(tmp_path=None) -> None:
    # The shipped config must load and keep S_claim in [0, 1] for full damage.
    config = load_actuarial_config("configs/actuarial_weights.yaml")
    eng = RiskEngine(config)
    comps = [DamageComponent(name, 1.0) for name in config.weights]
    severity, _ = eng.compute_severity(comps)
    assert 0.0 < severity <= 1.0 + 1e-9
    assert math.isclose(sum(eng.weights.values()), 1.0)


# --- triage thresholds ------------------------------------------------------

def test_fraud_overrides_high_cost(engine: RiskEngine) -> None:
    # Even a very expensive claim is audited, not merely reviewed, when tampered.
    assessment = engine.assess([DamageComponent("roof", 1.0)], p_tamper=0.9)
    assert assessment.decision is TriageDecision.REJECT_FORENSIC_AUDIT


def test_fraud_threshold_is_inclusive(engine: RiskEngine) -> None:
    assessment = engine.assess([], p_tamper=engine.config.tau_fraud)
    assert assessment.decision is TriageDecision.REJECT_FORENSIC_AUDIT


def test_high_cost_goes_to_manual_review(engine: RiskEngine) -> None:
    assessment = engine.assess([DamageComponent("roof", 1.0)], p_tamper=0.1)
    assert assessment.decision is TriageDecision.MANUAL_REVIEW
    assert assessment.severity >= engine.config.tau_cost


def test_cost_threshold_is_inclusive(engine: RiskEngine) -> None:
    # Craft a severity exactly at tau_cost via a single component.
    eng = RiskEngine(ActuarialConfig(weights={"roof": 1.0}, gamma=1.0, tau_cost=0.5))
    decision, _ = eng.decide(p_tamper=0.0, severity=0.5)
    assert decision is TriageDecision.MANUAL_REVIEW


def test_low_risk_low_cost_auto_approves(engine: RiskEngine) -> None:
    assessment = engine.assess([DamageComponent("wall", 0.05)], p_tamper=0.05)
    assert assessment.decision is TriageDecision.AUTO_APPROVE
    assert isinstance(assessment, ClaimAssessment)


def test_decide_rejects_invalid_inputs(engine: RiskEngine) -> None:
    with pytest.raises(ValueError):
        engine.decide(p_tamper=1.2, severity=0.1)
    with pytest.raises(ValueError):
        engine.decide(p_tamper=0.1, severity=-0.1)


def test_assessment_reasons_mention_the_trigger(engine: RiskEngine) -> None:
    fraud = engine.assess([], p_tamper=0.99)
    assert any("tau_fraud" in r for r in fraud.reasons)
    approved = engine.assess([], p_tamper=0.0)
    assert approved.decision is TriageDecision.AUTO_APPROVE
