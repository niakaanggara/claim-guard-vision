"""Actuarial risk & triage engine.

Severity index
--------------
For each damaged component ``i`` with area ratio ``r_i in [0, 1]`` and
repair-cost weight ``w_i``::

    S_claim = sum_i w_i * r_i ** gamma

``gamma > 1`` penalises large concentrated damage more than the same
area spread across many small components. With ``normalise_weights=True``
the weights are rescaled to sum to 1 first, so ``S_claim`` stays in
``[0, 1]`` when every ``r_i`` is in ``[0, 1]``.

Triage rules (evaluated in order — fraud always wins)
-----------------------------------------------------
1. ``P_tamper >= tau_fraud``            -> ``REJECT / FORENSIC AUDIT``
2. ``S_claim >= tau_cost``              -> ``MANUAL ADJUSTER REVIEW``
3. otherwise                            -> ``AUTO-APPROVE (FAST TRACK)``

Thresholds and weights live in ``configs/actuarial_weights.yaml`` and are
loaded with :func:`load_actuarial_config`; the engine itself is pure
Python and fully unit-testable without any model or image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import yaml


class TriageDecision(str, Enum):
    """Possible outcomes of the triage decision."""

    REJECT_FORENSIC_AUDIT = "REJECT / FORENSIC AUDIT"
    MANUAL_REVIEW = "MANUAL ADJUSTER REVIEW"
    AUTO_APPROVE = "AUTO-APPROVE (FAST TRACK)"


@dataclass(frozen=True)
class DamageComponent:
    """One damaged component, as produced by the segmentation stage."""

    name: str
    area_ratio: float  # r_i in [0, 1]

    def __post_init__(self) -> None:
        if not 0.0 <= self.area_ratio <= 1.0:
            raise ValueError(f"area_ratio must be in [0, 1], got {self.area_ratio}")


@dataclass(frozen=True)
class ClaimAssessment:
    """Result of assessing one claim."""

    severity: float  # S_claim
    p_tamper: float
    decision: TriageDecision
    contributions: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "p_tamper": self.p_tamper,
            "decision": self.decision.value,
            "contributions": dict(self.contributions),
            "reasons": list(self.reasons),
        }


@dataclass
class ActuarialConfig:
    """Validated actuarial configuration (usually loaded from YAML)."""

    weights: dict[str, float]
    gamma: float = 1.5
    tau_fraud: float = 0.65
    tau_cost: float = 0.45
    normalise_weights: bool = True
    caps: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if not 0.0 <= self.tau_fraud <= 1.0:
            raise ValueError(f"tau_fraud must be in [0, 1], got {self.tau_fraud}")
        if not 0.0 <= self.tau_cost <= 1.0:
            raise ValueError(f"tau_cost must be in [0, 1], got {self.tau_cost}")
        if not self.weights:
            raise ValueError("weights must not be empty")
        for name, w in self.weights.items():
            if w < 0:
                raise ValueError(f"weight for {name!r} must be >= 0, got {w}")


def load_actuarial_config(path: Union[str, Path] = "configs/actuarial_weights.yaml") -> ActuarialConfig:
    """Load and validate an actuarial config from a YAML file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ActuarialConfig(
        weights={str(k): float(v) for k, v in (raw.get("weights") or {}).items()},
        gamma=float(raw.get("gamma", 1.5)),
        tau_fraud=float(raw.get("tau_fraud", 0.65)),
        tau_cost=float(raw.get("tau_cost", 0.45)),
        normalise_weights=bool(raw.get("normalise_weights", True)),
        caps={str(k): float(v) for k, v in (raw.get("caps") or {}).items()},
    )


class RiskEngine:
    """Severity computation and rule-based triage for insurance claims."""

    def __init__(self, config: Optional[ActuarialConfig] = None) -> None:
        self.config = config or ActuarialConfig(weights={"other": 1.0})
        self._weights = self._effective_weights()

    def _effective_weights(self) -> dict[str, float]:
        weights = dict(self.config.weights)
        if self.config.normalise_weights:
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("weights must sum to a positive value when normalising")
            weights = {k: v / total for k, v in weights.items()}
        return weights

    @property
    def weights(self) -> dict[str, float]:
        """Return the (possibly normalised) weights actually used."""
        return dict(self._weights)

    def weight_for(self, component: str) -> float:
        """Return the weight for ``component``, falling back to ``other`` / 0."""
        if component in self._weights:
            return self._weights[component]
        return self._weights.get("other", 0.0)

    def compute_severity(
        self,
        components: Sequence[DamageComponent],
        weights: Optional[Mapping[str, float]] = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute ``S_claim`` and the per-component contributions.

        Parameters
        ----------
        components:
            Damaged components with ``area_ratio`` in ``[0, 1]``.
        weights:
            Optional override weights (used as-is, not normalised).

        Returns
        -------
        (severity, contributions)
            ``severity`` is ``sum_i w_i * r_i ** gamma`` (capped per
            component when a cap is configured), and ``contributions``
            maps each component name to its capped contribution.
        """
        effective = dict(weights) if weights is not None else self._weights
        severity = 0.0
        contributions: dict[str, float] = {}
        for comp in components:
            if comp.name not in effective and "other" not in effective:
                continue
            w = effective.get(comp.name, effective.get("other", 0.0))
            contribution = float(w) * (float(comp.area_ratio) ** self.config.gamma)
            cap = self.config.caps.get(comp.name)
            if cap is not None:
                contribution = min(contribution, float(cap))
            contributions[comp.name] = contributions.get(comp.name, 0.0) + contribution
            severity += contribution
        return float(severity), contributions

    def decide(self, p_tamper: float, severity: float) -> tuple[TriageDecision, list[str]]:
        """Apply the triage rules in order; fraud always wins."""
        if not 0.0 <= p_tamper <= 1.0:
            raise ValueError(f"p_tamper must be in [0, 1], got {p_tamper}")
        if severity < 0:
            raise ValueError(f"severity must be >= 0, got {severity}")
        if p_tamper >= self.config.tau_fraud:
            return TriageDecision.REJECT_FORENSIC_AUDIT, [
                f"P_tamper={p_tamper:.3f} >= tau_fraud={self.config.tau_fraud:.3f}: "
                "image is likely manipulated; route to forensic audit."
            ]
        if severity >= self.config.tau_cost:
            return TriageDecision.MANUAL_REVIEW, [
                f"S_claim={severity:.3f} >= tau_cost={self.config.tau_cost:.3f}: "
                "estimated cost is above the fast-track limit; a human adjuster must review."
            ]
        return TriageDecision.AUTO_APPROVE, [
            f"P_tamper={p_tamper:.3f} < tau_fraud and S_claim={severity:.3f} < tau_cost: "
            "low fraud risk and low estimated cost; fast-track approval."
        ]

    def assess(
        self,
        components: Sequence[DamageComponent],
        p_tamper: float,
        weights: Optional[Mapping[str, float]] = None,
    ) -> ClaimAssessment:
        """Compute severity and triage in one call for a single claim."""
        severity, contributions = self.compute_severity(components, weights=weights)
        decision, reasons = self.decide(p_tamper, severity)
        return ClaimAssessment(
            severity=severity,
            p_tamper=float(p_tamper),
            decision=decision,
            contributions=contributions,
            reasons=reasons,
        )


def severity_index(
    ratios: Sequence[float],
    weights: Sequence[float],
    gamma: float = 1.5,
) -> float:
    """Standalone ``S_claim = sum w_i * r_i ** gamma`` for quick calculations.

    Raises ``ValueError`` on length mismatch, negative weights, ratios
    outside ``[0, 1]``, or non-positive ``gamma``.
    """
    if len(ratios) != len(weights):
        raise ValueError(f"ratios and weights must have the same length, got {len(ratios)} vs {len(weights)}")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    total = 0.0
    for r, w in zip(ratios, weights):
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"ratios must be in [0, 1], got {r}")
        if w < 0:
            raise ValueError(f"weights must be >= 0, got {w}")
        total += float(w) * (float(r) ** gamma)
    return float(total)
