"""Minimal tests for NLI verifier and claim segmenter."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch
import pytest

from src.models.verifier import ClaimSegmenter, NLIVerifier, AdaptiveConstraintController


# ── ClaimSegmenter ───────────────────────────────────────────────────────────

def test_segmenter_sentence():
    seg = ClaimSegmenter()
    text = "Marie Curie was born in Warsaw in 1867. She won the Nobel Prize twice."
    claims = seg.segment(text, "sentence")
    assert len(claims) == 2
    assert all(isinstance(c, str) for c in claims)


def test_segmenter_short_text():
    seg = ClaimSegmenter()
    claims = seg.segment("OK.", "sentence")
    assert claims == []


def test_segmenter_atomic():
    seg = ClaimSegmenter()
    text = "She discovered polonium and won the Nobel Prize."
    claims = seg.segment(text, "atomic")
    assert isinstance(claims, list)


# ── NLIVerifier ──────────────────────────────────────────────────────────────

def _make_mock_pipe(label="entailment", score=0.9):
    def pipe(pairs, **kwargs):
        return [[{"label": label, "score": score},
                 {"label": "contradiction", "score": 1 - score}]
                for _ in pairs]
    return pipe


def test_verifier_returns_max_score():
    v = NLIVerifier(_make_mock_pipe("entailment", 0.85))
    result = v.verify_claim("She won a Nobel Prize.", ["Nobel Prize facts."])
    assert abs(result["max_entail_score"] - 0.85) < 1e-5


def test_verifier_empty_passages():
    v = NLIVerifier(_make_mock_pipe())
    result = v.verify_claim("Any claim.", [])
    assert result["max_entail_score"] == 0.0
    assert result["supporting_passages"] == []


def test_verifier_call_count():
    v = NLIVerifier(_make_mock_pipe())
    v.verify_claim("claim", ["p1", "p2", "p3"])
    assert v.call_count == 3
    v.reset_counter()
    assert v.call_count == 0


# ── AdaptiveConstraintController ─────────────────────────────────────────────

def _mock_cfg():
    cfg = MagicMock()
    cfg.uncertainty_low  = 0.30
    cfg.uncertainty_high = 0.70
    cfg.nli_threshold_low  = 0.60
    cfg.nli_threshold_mid  = 0.75
    cfg.nli_threshold_high = 0.85
    cfg.min_passages_strict = 2
    return cfg


def test_tier_assignment():
    v   = NLIVerifier(_make_mock_pipe("entailment", 0.9))
    acc = AdaptiveConstraintController(v, _mock_cfg())
    assert acc.get_tier(0.10) == "low"
    assert acc.get_tier(0.50) == "medium"
    assert acc.get_tier(0.80) == "high"


def test_adaptive_verify_low_tier():
    v   = NLIVerifier(_make_mock_pipe("entailment", 0.65))
    acc = AdaptiveConstraintController(v, _mock_cfg())
    result = acc.verify_claim_adaptive("claim", ["passage"], U=0.10)
    assert result.tier == "low"
    assert result.is_supported is True
    assert result.abstain is False


def test_adaptive_verify_high_tier_abstain():
    v   = NLIVerifier(_make_mock_pipe("contradiction", 0.95))
    acc = AdaptiveConstraintController(v, _mock_cfg())
    result = acc.verify_claim_adaptive("claim", ["passage"], U=0.80)
    assert result.tier == "high"
    assert result.abstain is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
