"""NLI-based claim verifier and claim segmenter for UA-ACD."""
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

import spacy
from nltk.tokenize import sent_tokenize


nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.add_pipe("sentencizer")


class ClaimSegmenter:
    """
    Decomposes generated text into verifiable claims.

    granularity='sentence'  returns sentence-level claims (faster).
    granularity='atomic'    further splits on coordinating conjunctions (finer).
    """

    def segment(self, text: str, granularity: str = "sentence") -> List[str]:
        sentences = sent_tokenize(text.strip())
        sentences = [s.strip() for s in sentences if len(s.split()) > 3]

        if granularity == "sentence":
            return sentences

        atomic = []
        for sent in sentences:
            atomic.extend(self._split_atomic(sent))
        return [c for c in atomic if len(c.split()) > 3]

    def _split_atomic(self, sentence: str) -> List[str]:
        doc = nlp(sentence)
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break
        if root is None:
            return [sentence]

        conjuncts = [c for c in root.conjuncts if c.pos_ == "VERB"]
        if not conjuncts:
            return [sentence]

        claims, prev = [], 0
        for conj in conjuncts:
            subtree_start = min(t.i for t in conj.subtree)
            split_point = doc[subtree_start - 1].idx if subtree_start > 0 else 0
            chunk = sentence[prev:split_point].strip().rstrip(",")
            if chunk:
                claims.append(chunk)
            prev = split_point
        claims.append(sentence[prev:].strip())
        return [c for c in claims if c]


@dataclass
class VerificationResult:
    claim:              str
    uncertainty:        float
    tier:               str
    nli_threshold:      float
    max_entail_score:   float
    n_supporting:       int
    is_supported:       bool
    abstain:            bool
    verification_calls: int


class NLIVerifier:
    """
    Claim verifier backed by a cross-encoder NLI pipeline.

    verify_claim(claim, passages) returns the max entailment score across
    all passages along with supporting passage indices.
    """

    ENTAIL_LABEL = "entailment"

    def __init__(self, pipe, batch_size: int = 16):
        self.pipe       = pipe
        self.batch_size = batch_size
        self.call_count = 0

    def verify_claim(self, claim: str, passages: List[str]) -> Dict:
        if not passages:
            return {
                "max_entail_score": 0.0,
                "supporting_passages": [],
                "per_passage_scores": [],
            }

        pairs = [
            f"{p[:512]} [SEP] {claim[:256]}" for p in passages
        ]
        results = self.pipe(pairs, batch_size=self.batch_size, truncation=True)
        self.call_count += len(passages)

        scores = []
        for res in results:
            label_map = {r["label"].lower(): r["score"] for r in res}
            scores.append(label_map.get(self.ENTAIL_LABEL, 0.0))

        max_score = max(scores) if scores else 0.0
        supporting = [passages[i] for i, s in enumerate(scores) if s >= 0.5]
        return {
            "max_entail_score":    max_score,
            "supporting_passages": supporting,
            "per_passage_scores":  scores,
        }

    def reset_counter(self):
        self.call_count = 0


class AdaptiveConstraintController:
    """
    Maps per-claim uncertainty scores to NLI verification thresholds.

    Three tiers:
      low    (U < 0.30): NLI > 0.60  (lightweight)
      medium (0.30-0.70): NLI > 0.75 (standard)
      high   (U >= 0.70): NLI > 0.85 from >= 2 passages (strict)
    """

    def __init__(self, verifier: NLIVerifier, cfg):
        self.verifier = verifier
        self.cfg      = cfg

    def get_tier(self, U: float) -> str:
        if U < self.cfg.uncertainty_low:
            return "low"
        elif U < self.cfg.uncertainty_high:
            return "medium"
        return "high"

    def get_nli_threshold(self, tier: str) -> float:
        return {
            "low":    self.cfg.nli_threshold_low,
            "medium": self.cfg.nli_threshold_mid,
            "high":   self.cfg.nli_threshold_high,
        }[tier]

    def verify_claim_adaptive(
        self, claim: str, passages: List[str], U: float
    ) -> VerificationResult:
        tier       = self.get_tier(U)
        nli_thresh = self.get_nli_threshold(tier)

        res          = self.verifier.verify_claim(claim, passages)
        max_score    = res["max_entail_score"]
        n_supporting = sum(
            1 for s in res["per_passage_scores"] if s >= nli_thresh
        )

        if tier == "high":
            if n_supporting >= self.cfg.min_passages_strict and max_score >= nli_thresh:
                is_supported, abstain = True, False
            elif max_score >= nli_thresh:
                is_supported, abstain = True, False
            else:
                is_supported, abstain = False, True
        else:
            is_supported = max_score >= nli_thresh
            abstain      = False

        return VerificationResult(
            claim=claim,
            uncertainty=U,
            tier=tier,
            nli_threshold=nli_thresh,
            max_entail_score=max_score,
            n_supporting=n_supporting,
            is_supported=is_supported,
            abstain=abstain,
            verification_calls=len(passages),
        )

    def score_generation(
        self,
        claims: List[str],
        passages: List[str],
        uncertainties: List[float],
    ) -> Dict:
        results, total_calls, n_supported = [], 0, 0
        for claim, U in zip(claims, uncertainties):
            vr = self.verify_claim_adaptive(claim, passages, U)
            results.append(vr)
            total_calls += vr.verification_calls
            n_supported += int(vr.is_supported)

        factuality = n_supported / len(claims) if claims else 0.0
        return {
            "factuality_score":    factuality,
            "results":             results,
            "verification_calls":  total_calls,
        }
