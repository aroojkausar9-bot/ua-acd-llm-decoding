"""Evaluation metrics and result aggregation for UA-ACD."""
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.models.generator import GenerationOutput
from src.models.verifier import NLIVerifier


def run_evaluation(
    system,
    topics: List[str],
    entity_names: List[str],
    method_name: str,
    verifier: NLIVerifier,
    granularity: str = "sentence",
) -> pd.DataFrame:
    """
    Evaluate a generation system over a list of topics.

    Args:
        system:       any object with a .generate(topic, entity, granularity) method
        topics:       list of query strings
        entity_names: parallel list of entity names for retrieval filtering
        method_name:  label for this system in the output DataFrame
        verifier:     NLIVerifier instance (reset before evaluation)
        granularity:  claim segmentation granularity

    Returns:
        DataFrame with one row per topic containing all evaluation metrics
    """
    rows = []
    verifier.reset_counter()

    for topic, entity in tqdm(
        zip(topics, entity_names),
        total=len(topics),
        desc=f"Evaluating [{method_name}]",
    ):
        try:
            out: GenerationOutput = system.generate(
                topic, entity=entity.lower(), granularity=granularity
            )
            n_claims    = len(out.claims)
            n_supported = sum(1 for r in out.verification_results if r.is_supported)
            n_abstain   = sum(1 for r in out.verification_results if r.abstain)
            tiers       = [r.tier for r in out.verification_results]
            nli_scores  = [r.max_entail_score for r in out.verification_results]

            rows.append({
                "topic":              topic,
                "entity":             entity,
                "method":             method_name,
                "text":               out.text,
                "n_claims":           n_claims,
                "n_supported":        n_supported,
                "n_unsupported":      n_claims - n_supported,
                "n_abstain":          n_abstain,
                "factuality":         out.factuality_score,
                "hallucination_rate": (n_claims - n_supported) / n_claims
                                      if n_claims > 0 else 0.0,
                "mean_nli":           np.mean(nli_scores) if nli_scores else 0.0,
                "ver_calls":          out.verification_calls,
                "ver_calls_per_claim": out.verification_calls / n_claims
                                       if n_claims > 0 else 0.0,
                "gen_time_s":         out.generation_time_s,
                "n_low_tier":         tiers.count("low"),
                "n_mid_tier":         tiers.count("medium"),
                "n_high_tier":        tiers.count("high"),
                "output_words":       len(out.text.split()),
            })
        except Exception as exc:
            rows.append({
                "topic": topic, "entity": entity, "method": method_name,
                "text": "", "n_claims": 0, "n_supported": 0,
                "n_unsupported": 0, "n_abstain": 0,
                "factuality": 0.0, "hallucination_rate": 1.0,
                "mean_nli": 0.0, "ver_calls": 0, "ver_calls_per_claim": 0.0,
                "gen_time_s": 0.0, "n_low_tier": 0, "n_mid_tier": 0,
                "n_high_tier": 0, "output_words": 0,
            })
            print(f"Error on [{method_name}] {entity}: {exc}")

    return pd.DataFrame(rows)


def compute_summary_metrics(df_all: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-method metrics into a summary table."""
    records = []
    for method, grp in df_all.groupby("method"):
        fact   = grp["factuality"]
        records.append({
            "Method":                   method,
            "FactScore (mean)":         round(fact.mean(), 3),
            "FactScore (std)":          round(fact.std(), 3),
            "Hallucination Rate":       round(grp["hallucination_rate"].mean(), 3),
            "Mean NLI Score":           round(grp["mean_nli"].mean(), 3),
            "Avg Ver Calls":            round(grp["ver_calls"].mean(), 1),
            "Ver Calls/Claim":          round(grp["ver_calls_per_claim"].mean(), 2),
            "Avg Gen Time (s)":         round(grp["gen_time_s"].mean(), 1),
            "Avg Output Words":         round(grp["output_words"].mean(), 0),
        })
    return pd.DataFrame(records).sort_values("FactScore (mean)", ascending=False)
