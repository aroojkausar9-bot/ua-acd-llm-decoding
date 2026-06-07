"""Configuration loader for UA-ACD."""
import yaml
import argparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UAACDConfig:
    # Model IDs
    generator_model_id: str = "microsoft/phi-2"
    nli_model_id:       str = "cross-encoder/nli-deberta-v3-small"
    embed_model_id:     str = "sentence-transformers/all-MiniLM-L6-v2"
    load_in_4bit:       bool = True

    # Generation
    max_new_tokens:       int   = 256
    num_return_sequences: int   = 4
    temperature:          float = 0.8
    top_p:                float = 0.92
    do_sample:            bool  = True
    min_new_tokens:       int   = 80

    # Retrieval
    top_k_retrieve: int   = 5
    bm25_weight:    float = 0.5

    # Uncertainty thresholds
    uncertainty_low:  float = 0.30
    uncertainty_high: float = 0.70

    # NLI thresholds per tier
    nli_threshold_low:   float = 0.60
    nli_threshold_mid:   float = 0.75
    nli_threshold_high:  float = 0.85
    min_passages_strict: int   = 2

    # Reranking weights
    alpha_factuality: float = 0.6
    alpha_fluency:    float = 0.4

    # Decoding modulation (UA-ACD-Dec-Mod)
    dec_temp_focused:      float = 0.55
    dec_temp_balanced:     float = 0.75
    dec_temp_explore:      float = 0.90
    dec_temp_low_thresh:   float = 1.5
    dec_temp_high_thresh:  float = 3.5
    dec_top_k_focused:     int   = 20
    dec_top_k_balanced:    int   = 50
    dec_top_k_explore:     int   = 100

    # Evaluation
    n_eval_samples:          int  = 50
    max_passages_per_entity: int  = 20
    track_verification_calls: bool = True

    # Static baseline
    static_nli_threshold: float = 0.75

    # Paths
    results_dir: str = "artifacts/logs"
    figures_dir: str = "figures"
    seed:        int = 42


def load_config(config_path: Optional[str] = None) -> UAACDConfig:
    """Load config from YAML, falling back to defaults."""
    cfg = UAACDConfig()
    if config_path is None:
        return cfg
    with open(config_path) as f:
        data = yaml.safe_load(f)

    if "model" in data:
        m = data["model"]
        if "generator_model_id" in m: cfg.generator_model_id = m["generator_model_id"]
        if "nli_model_id" in m:       cfg.nli_model_id       = m["nli_model_id"]
        if "embed_model_id" in m:     cfg.embed_model_id     = m["embed_model_id"]
        if "load_in_4bit" in m:       cfg.load_in_4bit       = m["load_in_4bit"]

    if "generation" in data:
        g = data["generation"]
        if "max_new_tokens" in g:       cfg.max_new_tokens       = g["max_new_tokens"]
        if "num_return_sequences" in g: cfg.num_return_sequences = g["num_return_sequences"]
        if "temperature" in g:          cfg.temperature          = g["temperature"]
        if "top_p" in g:                cfg.top_p                = g["top_p"]
        if "do_sample" in g:            cfg.do_sample            = g["do_sample"]
        if "min_new_tokens" in g:       cfg.min_new_tokens       = g["min_new_tokens"]

    if "retrieval" in data:
        r = data["retrieval"]
        if "top_k" in r:         cfg.top_k_retrieve = r["top_k"]
        if "bm25_weight" in r:   cfg.bm25_weight    = r["bm25_weight"]

    if "uncertainty" in data:
        u = data["uncertainty"]
        if "low_threshold" in u:  cfg.uncertainty_low  = u["low_threshold"]
        if "high_threshold" in u: cfg.uncertainty_high = u["high_threshold"]

    if "nli_thresholds" in data:
        n = data["nli_thresholds"]
        if "low" in n:                  cfg.nli_threshold_low   = n["low"]
        if "mid" in n:                  cfg.nli_threshold_mid   = n["mid"]
        if "high" in n:                 cfg.nli_threshold_high  = n["high"]
        if "min_passages_strict" in n:  cfg.min_passages_strict = n["min_passages_strict"]

    if "reranking" in data:
        rk = data["reranking"]
        if "alpha_factuality" in rk: cfg.alpha_factuality = rk["alpha_factuality"]
        if "alpha_fluency" in rk:    cfg.alpha_fluency    = rk["alpha_fluency"]

    if "evaluation" in data:
        e = data["evaluation"]
        if "n_eval_samples" in e: cfg.n_eval_samples = e["n_eval_samples"]
        if "seed" in e:           cfg.seed           = e["seed"]

    if "static_baseline" in data:
        cfg.static_nli_threshold = data["static_baseline"].get("nli_threshold", 0.75)

    if "output" in data:
        if "results_dir" in data["output"]: cfg.results_dir = data["output"]["results_dir"]
        if "figures_dir" in data["output"]: cfg.figures_dir = data["output"]["figures_dir"]

    return cfg
