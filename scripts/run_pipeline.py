"""
run_pipeline.py
---------------
Main entry point for UA-ACD evaluation. Reproduces all results from
Assignment 4 in a single run.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml

For CPU-only runs set load_in_4bit: false in the config.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification, pipeline,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer

from src.utils.config import load_config
from src.data.dataset import load_factscore_topics, load_wikipedia_passages
from src.models.retriever import build_indices, HybridRetriever
from src.models.verifier import (
    ClaimSegmenter, NLIVerifier, AdaptiveConstraintController,
)
from src.models.generator import (
    UncertaintyQuantifier,
    VanillaLLM, StandardRAG, StaticConstraintRAG,
    UAACDGenerator, UAACDDecModGenerator,
)
from src.evaluation.metrics import run_evaluation, compute_summary_metrics


def parse_args():
    p = argparse.ArgumentParser(description="UA-ACD pipeline runner")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Path to YAML config file")
    p.add_argument("--methods", nargs="+",
                   default=["all"],
                   choices=["all", "vanilla", "rag", "static", "uaacd", "decmod"],
                   help="Which methods to evaluate")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory for CSVs and figures")
    return p.parse_args()


def setup_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models(cfg):
    """Load generator model, NLI pipeline, and embedding model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # Generator model
    if cfg.load_in_4bit and torch.cuda.is_available():
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        gen_model = AutoModelForCausalLM.from_pretrained(
            cfg.generator_model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        gen_model = AutoModelForCausalLM.from_pretrained(
            cfg.generator_model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)

    gen_tokenizer = AutoTokenizer.from_pretrained(
        cfg.generator_model_id, trust_remote_code=True
    )
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    # NLI pipeline (CPU to save VRAM)
    nli_pipe = pipeline(
        "text-classification",
        model=cfg.nli_model_id,
        device=-1,
        top_k=None,
    )

    # Embedding model
    embed_model = SentenceTransformer(cfg.embed_model_id, device="cpu")

    return gen_model, gen_tokenizer, nli_pipe, embed_model, device


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    setup_reproducibility(cfg.seed)

    out_dir = args.output_dir or cfg.results_dir
    fig_dir = cfg.figures_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    run_all    = "all" in args.methods
    run_vanilla = run_all or "vanilla"  in args.methods
    run_rag     = run_all or "rag"     in args.methods
    run_static  = run_all or "static"  in args.methods
    run_uaacd   = run_all or "uaacd"   in args.methods
    run_decmod  = run_all or "decmod"  in args.methods

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    topics, entity_names = load_factscore_topics(cfg.n_eval_samples, cfg.seed)
    print(f"Loaded {len(topics)} topics")

    print("Building Wikipedia passage corpus...")
    all_passages, passage_to_entity = load_wikipedia_passages(
        entity_names, cfg.max_passages_per_entity
    )
    print(f"Corpus: {len(all_passages)} passages")

    # ── Models ───────────────────────────────────────────────────────────────
    print("Loading models...")
    gen_model, gen_tokenizer, nli_pipe, embed_model, device = load_models(cfg)

    print("Building retrieval indices...")
    faiss_idx, bm25_idx = build_indices(all_passages, embed_model)
    retriever = HybridRetriever(
        embed_model, faiss_idx, bm25_idx, all_passages, passage_to_entity,
        bm25_weight=cfg.bm25_weight, top_k=cfg.top_k_retrieve,
    )

    verifier  = NLIVerifier(nli_pipe)
    segmenter = ClaimSegmenter()
    acc       = AdaptiveConstraintController(verifier, cfg)
    uq        = UncertaintyQuantifier(
        gen_model, gen_tokenizer, embed_model,
        entropy_weight=0.5, consistency_weight=0.5, device=device,
    )

    # ── Systems ──────────────────────────────────────────────────────────────
    vanilla = VanillaLLM(gen_model, gen_tokenizer, cfg, segmenter, verifier, retriever)
    rag     = StandardRAG(gen_model, gen_tokenizer, cfg, segmenter, verifier, retriever)
    static  = StaticConstraintRAG(gen_model, gen_tokenizer, cfg, segmenter, verifier, retriever)
    uaacd   = UAACDGenerator(gen_model, gen_tokenizer, retriever, uq, acc, cfg, segmenter)
    decmod  = UAACDDecModGenerator(gen_model, gen_tokenizer, retriever, uq, acc, cfg, segmenter)

    # ── Evaluation ───────────────────────────────────────────────────────────
    all_results = {}

    if run_vanilla:
        print("\nEvaluating Vanilla LLM...")
        all_results["Vanilla LLM"] = run_evaluation(
            vanilla, topics, entity_names, "Vanilla LLM", verifier
        )

    if run_rag:
        print("\nEvaluating Standard RAG...")
        all_results["Standard RAG"] = run_evaluation(
            rag, topics, entity_names, "Standard RAG", verifier
        )

    if run_static:
        print("\nEvaluating Static Constraint...")
        all_results["Static Constraint"] = run_evaluation(
            static, topics, entity_names, "Static Constraint", verifier
        )

    if run_uaacd:
        print("\nEvaluating UA-ACD...")
        all_results["UA-ACD"] = run_evaluation(
            uaacd, topics, entity_names, "UA-ACD", verifier
        )

    if run_decmod:
        print("\nEvaluating UA-ACD-Dec-Mod...")
        all_results["UA-ACD-Dec-Mod"] = run_evaluation(
            decmod, topics, entity_names, "UA-ACD-Dec-Mod", verifier
        )

    # ── Save results ─────────────────────────────────────────────────────────
    df_all = pd.concat(all_results.values(), ignore_index=True)
    df_all.to_csv(os.path.join(out_dir, "ua_acd_all_results.csv"), index=False)

    summary = compute_summary_metrics(df_all)
    summary.to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    main()
