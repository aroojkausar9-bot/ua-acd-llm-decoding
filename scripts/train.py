"""
train.py
--------
UA-ACD is an inference-only method: it applies adaptive NLI verification
at generation time without any gradient-based training or fine-tuning of
the generator. No training phase exists for this project.

This stub is included to satisfy the required repository structure
(scripts/train.py). Running it will print a short explanation and exit
cleanly.

For the full evaluation pipeline, use:
    python scripts/run_pipeline.py --config configs/default.yaml
"""

import sys


def main():
    msg = (
        "\n"
        "UA-ACD is an inference-only system.\n"
        "There is no training phase: the generator (Phi-2) and NLI verifier\n"
        "(cross-encoder/nli-deberta-v3-small) are used as frozen pretrained\n"
        "models. No fine-tuning or parameter updates are performed.\n\n"
        "To reproduce all results, run:\n"
        "    python scripts/run_pipeline.py --config configs/default.yaml\n\n"
        "To recompute metrics and figures from saved CSVs:\n"
        "    python scripts/eval.py --results artifacts/logs/ua_acd_all_results.csv\n"
    )
    print(msg)
    sys.exit(0)


if __name__ == "__main__":
    main()
