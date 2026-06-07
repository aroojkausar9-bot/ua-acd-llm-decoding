"""Generator pipelines: VanillaLLM, StandardRAG, StaticConstraint, UAACDGenerator, UAACDDecModGenerator."""
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.models.verifier import (
    ClaimSegmenter, NLIVerifier, AdaptiveConstraintController,
    VerificationResult,
)


@dataclass
class GenerationOutput:
    text:                 str
    claims:               List[str]
    uncertainties:        List[float]
    verification_results: List[VerificationResult]
    factuality_score:     float
    verification_calls:   int
    generation_time_s:    float
    method:               str


class UncertaintyQuantifier:
    """
    Estimates epistemic uncertainty for generated text using two signals:
      1. Token-level entropy averaged over generated tokens.
      2. Semantic consistency: 1 - mean pairwise cosine similarity across
         N sampled continuations of the same prompt.
    The final uncertainty score is a weighted combination of both.
    """

    def __init__(
        self,
        gen_model,
        gen_tokenizer,
        embed_model,
        n_samples:          int   = 4,
        entropy_weight:     float = 0.5,
        consistency_weight: float = 0.5,
        device:             str   = "cpu",
    ):
        self.model         = gen_model
        self.tokenizer     = gen_tokenizer
        self.embed_model   = embed_model
        self.n_samples     = n_samples
        self.entropy_w     = entropy_weight
        self.consistency_w = consistency_weight
        self.device        = device

    @torch.no_grad()
    def token_entropy(
        self, input_ids: torch.Tensor, generated_ids: torch.Tensor
    ) -> float:
        full_ids   = torch.cat([input_ids, generated_ids], dim=-1)
        logits     = self.model(full_ids).logits
        gen_start  = input_ids.shape[-1]
        gen_logits = logits[0, gen_start - 1: gen_start - 1 + generated_ids.shape[-1], :]
        probs      = F.softmax(gen_logits, dim=-1)
        entropy    = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return entropy.mean().item()

    def semantic_consistency(
        self, prompt: str, n_samples: Optional[int] = None
    ) -> Tuple[float, List[str]]:
        n      = n_samples or self.n_samples
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        start = inputs.input_ids.shape[-1]
        texts = [
            self.tokenizer.decode(o[start:], skip_special_tokens=True).strip()
            for o in outs
        ]
        texts = [t for t in texts if t]
        if len(texts) < 2:
            return 0.5, texts

        embs = self.embed_model.encode(texts, normalize_embeddings=True)
        sims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sims.append(float(np.dot(embs[i], embs[j])))
        dispersion = 1.0 - (np.mean(sims) if sims else 0.5)
        return float(dispersion), texts

    def estimate(self, prompt: str, generated_text: str) -> float:
        """Compute combined uncertainty score for a (prompt, generated_text) pair."""
        inputs  = self.tokenizer(prompt, return_tensors="pt",
                                 truncation=True, max_length=512).to(self.device)
        gen_ids = self.tokenizer(generated_text, return_tensors="pt",
                                 truncation=True, max_length=256).input_ids.to(self.device)

        ent = self.token_entropy(inputs.input_ids, gen_ids)
        # Normalise entropy to [0,1] (max entropy for vocab ~13 nats)
        ent_norm = min(ent / 13.0, 1.0)

        disp, _ = self.semantic_consistency(prompt, n_samples=2)

        return float(self.entropy_w * ent_norm + self.consistency_w * disp)


class VanillaLLM:
    """Baseline 1: greedy decoding, no retrieval. Post-hoc NLI evaluation only."""

    def __init__(self, model, tokenizer, cfg, segmenter, verifier, retriever):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.segmenter = segmenter
        self.verifier  = verifier
        self.retriever = retriever

    @torch.no_grad()
    def generate(
        self, query: str, entity: Optional[str] = None, granularity: str = "sentence"
    ) -> GenerationOutput:
        t0     = time.time()
        prompt = f"Write a factual biography about: {query}\n\nBiography:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(next(self.model.parameters()).device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()

        passages = [p for p, _ in self.retriever.retrieve(query, entity=entity)]
        claims   = self.segmenter.segment(text, granularity)
        ver_results, total_calls, n_supported = [], 0, 0
        for c in claims:
            r = self.verifier.verify_claim(c, passages)
            total_calls += len(passages)
            supported    = r["max_entail_score"] >= self.cfg.static_nli_threshold
            n_supported += int(supported)
            ver_results.append(VerificationResult(
                claim=c, uncertainty=0.5, tier="medium",
                nli_threshold=self.cfg.static_nli_threshold,
                max_entail_score=r["max_entail_score"],
                n_supporting=sum(1 for s in r["per_passage_scores"] if s >= 0.5),
                is_supported=supported, abstain=False,
                verification_calls=len(passages),
            ))

        return GenerationOutput(
            text=text, claims=claims,
            uncertainties=[0.5] * len(claims),
            verification_results=ver_results,
            factuality_score=n_supported / len(claims) if claims else 0.0,
            verification_calls=total_calls,
            generation_time_s=time.time() - t0,
            method="Vanilla LLM",
        )


class StandardRAG:
    """Baseline 2: retrieval-augmented generation with static prompt grounding."""

    def __init__(self, model, tokenizer, cfg, segmenter, verifier, retriever):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.segmenter = segmenter
        self.verifier  = verifier
        self.retriever = retriever

    @torch.no_grad()
    def generate(
        self, query: str, entity: Optional[str] = None, granularity: str = "sentence"
    ) -> GenerationOutput:
        t0       = time.time()
        retrieved = self.retriever.retrieve(query, entity=entity)
        passages  = [p for p, _ in retrieved] if retrieved else ["No evidence found."]
        evidence  = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages[:3]))
        prompt    = (
            f"Using the following evidence, write a factual biography.\n\n"
            f"Evidence:\n{evidence}\n\nQuestion: {query}\n\nBiography:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=768
        ).to(next(self.model.parameters()).device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()

        claims = self.segmenter.segment(text, granularity)
        ver_results, total_calls, n_supported = [], 0, 0
        for c in claims:
            r = self.verifier.verify_claim(c, passages)
            total_calls += len(passages)
            supported    = r["max_entail_score"] >= self.cfg.static_nli_threshold
            n_supported += int(supported)
            ver_results.append(VerificationResult(
                claim=c, uncertainty=0.5, tier="medium",
                nli_threshold=self.cfg.static_nli_threshold,
                max_entail_score=r["max_entail_score"],
                n_supporting=sum(1 for s in r["per_passage_scores"] if s >= 0.5),
                is_supported=supported, abstain=False,
                verification_calls=len(passages),
            ))

        return GenerationOutput(
            text=text, claims=claims,
            uncertainties=[0.5] * len(claims),
            verification_results=ver_results,
            factuality_score=n_supported / len(claims) if claims else 0.0,
            verification_calls=total_calls,
            generation_time_s=time.time() - t0,
            method="Standard RAG",
        )


class StaticConstraintRAG:
    """
    Baseline 3: RAG with a fixed NLI threshold applied uniformly to all claims
    during multi-candidate beam reranking. Threshold = cfg.static_nli_threshold.
    """

    def __init__(self, model, tokenizer, cfg, segmenter, verifier, retriever):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.segmenter = segmenter
        self.verifier  = verifier
        self.retriever = retriever

    def _build_prompt(self, query: str, passages: List[str]) -> str:
        evidence = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages[:3]))
        return (
            f"You are a factual biography writer. Using the evidence below, "
            f"write a concise and accurate biography.\n\n"
            f"Evidence:\n{evidence}\n\nQuestion: {query}\n\nBiography:"
        )

    @torch.no_grad()
    def generate(
        self, query: str, entity: Optional[str] = None, granularity: str = "sentence"
    ) -> GenerationOutput:
        t0        = time.time()
        retrieved = self.retriever.retrieve(query, entity=entity)
        passages  = [p for p, _ in retrieved] if retrieved else ["No evidence found."]
        prompt    = self._build_prompt(query, passages)
        inputs    = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=768
        ).to(next(self.model.parameters()).device)

        outs = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            num_return_sequences=self.cfg.num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        start      = inputs.input_ids.shape[-1]
        candidates = [
            self.tokenizer.decode(o[start:], skip_special_tokens=True).strip()
            for o in outs
        ]

        best_text, best_fact, best_data = None, -1.0, None
        total_calls = 0
        for cand in candidates:
            claims = self.segmenter.segment(cand, granularity)
            if not claims:
                continue
            n_sup, ver_results = 0, []
            for c in claims:
                r = self.verifier.verify_claim(c, passages)
                total_calls += len(passages)
                sup = r["max_entail_score"] >= self.cfg.static_nli_threshold
                n_sup += int(sup)
                ver_results.append(VerificationResult(
                    claim=c, uncertainty=0.5, tier="medium",
                    nli_threshold=self.cfg.static_nli_threshold,
                    max_entail_score=r["max_entail_score"],
                    n_supporting=sum(1 for s in r["per_passage_scores"] if s >= 0.5),
                    is_supported=sup, abstain=False,
                    verification_calls=len(passages),
                ))
            fact = n_sup / len(claims)
            if fact > best_fact:
                best_fact, best_text, best_data = fact, cand, ver_results

        if best_text is None:
            best_text, best_data = candidates[0] if candidates else "", []
            best_fact = 0.0

        return GenerationOutput(
            text=best_text, claims=self.segmenter.segment(best_text, granularity),
            uncertainties=[0.5] * len(best_data),
            verification_results=best_data,
            factuality_score=best_fact,
            verification_calls=total_calls,
            generation_time_s=time.time() - t0,
            method="Static Constraint",
        )


class UAACDGenerator:
    """
    UA-ACD main pipeline.

    Per query:
      1. Retrieve evidence passages via hybrid retrieval.
      2. Generate N candidate biographies.
      3. For each candidate, estimate per-claim uncertainty.
      4. Apply adaptive NLI verification (threshold conditioned on uncertainty tier).
      5. Rerank by alpha * factuality + (1-alpha) * fluency.
      6. Return the best candidate.
    """

    def __init__(self, gen_model, gen_tokenizer, retriever, uq, acc, cfg, segmenter):
        self.model     = gen_model
        self.tokenizer = gen_tokenizer
        self.retriever = retriever
        self.uq        = uq
        self.acc       = acc
        self.cfg       = cfg
        self.segmenter = segmenter

    def build_prompt(self, query: str, passages: List[str]) -> str:
        def _trunc(p, n=120):
            words = p.split()
            return " ".join(words[:n]) + ("..." if len(words) > n else "")
        evidence = "\n".join(f"[{i+1}] {_trunc(p)}" for i, p in enumerate(passages[:5]))
        return (
            f"Read the following evidence carefully, then write a biography.\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Question: {query}\n\n"
            f"Instructions: Write a detailed biography of at least 6 sentences. "
            f"Every sentence must be directly supported by the evidence above. "
            f"Do not add facts not found in the evidence.\n\nBiography:"
        )

    @torch.no_grad()
    def generate_candidates(self, prompt: str, n: int = 4) -> List[str]:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=900
        ).to(next(self.model.parameters()).device)

        outs = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            min_new_tokens=getattr(self.cfg, "min_new_tokens", 80),
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            num_return_sequences=n,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        start = inputs.input_ids.shape[-1]
        return [
            self.tokenizer.decode(o[start:], skip_special_tokens=True).strip()
            for o in outs
        ]

    def get_claim_uncertainties(self, prompt: str, claims: List[str]) -> List[float]:
        return [self.uq.estimate(prompt, c) for c in claims]

    def compute_fluency_score(self, text: str) -> float:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(next(self.model.parameters()).device)
        with torch.no_grad():
            loss = self.model(**inputs, labels=inputs.input_ids).loss
        ppl = torch.exp(loss).item()
        return float(1.0 / (1.0 + ppl / 100.0))

    def generate(
        self, query: str, entity: Optional[str] = None, granularity: str = "sentence"
    ) -> GenerationOutput:
        t0        = time.time()
        retrieved = self.retriever.retrieve(query, entity=entity)
        passages  = [p for p, _ in retrieved] if retrieved else ["No evidence found."]
        prompt    = self.build_prompt(query, passages)
        candidates = self.generate_candidates(prompt, n=self.cfg.num_return_sequences)

        best_text, best_score, best_data = None, -1.0, None
        for cand_text in candidates:
            claims = self.segmenter.segment(cand_text, granularity)
            if not claims:
                continue
            uncertainties = self.get_claim_uncertainties(prompt, claims)
            score_info    = self.acc.score_generation(claims, passages, uncertainties)
            fluency       = self.compute_fluency_score(cand_text)

            # Soft reranking on continuous NLI scores for better candidate selection
            soft_scores = []
            for claim in claims:
                res = self.acc.verifier.verify_claim(claim, passages[:5])
                soft_scores.append(res["max_entail_score"])
            soft_fact = float(np.mean(soft_scores)) if soft_scores else 0.0

            combined = self.cfg.alpha_factuality * soft_fact + self.cfg.alpha_fluency * fluency
            if combined > best_score:
                best_score = combined
                best_text  = cand_text
                best_data  = {
                    "claims":        claims,
                    "uncertainties": uncertainties,
                    "ver_results":   score_info["results"],
                    "factuality":    score_info["factuality_score"],
                    "ver_calls":     score_info["verification_calls"],
                }

        if best_text is None:
            best_text = candidates[0] if candidates else ""
            best_data = {
                "claims": [], "uncertainties": [], "ver_results": [],
                "factuality": 0.0, "ver_calls": 0,
            }

        return GenerationOutput(
            text=best_text,
            claims=best_data["claims"],
            uncertainties=best_data["uncertainties"],
            verification_results=best_data["ver_results"],
            factuality_score=best_data["factuality"],
            verification_calls=best_data["ver_calls"],
            generation_time_s=time.time() - t0,
            method="UA-ACD",
        )


class UAACDDecModGenerator(UAACDGenerator):
    """
    UA-ACD with sentence-level entropy-conditioned decoding.

    After generating each sentence, the model's token-level entropy is used
    to set the temperature and top-k for the next sentence. High entropy
    tightens decoding to reduce hallucination drift; low entropy relaxes
    it slightly to allow fluent elaboration.
    """

    def _compute_sentence_entropy(self, sentence: str) -> float:
        if not sentence.strip():
            return 2.0
        inputs = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=128
        ).to(next(self.model.parameters()).device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs     = F.softmax(logits[0], dim=-1)
        log_probs = F.log_softmax(logits[0], dim=-1)
        entropy   = -(probs * log_probs).sum(dim=-1)
        return float(entropy.mean().item())

    def _entropy_to_params(self, entropy: float):
        if entropy < self.cfg.dec_temp_low_thresh:
            return self.cfg.dec_temp_focused, self.cfg.dec_top_k_focused
        elif entropy > self.cfg.dec_temp_high_thresh:
            return self.cfg.dec_temp_focused, self.cfg.dec_top_k_focused
        return self.cfg.dec_temp_balanced, self.cfg.dec_top_k_balanced

    @torch.no_grad()
    def generate_with_dec_mod(self, prompt: str) -> str:
        MAX_SENTENCES   = 6
        MAX_TOKENS_SENT = 60
        MIN_TOKENS_SENT = 10

        generated_sentences = []
        current_temp  = self.cfg.dec_temp_balanced
        current_top_k = self.cfg.dec_top_k_balanced
        context       = prompt

        for _ in range(MAX_SENTENCES):
            inputs = self.tokenizer(
                context, return_tensors="pt", truncation=True, max_length=900
            ).to(next(self.model.parameters()).device)

            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS_SENT,
                min_new_tokens=MIN_TOKENS_SENT,
                do_sample=True,
                temperature=current_temp,
                top_k=current_top_k,
                top_p=0.92,
                repetition_penalty=1.1,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            new_text = self.tokenizer.decode(
                out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
            ).strip()

            for punct in [".", "!", "?"]:
                if punct in new_text:
                    new_text = new_text[:new_text.index(punct) + 1]
                    break

            if not new_text or len(new_text.split()) < 3:
                break

            generated_sentences.append(new_text)
            entropy        = self._compute_sentence_entropy(new_text)
            current_temp, current_top_k = self._entropy_to_params(entropy)
            context        = context + " " + new_text

        return " ".join(generated_sentences)

    def generate(
        self, query: str, entity: Optional[str] = None, granularity: str = "sentence"
    ) -> GenerationOutput:
        t0        = time.time()
        retrieved = self.retriever.retrieve(query, entity=entity)
        passages  = [p for p, _ in retrieved] if retrieved else ["No evidence found."]
        prompt    = self.build_prompt(query, passages)

        candidates = []
        for _ in range(self.cfg.num_return_sequences):
            try:
                cand = self.generate_with_dec_mod(prompt)
                if cand.strip():
                    candidates.append(cand)
            except Exception:
                pass

        if not candidates:
            candidates = self.generate_candidates(prompt, n=2)

        best_text, best_score, best_data = None, -1.0, None
        for cand_text in candidates:
            claims = self.segmenter.segment(cand_text, granularity)
            if not claims:
                continue
            uncertainties = self.get_claim_uncertainties(prompt, claims)
            score_info    = self.acc.score_generation(claims, passages, uncertainties)
            fluency       = self.compute_fluency_score(cand_text)
            combined = (
                self.cfg.alpha_factuality * score_info["factuality_score"]
                + self.cfg.alpha_fluency  * fluency
            )
            if combined > best_score:
                best_score = combined
                best_text  = cand_text
                best_data  = {
                    "claims":        claims,
                    "uncertainties": uncertainties,
                    "ver_results":   score_info["results"],
                    "factuality":    score_info["factuality_score"],
                    "ver_calls":     score_info["verification_calls"],
                }

        if best_text is None:
            best_text = candidates[0] if candidates else ""
            best_data = {
                "claims": [], "uncertainties": [], "ver_results": [],
                "factuality": 0.0, "ver_calls": 0,
            }

        return GenerationOutput(
            text=best_text,
            claims=best_data["claims"],
            uncertainties=best_data["uncertainties"],
            verification_results=best_data["ver_results"],
            factuality_score=best_data["factuality"],
            verification_calls=best_data["ver_calls"],
            generation_time_s=time.time() - t0,
            method="UA-ACD-Dec-Mod",
        )
