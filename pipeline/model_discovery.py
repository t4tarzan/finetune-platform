"""
Model Discovery Agent — searches HuggingFace for models matching a niche,
evaluates them against a domain test set, and returns a ranked shortlist.

Flow:
  1. Parse niche → extract keywords, task type, size constraints
  2. Search HF Hub matching criteria
  3. Filter by architecture compatibility (MLX-friendly)
  4. Run candidates through local eval harness
  5. Rank by accuracy vs parameter count (efficiency frontier)
  6. Recommend: use as-is if threshold met, else fine-tune
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@dataclass
class ModelCandidate:
    """A model found and evaluated by the discovery agent."""
    model_id: str
    hf_url: str
    pipeline_tag: str
    downloads: int
    likes: int
    parameters: str = ""
    architecture: str = ""
    license: str = ""
    eval_accuracy: Optional[float] = None
    eval_latency_ms: Optional[float] = None
    eval_grounding: Optional[float] = None
    loaded_in_memory_gb: Optional[float] = None
    is_mlx_compatible: bool = False
    rank_score: float = 0.0  # Composite score for ranking
    recommendation: str = ""  # "use-as-is", "fine-tune", "skip"


@dataclass
class DiscoveryReport:
    """Full report from a discovery run."""
    niche: str
    niche_keywords: list[str]
    candidates: list[dict]
    top_recommendation: Optional[dict]
    can_skip_training: bool
    summary: str


class ModelDiscoveryAgent:
    """Searches HuggingFace for models matching a niche and evaluates them."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.api = None  # Lazy import

    def _get_api(self):
        if self.api is None:
            from huggingface_hub import HfApi
            self.api = HfApi()
        return self.api

    def _parse_niche(self, niche_desc: str) -> dict:
        """Parse a niche description into search parameters."""
        niche_lower = niche_desc.lower()

        # Determine task type
        task = "text-generation"
        if any(w in niche_lower for w in ["embedding", "embed", "sentence", "similarity"]):
            task = "feature-extraction"
        elif any(w in niche_lower for w in ["classify", "classification", "classifier"]):
            task = "text-classification"
        elif any(w in niche_lower for w in ["qa", "question", "answer"]):
            task = "text-generation"  # QA is text-gen
        elif any(w in niche_lower for w in ["code", "coding", "programming"]):
            task = "text-generation"

        # Extract meaningful keywords from the description
        stop_words = {"the", "a", "an", "in", "on", "at", "for", "to", "of",
                      "and", "or", "is", "are", "was", "were", "with", "from",
                      "that", "this", "these", "those", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did", "will", "would",
                      "could", "should", "may", "might", "shall", "can", "need",
                      "using", "based", "their", "them", "they", "its", "also",
                      "very", "just", "about", "than", "then", "each", "some"}

        words = niche_lower.replace(",", "").replace(".", "").replace("\n", " ").split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return {
            "task": task,
            "keywords": keywords[:10],
            "search_query": " ".join(keywords[:5]),
        }

    def search_hf(
        self,
        niche_desc: str,
        max_candidates: int = 10,
        min_downloads: int = 1000,
        max_parameters: Optional[str] = "70B",
    ) -> list[ModelCandidate]:
        """Search HuggingFace Hub for candidate models."""
        parsed = self._parse_niche(niche_desc)
        api = self._get_api()

        print(f"\nSearching HuggingFace for: '{niche_desc}'")
        print(f"  Task: {parsed['task']}")
        print(f"  Keywords: {parsed['keywords']}")
        print(f"  Search query: '{parsed['search_query']}'")
        print(f"  Max candidates: {max_candidates}")
        print()

        # Search by task, sorted by downloads
        models = list(api.list_models(
            pipeline_tag=parsed["task"],
            sort="downloads",
            limit=50,
        ))

        # Also do a keyword search on top of the task search
        keyword_models = list(api.list_models(
            search=parsed["search_query"],
            sort="downloads",
            limit=20,
        ))

        # Merge, deduplicate by model_id
        seen = set()
        candidates = []

        for model in models + keyword_models:
            if model.modelId in seen:
                continue
            seen.add(model.modelId)

            # Filter by minimum downloads
            if (model.downloads or 0) < min_downloads:
                continue

            # Rough parameter size estimate from modelId naming
            params_str = self._estimate_parameters(model.modelId)

            # Architecture guess
            arch = self._detect_architecture(model.modelId)

            # MLX compatibility — check if the model has MLX variants
            is_mlx = "mlx" in model.modelId.lower() or any(
                tag and ("mlx" in tag.lower() if tag else False)
                for tag in (getattr(model, 'tags', None) or [])
            )

            candidate = ModelCandidate(
                model_id=model.modelId,
                hf_url=f"https://huggingface.co/{model.modelId}",
                pipeline_tag=model.pipeline_tag or parsed["task"],
                downloads=model.downloads or 0,
                likes=model.likes or 0,
                parameters=params_str,
                architecture=arch,
                license=model.card_data.get("license", "") if hasattr(model, "card_data") and model.card_data else "",
                is_mlx_compatible=is_mlx,
            )
            candidates.append(candidate)

        # Sort by composite score (downloads + likes weighted)
        max_downloads = max((c.downloads for c in candidates), default=1)
        max_likes = max((c.likes for c in candidates), default=1)

        for c in candidates:
            dl_score = c.downloads / max_downloads if max_downloads > 0 else 0
            like_score = c.likes / max_likes if max_likes > 0 else 0
            mlx_bonus = 0.2 if c.is_mlx_compatible else 0
            c.rank_score = round(dl_score * 0.5 + like_score * 0.3 + mlx_bonus, 3)

        candidates.sort(key=lambda c: c.rank_score, reverse=True)

        return candidates[:max_candidates]

    def _estimate_parameters(self, model_id: str) -> str:
        """Rough estimate of parameter count from model name."""
        import re
        model_lower = model_id.lower()

        patterns = [
            (r'(\d+)b', lambda m: f"{m.group(1)}B"),
            (r'(\d+)m', lambda m: f"{m.group(1)}M"),
            (r'-(\d+)-', lambda m: f"{m.group(1)}B" if int(m.group(1)) < 1000 else f"{m.group(1)}M"),
        ]

        for pattern, formatter in patterns:
            match = re.search(pattern, model_lower)
            if match:
                return formatter(match)

        return "unknown"

    def _detect_architecture(self, model_id: str) -> str:
        """Detect model architecture family from name."""
        model_lower = model_id.lower()
        archs = {
            "qwen": "Qwen",
            "llama": "Llama",
            "mistral": "Mistral",
            "gemma": "Gemma",
            "deepseek": "DeepSeek",
            "phi": "Phi",
            "falcon": "Falcon",
            "mpt": "MPT",
            "olmo": "OLMo",
            "solar": "Solar",
            "nemotron": "Nemotron",
            "mixtral": "Mixtral",
            "starcoder": "StarCoder",
            "codellama": "CodeLlama",
        }

        for key, name in archs.items():
            if key in model_lower:
                return name
        return "unknown"

    def evaluate_candidate(
        self,
        candidate: ModelCandidate,
        test_set_path: str,
        use_mlx: bool = True,
    ) -> ModelCandidate:
        """
        Evaluate a model candidate by loading it locally and running the test set.
        For models that aren't MLX-native, tries to load via transformers + MLX.
        """
        from pipeline.eval_harness import EvalHarness

        print(f"\n  Evaluating: {candidate.model_id}")

        # If it's MLX-native, use the eval harness directly
        if candidate.is_mlx_compatible or use_mlx:
            try:
                harness = EvalHarness(self.config)
                # Use cmd to query the model via MLX/HF
                from mlx_lm import load

                start = time.time()
                model, tokenizer = load(candidate.model_id, tokenizer_config={"trust_remote_code": True})
                load_time = time.time() - start

                # Rough memory estimate
                import mlx.core as mx
                try:
                    peak = mx.get_peak_memory() / 1e9
                except Exception:
                    peak = 0

                candidate.loaded_in_memory_gb = round(peak, 2) if peak else 0
                print(f"    Loaded: {load_time:.1f}s, peak mem: {candidate.loaded_in_memory_gb}GB")

                # Run a quick eval using the eval harness cmd approach
                # (We use the model directly for a real eval)
                # For now, run a small batch test
                eval_results = self._quick_eval(model, tokenizer, test_set_path)
                candidate.eval_accuracy = eval_results.get("accuracy", 0)
                candidate.eval_latency_ms = eval_results.get("latency_ms", 0)
                candidate.eval_grounding = eval_results.get("grounding", 0)

                print(f"    Eval: accuracy={candidate.eval_accuracy:.1%}, "
                      f"latency={candidate.eval_latency_ms:.0f}ms")

            except Exception as e:
                print(f"    [SKIP] Could not load locally: {e}")
        else:
            print(f"    [SKIP] Not MLX-compatible, would need transformers")
            candidate.recommendation = "skip"

        return candidate

    def _quick_eval(self, model, tokenizer, test_set_path: str, num_samples: int = 5) -> dict:
        """Quick evaluation using the loaded MLX model directly."""
        if not os.path.exists(test_set_path):
            return {"accuracy": 0, "latency_ms": 0, "grounding": 0}

        with open(test_set_path) as f:
            data = [json.loads(l) for l in f if l.strip()]

        if len(data) == 0:
            return {"accuracy": 0, "latency_ms": 0, "grounding": 0}

        data = data[:num_samples]
        correct = 0
        total_latency = 0

        from mlx_lm import generate as mlx_generate

        for row in data:
            prompt = row.get("prompt", row.get("question", ""))
            expected = row.get("completion", row.get("reference_answer", ""))

            start = time.time()
            response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=64, verbose=False)
            elapsed = (time.time() - start) * 1000
            total_latency += elapsed

            # Check answer
            if expected.lower().strip() in response.lower().strip():
                correct += 1

        return {
            "accuracy": correct / max(len(data), 1),
            "latency_ms": total_latency / max(len(data), 1),
            "grounding": 0,  # Would need citation check
        }

    def rank_candidates(
        self,
        candidates: list[ModelCandidate],
        accuracy_threshold: float = 0.7,
    ) -> list[ModelCandidate]:
        """
        Rank candidates by efficiency frontier (accuracy vs size).
        Recommend: use-as-is if accuracy ≥ threshold, fine-tune if close, skip otherwise.
        """
        for c in candidates:
            if c.eval_accuracy is not None:
                if c.eval_accuracy >= accuracy_threshold:
                    c.recommendation = "use-as-is"
                elif c.eval_accuracy >= accuracy_threshold * 0.7:
                    c.recommendation = "fine-tune"
                else:
                    c.recommendation = "skip"

        return candidates

    def discover(
        self,
        niche_desc: str,
        test_set_path: Optional[str] = None,
        max_candidates: int = 10,
        accuracy_threshold: float = 0.7,
        evaluate_locally: bool = True,
    ) -> DiscoveryReport:
        """
        Full discovery pipeline: search → evaluate → rank → recommend.

        Returns a DiscoveryReport with ranked candidates and recommendation.
        """
        parsed = self._parse_niche(niche_desc)

        # Step 1: Search
        candidates = self.search_hf(niche_desc, max_candidates=max_candidates)

        if not candidates:
            return DiscoveryReport(
                niche=niche_desc,
                niche_keywords=parsed["keywords"],
                candidates=[],
                top_recommendation=None,
                can_skip_training=False,
                summary="No candidates found on HuggingFace matching the criteria.",
            )

        # Step 2: Evaluate locally (if test set provided)
        if evaluate_locally and test_set_path and os.path.exists(test_set_path):
            print(f"\nEvaluating candidates against: {test_set_path}")
            for i, candidate in enumerate(candidates):
                print(f"\n[{i+1}/{len(candidates)}] {candidate.model_id}")
                candidates[i] = self.evaluate_candidate(candidate, test_set_path)

        # Step 3: Rank
        candidates = self.rank_candidates(candidates, accuracy_threshold)

        # Step 4: Determine if we can skip training
        use_as_is = [c for c in candidates if c.recommendation == "use-as-is"]
        fine_tune = [c for c in candidates if c.recommendation == "fine-tune"]

        top = candidates[0] if candidates else None
        can_skip = len(use_as_is) > 0

        # Build summary
        if can_skip:
            summary = (
                f"Found {len(use_as_is)} models that already meet the accuracy threshold. "
                f"Top: {use_as_is[0].model_id} ({use_as_is[0].eval_accuracy:.1%}). "
                "No training needed — use as-is."
            )
        elif fine_tune:
            summary = (
                f"No models meet the threshold. Best candidate: {fine_tune[0].model_id} "
                f"({fine_tune[0].eval_accuracy:.1%}). Recommend fine-tuning."
            )
        else:
            summary = "No suitable models found. Recommend training from scratch."

        report = DiscoveryReport(
            niche=niche_desc,
            niche_keywords=parsed["keywords"],
            candidates=[asdict(c) for c in candidates],
            top_recommendation=asdict(top) if top else None,
            can_skip_training=can_skip,
            summary=summary,
        )

        return report

    def print_report(self, report: DiscoveryReport):
        """Print a human-readable discovery report."""
        print(f"\n{'='*60}")
        print(f"Model Discovery Report")
        print(f"{'='*60}")
        print(f"  Niche: {report.niche}")
        print(f"  Keywords: {', '.join(report.niche_keywords)}")
        print(f"  Candidates evaluated: {len(report.candidates)}")
        print(f"\n  {report.summary}")
        print(f"\n  Ranked candidates:")

        for i, c in enumerate(report.candidates):
            acc = c.get("eval_accuracy", "N/A")
            acc_str = f"{acc:.1%}" if isinstance(acc, float) else acc
            lat = c.get("eval_latency_ms", "N/A")
            lat_str = f"{lat:.0f}ms" if isinstance(lat, (int, float)) else lat
            rec = c.get("recommendation", "")
            marker = {"use-as-is": "✓", "fine-tune": "→", "skip": "✗"}.get(rec, "?")

            print(f"  {marker} [{i+1}] {c['model_id']}")
            print(f"       Acc: {acc_str} · Latency: {lat_str} · Downloads: {c['downloads']:,}")
            if rec:
                print(f"       {rec}")

        if report.top_recommendation:
            top = report.top_recommendation
            print(f"\n  Top recommendation: {top['model_id']}")
            print(f"  Recommendation: {top.get('recommendation', '?')}")
            if top.get("eval_accuracy"):
                print(f"  Accuracy: {top['eval_accuracy']:.1%}")

        print(f"{'='*60}")


def main():
    """CLI entry point for model discovery."""
    import argparse

    parser = argparse.ArgumentParser(description="Model Discovery Agent")
    parser.add_argument("--niche", type=str, required=True, help="Niche description (e.g. 'medical coding QA')")
    parser.add_argument("--test-set", type=str, default=None, help="Path to test JSONL for evaluation")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.7, help="Accuracy threshold for 'use-as-is'")
    parser.add_argument("--skip-eval", action="store_true", help="Skip local evaluation")

    args = parser.parse_args()

    agent = ModelDiscoveryAgent()
    report = agent.discover(
        niche_desc=args.niche,
        test_set_path=args.test_set,
        max_candidates=args.max_candidates,
        accuracy_threshold=args.threshold,
        evaluate_locally=not args.skip_eval,
    )

    agent.print_report(report)

    # Save report
    output_dir = "data/discovery"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"discovery_{args.niche.replace(' ', '_')[:30]}.json")
    # Clean path
    import re
    report_path = re.sub(r'[^a-zA-Z0-9_/.-]', '', report_path)
    with open(report_path, "w") as f:
        json.dump({
            "niche": report.niche,
            "niche_keywords": report.niche_keywords,
            "candidates": report.candidates,
            "top_recommendation": report.top_recommendation,
            "can_skip_training": report.can_skip_training,
            "summary": report.summary,
        }, f, indent=2)

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
