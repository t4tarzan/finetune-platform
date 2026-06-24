"""
Eval Harness — Domain-Specific Benchmark Runner

Measures model performance on a held-out test set:
  - Accuracy: % correct answers
  - Grounding: % answers with source citations
  - Consistency: same answer across multiple runs

Supports RAG eval via TurboVec for retrieval-augmented context.
"""

import json
import os
import re
import time
import subprocess
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class EvalResult:
    """Result for a single eval question."""
    question: str
    expected: str
    actual: str
    correct: bool
    has_citation: bool
    latency_ms: float
    confidence_score: float = 0.0


@dataclass
class BenchmarkScore:
    """Aggregate benchmark results for one iteration."""
    niche: str
    model_name: str
    iteration: int
    accuracy: float
    grounding: float
    consistency: float
    num_questions: int
    avg_latency_ms: float
    training_rows: int = 0
    consensus_agreement: float = 0.0
    notes: str = ""


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


class EvalHarness:
    """Evaluates a model against a domain-specific test set."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()

    def _query_cmd_model(self, model_id: str, prompt: str, timeout: int = 60) -> str:
        """Query a model via cmd -p."""
        try:
            start = time.time()
            result = subprocess.run(
                ["cmd", "-t", "-m", model_id, "-p", prompt],
                capture_output=True, text=True, timeout=timeout
            )
            elapsed = (time.time() - start) * 1000
            if result.returncode == 0:
                return result.stdout.strip()
            return f"[ERROR: {result.stderr[:100]}]"
        except Exception as e:
            return f"[ERROR: {e}]"

    def _query_ollama(self, model_name: str, prompt: str) -> str:
        """Query a local Ollama model via its HTTP API (OLLAMA_HOST or localhost)."""
        import urllib.request
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not host.startswith("http"):
            host = "http://" + host
        try:
            payload = json.dumps({
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{host}/api/chat", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            if data.get("error"):
                return f"[ERROR: {data['error']}]"
            return ((data.get("message") or {}).get("content") or "").strip()
        except Exception as e:
            return f"[ERROR: {e}]"

    def _query_inference(self, model_name: str, prompt: str, timeout: int = 120) -> str:
        """Query a model served by the in-app inference server (:7200)."""
        import urllib.request
        port = self.config.get("ports", {}).get("inference_api", 7200)
        try:
            payload = json.dumps({
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            choice = (data.get("choices") or [{}])[0]
            return ((choice.get("message") or {}).get("content") or choice.get("text") or "").strip()
        except Exception as e:
            return f"[ERROR: {e}]"

    def _is_correct(self, actual: str, expected: str) -> bool:
        """Check if the model's answer matches the expected answer."""
        actual_clean = actual.strip().lower()
        expected_clean = expected.strip().lower()
        # Direct match
        if expected_clean in actual_clean:
            return True
        # Fuzzy: check key terms
        exp_words = set(expected_clean.split())
        act_words = set(actual_clean.split())
        if len(exp_words) > 0:
            overlap = len(exp_words & act_words) / len(exp_words)
            return overlap >= 0.5
        return False

    def _has_citation(self, text: str) -> bool:
        """Check if the response includes source citations."""
        citation_patterns = [
            r'\[\d+\]',           # [1], [2]
            r'\(source:',         # (source: ...)
            r'according to',      # "according to [source]"
            r'cited in',          # "cited in [source]"
            r'Reference:',        # "Reference:"
            r'Source:',           # "Source:"
        ]
        return any(re.search(p, text.lower()) for p in citation_patterns)

    def evaluate(
        self,
        model_id: str,
        test_set_path: str,
        model_type: str = "cmd",
        num_repeats: int = 1,
        max_questions: int = None,
    ) -> dict:
        """
        Evaluate a model against a test set.

        Args:
            model_id: Model identifier (cmd model name or Ollama model name)
            test_set_path: Path to JSONL test file with {prompt, completion} or {question, reference_answer}
            model_type: "cmd" for commandcode API, "ollama" for local
            num_repeats: Number of times to repeat for consistency scoring

        Returns:
            BenchmarkScore as dict
        """
        with open(test_set_path) as f:
            test_data = [json.loads(line) for line in f if line.strip()]
        if max_questions is not None:
            test_data = test_data[:max_questions]

        print(f"\nEvaluating {model_id} on {len(test_data)} questions...")
        print(f"{'='*60}")

        all_results = []
        niche_name = os.path.basename(os.path.dirname(test_set_path))
        iteration = 0  # Will be filled by caller

        for repeat in range(num_repeats):
            if num_repeats > 1:
                print(f"\n--- Run {repeat + 1}/{num_repeats} ---")

            for i, row in enumerate(test_data):
                prompt = row.get("prompt", row.get("question", ""))
                expected = row.get("completion", row.get("reference_answer", ""))

                if model_type == "cmd":
                    actual = self._query_cmd_model(model_id, prompt)
                elif model_type == "inference":
                    actual = self._query_inference(model_id, prompt)
                else:
                    actual = self._query_ollama(model_id, prompt)

                correct = self._is_correct(actual, expected)
                cited = self._has_citation(actual)

                result = EvalResult(
                    question=prompt[:80],
                    expected=expected,
                    actual=actual[:200],
                    correct=correct,
                    has_citation=cited,
                    latency_ms=0,  # Not tracking per-query latency here
                )
                all_results.append(result)

                marker = "✓" if correct else "✗"
                cite_mark = "📎" if cited else ""
                print(f"  [{i+1}/{len(test_data)}] {marker} {prompt[:60]}...{cite_mark}")

        # Compute aggregate metrics
        num_questions = len(test_data) * num_repeats
        num_correct = sum(1 for r in all_results if r.correct)
        num_cited = sum(1 for r in all_results if r.has_citation)

        accuracy = num_correct / max(num_questions, 1)
        grounding = num_cited / max(num_questions, 1)

        # Consistency: for questions asked multiple times, check same answer
        consistency = 1.0
        if num_repeats > 1:
            q_repeats = {}
            for i, r in enumerate(all_results):
                q_idx = i // len(test_data)
                if q_idx not in q_repeats:
                    q_repeats[q_idx] = []
                q_repeats[q_idx].append(r.correct)

            # Consistency = % of questions where answer is same across runs
            consistent = sum(
                1 for answers in q_repeats.values()
                if all(a == answers[0] for a in answers)
            )
            consistency = consistent / max(len(q_repeats), 1)

        score = BenchmarkScore(
            niche=niche_name,
            model_name=model_id,
            iteration=iteration,
            accuracy=round(accuracy, 4),
            grounding=round(grounding, 4),
            consistency=round(consistency, 4),
            num_questions=num_questions,
            avg_latency_ms=0,
        )

        print(f"\n{'='*60}")
        print(f"Results for {model_id} on '{niche_name}':")
        print(f"  Accuracy:    {accuracy:.1%} ({num_correct}/{num_questions})")
        print(f"  Grounding:   {grounding:.1%} ({num_cited}/{num_questions})")
        print(f"  Consistency: {consistency:.1%}")
        print(f"{'='*60}")

        return asdict(score)


class BenchmarkLeaderboard:
    """Manages benchmark scores across iterations."""

    def __init__(self, leaderboard_path: str = None):
        self.config = load_config()
        self.leaderboard_path = leaderboard_path or os.path.join(
            self.config.get("paths", {}).get("benchmarks", "benchmarks"),
            "leaderboard.json",
        )

    def load(self) -> dict:
        """Load the leaderboard from disk."""
        if os.path.exists(self.leaderboard_path):
            with open(self.leaderboard_path) as f:
                return json.load(f)
        return {}

    def save(self, leaderboard: dict):
        """Save the leaderboard to disk."""
        os.makedirs(os.path.dirname(self.leaderboard_path), exist_ok=True)
        with open(self.leaderboard_path, "w") as f:
            json.dump(leaderboard, f, indent=2)

    def get_niche_leaderboard(self, niche: str) -> dict:
        """Get the leaderboard for a specific niche."""
        leaderboard = self.load()
        return leaderboard.get(niche, {
            "niche": niche,
            "base_model": "",
            "baseline": None,
            "iterations": [],
        })

    def set_baseline(self, niche: str, base_model: str, score: dict):
        """Set the baseline score for a niche."""
        leaderboard = self.load()
        if niche not in leaderboard:
            leaderboard[niche] = {
                "niche": niche,
                "base_model": base_model,
                "baseline": None,
                "iterations": [],
            }
        leaderboard[niche]["base_model"] = base_model
        leaderboard[niche]["baseline"] = score
        self.save(leaderboard)

    def add_iteration(
        self,
        niche: str,
        score: dict,
        training_rows: int = 0,
        consensus_agreement: float = 0.0,
    ):
        """Add a new iteration result and compute delta from baseline."""
        leaderboard = self.load()
        if niche not in leaderboard or leaderboard[niche]["baseline"] is None:
            print(f"WARNING: No baseline set for '{niche}'. Setting current as baseline.")
            self.set_baseline(niche, score.get("model_name", ""), score)
            return

        baseline = leaderboard[niche]["baseline"]
        iteration_num = len(leaderboard[niche]["iterations"]) + 1

        delta = {}
        for metric in ["accuracy", "grounding", "consistency"]:
            delta[metric] = round(
                score.get(metric, 0) - baseline.get(metric, 0), 4
            )

        entry = {
            "iteration": iteration_num,
            "training_rows": training_rows,
            "consensus_agreement": consensus_agreement,
            "results": {
                "accuracy": score.get("accuracy", 0),
                "grounding": score.get("grounding", 0),
                "consistency": score.get("consistency", 0),
            },
            "delta": delta,
        }

        leaderboard[niche]["iterations"].append(entry)
        self.save(leaderboard)

        print(f"\n📊 Iteration {iteration_num} added to leaderboard for '{niche}'")
        print(f"    Delta: accuracy={delta['accuracy']:+.1%}, "
              f"grounding={delta['grounding']:+.1%}, "
              f"consistency={delta['consistency']:+.1%}")

    def print_leaderboard(self, niche: str = None):
        """Print the leaderboard for display."""
        leaderboard = self.load()

        if niche:
            entries = {niche: leaderboard.get(niche)}
        else:
            entries = leaderboard

        for name, data in entries.items():
            if not data:
                continue
            print(f"\n{'='*60}")
            print(f"Leaderboard: {name}")
            print(f"  Base model: {data.get('base_model', '?')}")
            print(f"{'='*60}")

            baseline = data.get("baseline")
            if baseline:
                print(f"  Baseline: accuracy={baseline['accuracy']:.1%}, "
                      f"grounding={baseline['grounding']:.1%}, "
                      f"consistency={baseline['consistency']:.1%}")

            for it in data.get("iterations", []):
                res = it["results"]
                delta = it["delta"]
                print(f"  Iter {it['iteration']}: "
                      f"acc={res['accuracy']:.1%} ({delta['accuracy']:+.1%}) | "
                      f"ground={res['grounding']:.1%} ({delta['grounding']:+.1%}) | "
                      f"train_rows={it['training_rows']}")
            print(f"{'='*60}")


if __name__ == "__main__":
    # Quick test
    lb = BenchmarkLeaderboard()
    lb.print_leaderboard()
