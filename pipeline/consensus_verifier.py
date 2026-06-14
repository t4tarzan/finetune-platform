"""
Consensus Verification Module

Queries 3+ diverse models via commandcode API, measures agreement,
and only passes consensus-verified data to the training pipeline.

Architecture:
  Raw data → Query N models in parallel → Agreement scoring → Verified set

Usage:
  from consensus_verifier import ConsensusVerifier
  verifier = ConsensusVerifier()
  verified, rejected, report = verifier.verify(dataset)
"""

import json
import subprocess
import tempfile
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from dataclasses import dataclass, field, asdict
import yaml


@dataclass
class DataPoint:
    """A single training data point to verify."""
    id: str
    question: str
    reference_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelVerdict:
    """A single model's response for a data point."""
    model: str
    data_id: str
    agrees: bool  # Does the model confirm this answer is factually correct?
    confidence: float  # 0.0 to 1.0
    reasoning: str = ""
    latency_ms: float = 0.0


@dataclass
class ConsensusResult:
    """The consensus result for a single data point."""
    data_id: str
    verdicts: list[ModelVerdict]
    num_agree: int
    num_total: int
    avg_confidence: float
    is_verified: bool  # Passes consensus threshold?
    disagreements: list[str] = field(default_factory=list)


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


class ConsensusVerifier:
    """Verifies training data by consensus across multiple models."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        consensus_cfg = self.config.get("consensus", {})
        self.models = self.config.get("consensus_models", [])
        self.min_agree = consensus_cfg.get("min_agree", 3)
        self.confidence_threshold = consensus_cfg.get("confidence_threshold", 0.7)
        self.max_retries = consensus_cfg.get("max_retries", 2)
        self.timeout = consensus_cfg.get("timeout_seconds", 30)
        self.verification_results = []

    def _build_prompt(self, datapoint: DataPoint) -> str:
        """Build the verification prompt for a single data point."""
        return f"""You are a strict factual verifier. Your task is to determine if the given answer is factually correct based on the context provided.

CONTEXT:
{datapoint.context[:2000] if datapoint.context else "(No context provided — use your own knowledge)"}

QUESTION: {datapoint.question}

PROPOSED ANSWER: {datapoint.reference_answer}

Evaluate the proposed answer strictly. Respond in JSON format ONLY:
{{
  "agrees": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your assessment"
}}

Where:
- "agrees" is true if the answer is factually correct based on the context/your knowledge
- "confidence" is your confidence level (0.0 = no confidence, 1.0 = completely certain)
- "reasoning" is a short explanation of why you agree or disagree"""

    def _query_model(self, model: str, prompt: str, retries: int = 0) -> Optional[str]:
        """Query a single model via cmd -p and return the raw response."""
        try:
            cmd = [
                "cmd", "-t", "-m", model,
                "-p", prompt
            ]
            start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            elapsed = (time.time() - start) * 1000

            if result.returncode != 0:
                err = result.stderr.strip()[:200]
                print(f"  [WARN] {model} returned code {result.returncode}: {err}")
                if retries < self.max_retries:
                    return self._query_model(model, prompt, retries + 1)
                return None

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            print(f"  [WARN] {model} timed out after {self.timeout}s")
            if retries < self.max_retries:
                return self._query_model(model, prompt, retries + 1)
            return None
        except Exception as e:
            print(f"  [WARN] {model} error: {e}")
            if retries < self.max_retries:
                return self._query_model(model, prompt, retries + 1)
            return None

    def _parse_response(self, model: str, data_id: str, raw: Optional[str]) -> Optional[ModelVerdict]:
        """Parse a model's JSON response into a ModelVerdict."""
        if not raw:
            return None

        # Try to extract JSON from the response (handle markdown-wrapped JSON)
        json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if not json_match:
            print(f"  [WARN] {model}: no JSON found in response: {raw[:100]}...")
            return None

        try:
            data = json.loads(json_match.group())
            return ModelVerdict(
                model=model,
                data_id=data_id,
                agrees=bool(data.get("agrees", False)),
                confidence=float(data.get("confidence", 0.0)),
                reasoning=str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"  [WARN] {model}: JSON parse error: {e}")
            return None

    def _verify_single(self, datapoint: DataPoint) -> ConsensusResult:
        """Verify a single data point across all consensus models."""
        prompt = self._build_prompt(datapoint)
        verdicts = []

        for model in self.models:
            raw = self._query_model(model, prompt)
            verdict = self._parse_response(model, datapoint.id, raw)
            if verdict:
                verdicts.append(verdict)

        num_agree = sum(1 for v in verdicts if v.agrees)
        num_total = len(verdicts)
        avg_conf = sum(v.confidence for v in verdicts) / max(num_total, 1)

        disagreements = [
            v.model for v in verdicts
            if not v.agrees or v.confidence < self.confidence_threshold
        ]

        is_verified = (
            num_agree >= self.min_agree
            and avg_conf >= self.confidence_threshold
        )

        return ConsensusResult(
            data_id=datapoint.id,
            verdicts=verdicts,
            num_agree=num_agree,
            num_total=num_total,
            avg_confidence=avg_conf,
            is_verified=is_verified,
            disagreements=disagreements,
        )

    def verify(self, dataset: list[DataPoint], max_workers: int = 3) -> tuple:
        """
        Verify a dataset across all consensus models.

        Returns:
            (verified_data, rejected_data, consensus_report)
        """
        verified = []
        rejected = []
        results = []

        print(f"Verifying {len(dataset)} data points across {len(self.models)} models...")
        print(f"Models: {', '.join(self.models)}")
        print(f"Threshold: {self.min_agree}/{len(self.models)} agreement, "
              f"confidence >= {self.confidence_threshold}")
        print()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._verify_single, dp): dp for dp in dataset}

            for i, future in enumerate(as_completed(futures)):
                dp = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.verification_results.append(result)

                    status = "✓ VERIFIED" if result.is_verified else "✗ REJECTED"
                    print(f"  [{i+1}/{len(dataset)}] {dp.id}: {status} "
                          f"(agree: {result.num_agree}/{result.num_total}, "
                          f"conf: {result.avg_confidence:.2f})")

                    if result.is_verified:
                        verified.append(dp)
                    else:
                        rejected.append(dp)

                except Exception as e:
                    print(f"  [{i+1}/{len(dataset)}] {dp.id}: ERROR {e}")
                    rejected.append(dp)

        report = self._generate_report(dataset, verified, rejected, results)
        return verified, rejected, report

    def _generate_report(self, dataset, verified, rejected, results):
        """Generate a detailed consensus report."""
        total = len(dataset)
        verified_count = len(verified)
        rejected_count = len(rejected)

        per_model_agreement = {}
        # Collect all verdicts from all results
        all_verdicts = [v for r in results for v in r.verdicts]
        for model in self.models:
            model_verdicts = [v for v in all_verdicts if v.model == model]
            if model_verdicts:
                agrees = sum(1 for v in model_verdicts if v.agrees)
                per_model_agreement[model] = {
                    "total": len(model_verdicts),
                    "agree": agrees,
                    "agree_rate": round(agrees / len(model_verdicts), 3),
                    "avg_confidence": round(
                        sum(v.confidence for v in model_verdicts) / len(model_verdicts), 3
                    )
                }

        report = {
            "summary": {
                "total_datapoints": total,
                "verified": verified_count,
                "rejected": rejected_count,
                "verification_rate": round(verified_count / max(total, 1), 3),
                "min_agreement_required": self.min_agree,
                "confidence_threshold": self.confidence_threshold,
                "models_used": self.models,
            },
            "per_model_agreement": per_model_agreement,
            "rejected_ids": [dp.id for dp in rejected],
            "verified_ids": [dp.id for dp in verified],
        }
        return report


def save_verified_dataset(
    verified: list[DataPoint],
    rejected: list[DataPoint],
    report: dict,
    output_dir: str = "data",
):
    """Save verified and rejected datasets to jsonl files."""
    os.makedirs(output_dir, exist_ok=True)

    def dp_to_dict(dp):
        return {
            "id": dp.id,
            "question": dp.question,
            "reference_answer": dp.reference_answer,
            "context": dp.context,
            "metadata": dp.metadata,
        }

    with open(os.path.join(output_dir, "verified_train.jsonl"), "w") as f:
        for dp in verified:
            f.write(json.dumps(dp_to_dict(dp)) + "\n")

    with open(os.path.join(output_dir, "rejected_train.jsonl"), "w") as f:
        for dp in rejected:
            f.write(json.dumps(dp_to_dict(dp)) + "\n")

    report_path = os.path.join(output_dir, "consensus_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved {len(verified)} verified, {len(rejected)} rejected datapoints")
    print(f"Report: {report_path}")
    return report


if __name__ == "__main__":
    # Quick test with sample data
    test_data = [
        DataPoint(
            id="test-001",
            question="What company developed the Transformer architecture?",
            reference_answer="Google developed the Transformer architecture in the 2017 paper 'Attention Is All You Need'.",
            context="The Transformer architecture was introduced by Vaswani et al. from Google in 2017.",
        ),
        DataPoint(
            id="test-002",
            question="What is the capital of France?",
            reference_answer="Paris is the capital of France.",
            context="",
        ),
    ]

    verifier = ConsensusVerifier()
    verified, rejected, report = verifier.verify(test_data)
    save_verified_dataset(verified, rejected, report)
    print(json.dumps(report, indent=2))
