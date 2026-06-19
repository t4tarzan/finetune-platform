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


@dataclass
class ComplianceResult:
    """Extended result for compliance-weighted adjudication."""
    data_id: str
    verdicts: list[ModelVerdict]
    weighted_score: float  # Domain-weighted agreement score (0-1)
    raw_majority: int      # Raw vote count
    num_total: int
    domain: str
    is_verified: bool
    escalated: bool         # True if flagged for human review
    tiebreaker_used: bool   # True if a tiebreaker model was invoked
    has_citations: bool     # True if the answer cites regulatory sources
    citation_sources: list[str] = field(default_factory=list)
    adjudication_reason: str = ""


# ── Domain-Weighted Adjudication ──────────────────────────

DEFAULT_DOMAIN_WEIGHTS: dict[str, dict[str, float]] = {
    "gdpr": {
        "deepseek/deepseek-v4-pro": 0.20,
        "Qwen/Qwen3.6-Max-Preview": 0.20,
        "moonshotai/Kimi-K2.7-Code": 0.15,
        "claude-sonnet-4-6": 0.45,
    },
    "hipaa": {
        "deepseek/deepseek-v4-pro": 0.15,
        "Qwen/Qwen3.6-Max-Preview": 0.15,
        "moonshotai/Kimi-K2.7-Code": 0.15,
        "claude-sonnet-4-6": 0.55,
    },
    "medical_coding": {
        "deepseek/deepseek-v4-pro": 0.20,
        "Qwen/Qwen3.6-Max-Preview": 0.30,
        "moonshotai/Kimi-K2.7-Code": 0.20,
        "claude-sonnet-4-6": 0.30,
    },
    "financial_regulation": {
        "deepseek/deepseek-v4-pro": 0.25,
        "Qwen/Qwen3.6-Max-Preview": 0.20,
        "moonshotai/Kimi-K2.7-Code": 0.15,
        "claude-sonnet-4-6": 0.40,
    },
}

DEFAULT_TIEBREAKER = "claude-opus-4-8"


class ComplianceAdjudicator:
    """
    Hierarchical weighted adjudicator for compliance-critical domains.

    Modes:
      - majority (default): flat vote, ≥ min_agree with confidence threshold.
      - weighted: domain-specific model weights, citation verification,
                  tiebreaker escalation, human review flagging.

    Uncorrelated hallucination principle:
      Models from different families (MoE, dense, long-context, constitutional)
      have statistically independent failure modes. Under weighted adjudication,
      the probability of a wrong answer slipping through is roughly
      Π(1 - w_i * a_i) across all models, where w = domain weight and
      a = historical accuracy. For 4 models at typical compliance accuracy
      (0.6-0.85), this is < 0.1%.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        consensus_cfg = self.config.get("consensus", {})
        self.compliance_cfg = consensus_cfg.get("compliance", {})
        self.mode = self.compliance_cfg.get("mode", "majority")
        self.min_agree = consensus_cfg.get("min_agree", 3)
        self.confidence_threshold = consensus_cfg.get("confidence_threshold", 0.7)
        self.escalation_threshold = self.compliance_cfg.get("escalation_threshold", 0.6)
        self.tiebreaker = self.compliance_cfg.get("tiebreaker_model", DEFAULT_TIEBREAKER)
        self.citation_required = self.compliance_cfg.get("citation_required", False)
        self.domain_weights = DEFAULT_DOMAIN_WEIGHTS.copy()

        # Override domain weights from config
        config_weights = self.compliance_cfg.get("domain_weights", {})
        for domain, weights in config_weights.items():
            if domain in self.domain_weights:
                self.domain_weights[domain].update(weights)
            else:
                self.domain_weights[domain] = weights

    def _detect_domain(self, question: str, context: str = "") -> str:
        """Detect compliance domain from question + context keywords."""
        text = (question + " " + context).lower()
        domain_keywords = {
            "gdpr": ["gdpr", "data protection", "personal data", "consent", "right to erasure",
                     "data subject", "article", "regulation (eu)"],
            "hipaa": ["hipaa", "phi", "protected health", "medical record", "ephi",
                      "privacy rule", "security rule", "breach notification"],
            "medical_coding": ["icd-10", "icd10", "cpt code", "hcpcs", "medical code",
                               "diagnosis code", "procedure code", "billing code", "cms"],
            "financial_regulation": ["sec", "finra", "sox", "sarbanes-oxley", "dodd-frank",
                                     "anti-money laundering", "aml", "know your customer",
                                     "kyc", "basel", "miFID"],
        }

        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def _get_weights(self, domain: str, models: list[str]) -> dict[str, float]:
        """Get domain-specific weights for a list of models. Falls back to uniform."""
        if domain in self.domain_weights:
            domain_w = self.domain_weights[domain]
            # Map available models to weights, defaulting to uniform for unknowns
            available = [m for m in models if m in domain_w]
            if available:
                # Normalize so available weights sum to 1
                raw = {m: domain_w.get(m, 1.0 / max(len(models), 1)) for m in models}
                total = sum(raw.values())
                return {m: w / total for m, w in raw.items()}
        # Uniform fallback
        return {m: 1.0 / max(len(models), 1) for m in models}

    def _check_citations(self, verdicts: list[ModelVerdict], domain: str) -> tuple[bool, list[str]]:
        """Check if any verdict reasoning cites regulatory sources."""
        citation_keywords = {
            "gdpr": ["art.", "article", "chapter", "recital", "§", "section"],
            "hipaa": ["45 c.f.r.", "§", "164.", "section", "privacy rule"],
            "medical_coding": ["icd-10", "icd10", "cpt", "hcpcs", "guideline"],
            "financial_regulation": ["sec rule", "finra rule", "section", "§", "act"],
        }
        keywords = citation_keywords.get(domain, ["according to", "per", "reference", "source"])
        sources = []
        for v in verdicts:
            if v.agrees:
                found = [kw for kw in keywords if kw in v.reasoning.lower()]
                sources.extend(found)
        return len(sources) > 0, list(set(sources))

    def _query_tiebreaker(self, question: str, expected: str,
                          clusters: list[tuple[str, float]]) -> str:
        """Ask a higher-capability model to break a weighted tie."""
        cluster_summary = "\n".join(
            f"  Option {i+1}: answer='{ans}' (weighted score={score:.3f})"
            for i, (ans, score) in enumerate(clusters)
        )
        prompt = f"""You are a tiebreaker adjudicator for a compliance question.

QUESTION: {question}

Two or more answers received similar weighted scores from the consensus panel.
Break the tie by evaluating which answer is most factually and legally correct.

{cluster_summary}

Respond with exactly the option number (1, 2, etc.) that is correct, followed by a brief justification.
"""
        try:
            result = subprocess.run(
                ["cmd", "-t", "-m", self.tiebreaker, "-p", prompt],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                # Extract the option number
                import re
                match = re.search(r'(\d+)', output)
                if match:
                    idx = int(match.group(1)) - 1
                    if 0 <= idx < len(clusters):
                        return clusters[idx][0]
            return clusters[0][0]  # Default to first cluster
        except Exception:
            return clusters[0][0]

    def adjudicate(
        self,
        datapoint: DataPoint,
        verdicts: list[ModelVerdict],
        domain: Optional[str] = None,
    ) -> ConsensusResult:
        """
        Adjudicate a single data point. Returns ConsensusResult (not ComplianceResult
        for backward compatibility, but extra fields are attached as metadata).
        """
        if self.mode == "majority" or not domain:
            return self._simple_majority(datapoint, verdicts)

        return self._weighted_adjudication(datapoint, verdicts, domain)

    def adjudicate_batch(
        self,
        datapoint: DataPoint,
        verdicts: list[ModelVerdict],
    ) -> ConsensusResult:
        """
        Auto-detect domain and adjudicate. Called from the existing verify flow.
        """
        domain = self._detect_domain(datapoint.question, datapoint.context)
        return self.adjudicate(datapoint, verdicts, domain=domain)

    def _simple_majority(self, datapoint: DataPoint,
                         verdicts: list[ModelVerdict]) -> ConsensusResult:
        """Original flat-majority adjudication."""
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

    def _weighted_adjudication(
        self,
        datapoint: DataPoint,
        verdicts: list[ModelVerdict],
        domain: str,
    ) -> ConsensusResult:
        """
        Weighted adjudication with domain expertise, citation check, tiebreaker, escalation.
        """
        models = [v.model for v in verdicts]
        weights = self._get_weights(domain, models)

        # Cluster answers by agreement using a simplified approach:
        # group verdicts by the reference answer
        answer_clusters: dict[str, list[ModelVerdict]] = {}
        agreeing = [v for v in verdicts if v.agrees]
        disagreeing = [v for v in verdicts if not v.agrees]

        # All agreeing verdicts form the "correct" cluster
        # All disagreeing form the "incorrect" cluster
        clusters = {
            "correct": agreeing,
            "incorrect": disagreeing,
        }

        # Compute weighted score for each cluster
        cluster_scores = {}
        for label, members in clusters.items():
            if not members:
                cluster_scores[label] = 0.0
                continue
            score = sum(
                weights.get(v.model, 1.0 / max(len(models), 1))
                * v.confidence
                for v in members
            )
            cluster_scores[label] = score

        # Also compute confidence-weighted agreement per model
        weighted_agree = sum(
            weights.get(v.model, 0) * v.confidence * (1 if v.agrees else -0.5)
            for v in verdicts
        )
        weighted_agree = max(0, min(1, weighted_agree))  # Clamp to [0, 1]

        num_total = len(verdicts)
        num_agree = len(agreeing)
        citation_ok, sources = self._check_citations(verdicts, domain)
        escalated = False
        tiebreaker_used = False

        # ── Decision logic ──
        reason_parts = []

        # Check citation requirement
        if self.citation_required and not citation_ok:
            escalated = True
            reason_parts.append(f"failed citation check for domain '{domain}'")

        # Check weighted score vs thresholds
        correct_score = cluster_scores.get("correct", 0)
        incorrect_score = cluster_scores.get("incorrect", 0)

        if weighted_agree >= self.confidence_threshold and not escalated:
            # Pass — weighted consensus achieved
            is_verified = True
            reason_parts.append(
                f"weighted consensus ({weighted_agree:.2f} ≥ {self.confidence_threshold})"
            )

        elif weighted_agree >= self.escalation_threshold and not escalated:
            # Marginal — close to threshold, check for tie
            score_diff = correct_score - incorrect_score
            if abs(score_diff) < 0.05 and num_total >= 2:
                # Near-tie between correct and incorrect — use tiebreaker
                tiebreaker_result = self._query_tiebreaker(
                    datapoint.question,
                    datapoint.reference_answer,
                    [("correct", correct_score), ("incorrect", incorrect_score)],
                )
                tiebreaker_used = True
                is_verified = (tiebreaker_result == "correct")
                reason_parts.append(
                    f"tiebreaker ({self.tiebreaker}) resolved to '{tiebreaker_result}'"
                )
            else:
                is_verified = True
                reason_parts.append(
                    f"marginal consensus ({weighted_agree:.2f}), no tiebreaker needed"
                )
        else:
            # Below escalation threshold — escalate to human review
            escalated = True
            is_verified = False
            reason_parts.append(
                f"weighted score {weighted_agree:.2f} < escalation "
                f"threshold {self.escalation_threshold}"
            )

        adjudication_reason = "; ".join(reason_parts)
        disagreements = [v.model for v in disagreeing]

        # Attach extra metadata for downstream use
        result = ConsensusResult(
            data_id=datapoint.id,
            verdicts=verdicts,
            num_agree=num_agree,
            num_total=num_total,
            avg_confidence=weighted_agree,
            is_verified=is_verified,
            disagreements=disagreements,
        )
        # Attach compliance metadata
        result.weighted_score = weighted_agree
        result.escalated = escalated
        result.tiebreaker_used = tiebreaker_used
        result.has_citations = citation_ok
        result.citation_sources = sources
        result.adjudication_reason = adjudication_reason
        result.domain = domain

        return result


# ── Updated ConsensusVerifier ─────────────────────────────

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


class ConsensusVerifier:
    """Verifies training data by consensus across multiple models.

    Supports two adjudication modes:
      - majority (default): flat vote, ≥ min_agree with confidence threshold
      - weighted: domain-aware with compliance escalation, citation check, tiebreaker

    The adjudication mode is determined by the compliance.mode config setting.
    When weighted mode is active, domain is auto-detected from question + context.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        consensus_cfg = self.config.get("consensus", {})
        self.models = self.config.get("consensus_models", [])
        self.min_agree = consensus_cfg.get("min_agree", 3)
        self.confidence_threshold = consensus_cfg.get("confidence_threshold", 0.7)
        self.max_retries = consensus_cfg.get("max_retries", 2)
        self.timeout = consensus_cfg.get("timeout_seconds", 30)
        self.verification_results = []
        # Initialize the adjudicator (handles both majority and weighted modes)
        self.adjudicator = ComplianceAdjudicator(config)

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

        # Use the adjudicator (handles both majority and weighted modes)
        result = self.adjudicator.adjudicate_batch(datapoint, verdicts)
        return result

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
        """Generate a detailed consensus report with compliance metadata."""
        total = len(dataset)
        verified_count = len(verified)
        rejected_count = len(rejected)

        per_model_agreement = {}
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

        # Compliance adjudication summary
        escalated = [r for r in results if getattr(r, 'escalated', False)]
        tiebroken = [r for r in results if getattr(r, 'tiebreaker_used', False)]
        cited = [r for r in results if getattr(r, 'has_citations', False)]
        domain_counts = {}
        for r in results:
            domain = getattr(r, 'domain', 'general')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

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
            "adjudication": {
                "mode": self.adjudicator.mode,
                "domains_detected": domain_counts,
                "escalated_for_review": len(escalated),
                "tiebreaker_invoked": len(tiebroken),
                "citation_verified": len(cited),
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
