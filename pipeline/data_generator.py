"""
Internal Dataset Generator — generates training data using our own models.

No external API keys needed. Uses commandcode API models (30+ available)
for schema inference and data generation. Falls back to local MLX models.

Flow:
  1. User describes the domain in natural language
  2. Schema inference model determines columns
  3. Data generation model generates Q&A pairs
  4. Consensus verifier validates each row
  5. Verified training set written to DuckDB + JSONL

Usage:
  from pipeline.data_generator import DataGenerator
  gen = DataGenerator()
  gen.generate("medical coding", "ICD-10 billing codes with descriptions")
"""

import json
import os
import subprocess
import sys
import time
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

# ── Provider Detection ────────────────────────────────────

def _load_env():
    """Load .env file if it exists."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


def _cmd_available() -> bool:
    """Check if commandcode CLI is available and authenticated."""
    try:
        result = subprocess.run(
            ["cmd", "status"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _bigset_available() -> bool:
    """Check if BigSet CLI is available and API keys are set."""
    try:
        subprocess.run(["bigset", "list"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ollama_available() -> bool:
    """Check if Ollama is running."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", 11434))
    sock.close()
    return result == 0


# Load env on import
_load_env()


AVAILABLE_PROVIDERS = {}
if _cmd_available():
    AVAILABLE_PROVIDERS["cmd"] = "commandcode API (30+ models)"
if _bigset_available():
    AVAILABLE_PROVIDERS["bigset"] = "BigSet (web research)"
if _ollama_available():
    AVAILABLE_PROVIDERS["ollama"] = "Ollama (local models)"

DEFAULT_PROVIDER = "cmd" if "cmd" in AVAILABLE_PROVIDERS else list(AVAILABLE_PROVIDERS.keys())[0] if AVAILABLE_PROVIDERS else None

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


SCHEMA_PROMPT = """You are a dataset designer. Given a domain description, design a structured dataset schema.

Domain: {niche}
Description: {description}

Design 2-4 columns that would make useful training data. Each column should be:
- A natural question someone would ask about this domain
- A factual answer that can be verified

Respond in JSON format ONLY:
{{
  "columns": ["question", "answer"],
  "description": "Brief description of what each column contains",
  "topics": ["topic1", "topic2", "topic3"]
}}"""

GENERATE_PROMPT = """You are a dataset generator. Generate high-quality training data for the domain below.

Domain: {niche}
Description: {description}
Required columns: {columns}

Generate {count} diverse, factual rows of training data. Each row must:
- Ask a realistic question someone in this domain would ask
- Provide a factually correct answer
- Cover different subtopics within the domain

Respond in JSON format ONLY with a JSON array of objects:
[
  {{"question": "What is...?", "reference_answer": "The answer is...", "context": "Optional context about the topic"}},
  ...
]"""


class DataGenerator:
    """Generates training data using available providers — auto-detects what's available."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.models = self.config.get("consensus_models", [])
        self.gen_model = self.models[0] if self.models else "deepseek/deepseek-v4-pro"

        # Detect available providers
        self.provider = DEFAULT_PROVIDER
        self.providers_available = AVAILABLE_PROVIDERS

        if not self.provider:
            print("[WARN] No generation provider available.")
            print("  Install: cmd login (commandcode API) or set API keys in .env")
            print("  See .env.example for details")

    def _query_model(self, model: str, prompt: str, timeout: int = 60) -> Optional[str]:
        """Query a model via available provider."""
        if self.provider == "cmd":
            return self._query_cmd(model, prompt, timeout)
        elif self.provider == "bigset":
            print("  [WARN] BigSet generation not implemented yet")
            return None
        elif self.provider == "ollama":
            return self._query_ollama(model, prompt, timeout)
        return None

    def _query_cmd(self, model: str, prompt: str, timeout: int = 60) -> Optional[str]:
        """Query a model via cmd -p."""
        try:
            result = subprocess.run(
                ["cmd", "-t", "-m", model, "-p", prompt],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            print(f"    [WARN] cmd returned code {result.returncode}")
            return None
        except Exception as e:
            print(f"    [WARN] cmd error: {e}")
            return None

    def _query_ollama(self, model: str, prompt: str, timeout: int = 120) -> Optional[str]:
        """Query a local Ollama model."""
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    def _extract_json(self, text: str):
        """Extract JSON from model response (handles markdown wrapping)."""
        # Try to find JSON array or object in the response
        json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _infer_schema(self, niche: str, description: str) -> dict:
        """Use a model to infer the dataset schema from the description."""
        print(f"  Inferring schema for '{niche}'...")
        prompt = SCHEMA_PROMPT.format(niche=niche, description=description)
        response = self._query_model(self.gen_model, prompt)

        if not response:
            print(f"  [WARN] Schema inference failed, using defaults")
            return {"columns": ["question", "reference_answer"], "topics": [niche]}

        schema = self._extract_json(response)
        if schema and "columns" in schema:
            print(f"  Schema: {', '.join(schema['columns'])}")
            return schema

        print(f"  [WARN] Could not parse schema, using defaults")
        return {"columns": ["question", "reference_answer"], "topics": [niche]}

    def _generate_rows(self, niche: str, description: str, columns: list[str],
                       count: int, batch_size: int = 10) -> list[dict]:
        """Generate training rows using the model."""
        all_rows = []
        batches = (count + batch_size - 1) // batch_size

        for batch in range(batches):
            remaining = min(batch_size, count - len(all_rows))
            print(f"  Generating batch {batch + 1}/{batches} ({remaining} rows)...")

            prompt = GENERATE_PROMPT.format(
                niche=niche,
                description=description,
                columns=", ".join(columns),
                count=remaining,
            )

            response = self._query_model(self.gen_model, prompt, timeout=120)
            if not response:
                print(f"  [WARN] Batch {batch + 1} returned no response")
                continue

            rows = self._extract_json(response)
            if rows and isinstance(rows, list):
                # Normalize to our format
                for row in rows:
                    q = row.get("question") or row.get(columns[0], "")
                    a = row.get("reference_answer") or row.get("answer") or row.get(columns[1] if len(columns) > 1 else columns[0], "")
                    c = row.get("context") or row.get(columns[2] if len(columns) > 2 else "", "")
                    normalized = {"question": q, "reference_answer": a, "context": c}
                    if normalized["question"] and normalized["reference_answer"]:
                        all_rows.append(normalized)
                print(f"    Got {len(rows)} rows from model")
            else:
                print(f"    [WARN] Could not parse batch response")

        return all_rows

    def generate(self, niche: str, description: str, count: int = 50,
                 run_consensus: bool = True, output_dir: str = None) -> dict:
        """
        Full generation pipeline.

        Args:
            niche: Short domain name (e.g., "medical-coding")
            description: Natural language description of the domain
            count: Number of rows to generate
            run_consensus: Whether to run consensus verification after generation
            output_dir: Where to save the output files

        Returns: {
            "niche": niche,
            "rows_generated": total_rows,
            "verified": verified_count,
            "verified_path": path to verified data,
            "consensus_report": report dict (if run_consensus)
        }
        """
        print(f"\n{'='*60}")
        print(f"Generating dataset: {niche}")
        print(f"  Description: {description}")
        print(f"  Target rows: {count}")
        print(f"  Generation model: {self.gen_model}")
        print(f"{'='*60}")

        output_dir = output_dir or os.path.join(
            self.config.get("paths", {}).get("data", "data"), niche
        )
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Infer schema
        schema = self._infer_schema(niche, description)
        columns = schema.get("columns", ["question", "reference_answer"])

        # Step 2: Generate rows
        rows = self._generate_rows(niche, description, columns, count)

        if not rows:
            print(f"\n[ERROR] No rows generated.")
            return {"niche": niche, "rows_generated": 0, "verified": 0}

        print(f"\n  Generated {len(rows)} raw rows")

        # Step 3: Save raw data
        raw_path = os.path.join(output_dir, "raw_generated.jsonl")
        with open(raw_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  Raw data saved: {raw_path}")

        # Step 4: Run consensus verification (optional)
        verified_path = None
        report = None
        verified_count = 0

        if run_consensus:
            print(f"\n  Running consensus verification...")
            from pipeline.consensus_verifier import ConsensusVerifier, DataPoint

            datapoints = [
                DataPoint(
                    id=f"gen-{i+1}",
                    question=row["question"],
                    reference_answer=row["reference_answer"],
                    context=row.get("context", ""),
                    metadata={"source": "generated", "niche": niche},
                )
                for i, row in enumerate(rows)
            ]

            verifier = ConsensusVerifier(self.config)
            verified_dps, rejected_dps, report = verifier.verify(datapoints)
            verified_count = len(verified_dps)

            # Save verified data
            from pipeline.consensus_verifier import save_verified_dataset
            save_verified_dataset(verified_dps, rejected_dps, report, output_dir)
            verified_path = os.path.join(output_dir, "verified_train.jsonl")

            print(f"  Verified: {verified_count}/{len(rows)} rows passed consensus")

        # Step 5: Import into DuckDB
        try:
            from pipeline.data_store import DataStore
            ds = DataStore()
            ds.import_jsonl(raw_path, domain=niche)
            print(f"  Imported {len(rows)} rows into DuckDB")
        except Exception as e:
            print(f"  [WARN] DuckDB import skipped: {e}")

        print(f"{'='*60}")
        print(f"Generation complete:")
        print(f"  Total generated: {len(rows)}")
        print(f"  Consensus passed: {verified_count}")
        print(f"  Raw data: {raw_path}")
        if verified_path:
            print(f"  Verified: {verified_path}")
        print(f"{'='*60}")

        result = {
            "niche": niche,
            "rows_generated": len(rows),
            "verified_count": verified_count,
            "raw_path": raw_path,
            "verified_path": verified_path,
        }
        if report:
            result["consensus_report"] = report

        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Internal Dataset Generator")
    parser.add_argument("--niche", type=str, required=True, help="Short domain name")
    parser.add_argument("--desc", type=str, required=True, help="Domain description")
    parser.add_argument("--count", type=int, default=30, help="Rows to generate")
    parser.add_argument("--skip-consensus", action="store_true", help="Skip consensus verification")

    args = parser.parse_args()
    gen = DataGenerator()
    result = gen.generate(args.niche, args.desc, args.count, run_consensus=not args.skip_consensus)

    print(f"\nResult: {result['rows_generated']} generated, {result['verified_count']} verified")
    print(f"Verified data: {result.get('verified_path', 'N/A')}")


if __name__ == "__main__":
    main()
