"""
Recursive Loop Orchestrator

Ties together the full pipeline: generate data → verify → fine-tune → eval → repeat.

Each iteration:
  1. Describe niche → BigSet generates dataset
  2. Consensus verify with 3+ models → verified training set
  3. Fine-tune with MLX LoRA → exported model
  4. Eval against baseline → benchmark delta
  5. If improved: save, register, analyze gaps for next iteration
"""

import json
import os
import sys
import time
import subprocess
import yaml
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.consensus_verifier import (
    ConsensusVerifier, DataPoint, save_verified_dataset
)
from pipeline.train_qlora import prepare_training_data, fine_tune, export_to_gguf, register_with_ollama
from pipeline.eval_harness import EvalHarness, BenchmarkLeaderboard


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


class RecursiveLoop:
    """Orchestrates the recursive fine-tuning loop."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.eval_harness = EvalHarness(self.config)
        self.leaderboard = BenchmarkLeaderboard()
        self.pipeline_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.pipeline_dir)

    def generate_dataset_bigset(self, niche_desc: str, niche_name: str, max_rows: int = 50) -> str:
        """Generate a dataset using BigSet from a natural language description."""
        output_path = os.path.join(
            self.project_root, "data", niche_name,
            f"bigset_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Generating dataset via BigSet")
        print(f"{'='*60}")
        print(f"  Description: {niche_desc}")
        print(f"  Max rows: {max_rows}")
        print(f"  Output: {output_path}")

        # BigSet CLI: bigset create <description> --rows <N> --wait --csv <path>
        cmd = [
            "bigset", "create", niche_desc,
            "--rows", str(max_rows),
            "--wait",
            "--csv", output_path,
        ]

        print(f"  Running: {' '.join(cmd)}")
        print(f"  This may take 2-5 minutes...")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0 or os.path.exists(output_path):
                print(f"  ✓ BigSet dataset generated: {output_path}")
                return output_path
            else:
                print(f"  ✗ BigSet failed: {result.stderr[:500]}")
                print(f"  Falling back: no dataset generated")
                return None
        except subprocess.TimeoutExpired:
            print(f"  ✗ BigSet timed out after 10 minutes")
            return None
        except FileNotFoundError:
            print(f"  ✗ BigSet not found. Install with: npm install --global @adamexu/bigset")
            print(f"  Then start with: bigset")
            return None

    def bigset_csv_to_datapoints(self, csv_path: str, niche_desc: str) -> list[DataPoint]:
        """Convert BigSet CSV output to DataPoint objects for consensus verification."""
        import csv

        datapoints = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Use all columns as context, pick first as question/answer
                columns = list(row.keys())
                if len(columns) == 0:
                    continue

                # Build a question from the data
                question = f"In the domain of '{niche_desc}', what is the {columns[0]} for this entry?"
                reference = row[columns[0]]
                context = "; ".join(f"{k}: {v}" for k, v in row.items())

                datapoints.append(DataPoint(
                    id=f"bigset-{i+1}",
                    question=question,
                    reference_answer=reference,
                    context=context,
                    metadata={"source": csv_path, "row": i + 1, "columns": columns},
                ))

        return datapoints

    def run_iteration(
        self,
        niche_name: str,
        niche_desc: str,
        iteration: int = 1,
        epochs: int = None,
        max_rows: int = 50,
        skip_data_generation: bool = False,
        existing_data_path: str = None,
    ) -> dict:
        """
        Run a single iteration of the recursive loop.

        Args:
            niche_name: Short name for the domain (e.g. "ai-startups-sf")
            niche_desc: Natural language description for BigSet
            iteration: Iteration number
            epochs: Training epochs
            max_rows: Max rows to generate
            skip_data_generation: Use existing data instead of BigSet
            existing_data_path: Path to existing verified data jsonl

        Returns:
            dict with iteration results
        """
        print(f"\n{'#'*60}")
        print(f"# ITERATION {iteration}: {niche_name}")
        print(f"# {niche_desc}")
        print(f"{'#'*60}")

        # Step 0: Load or generate data
        if skip_data_generation and existing_data_path:
            verified_path = existing_data_path
            print(f"Using existing data: {verified_path}")
        else:
            # Step 0a: Generate dataset via BigSet
            csv_path = self.generate_dataset_bigset(niche_desc, niche_name, max_rows)
            if not csv_path or not os.path.exists(csv_path):
                print("No data generated. Aborting iteration.")
                return {"iteration": iteration, "status": "failed", "reason": "no_data"}

            # Step 0b: Convert CSV to DataPoints
            raw_datapoints = self.bigset_csv_to_datapoints(csv_path, niche_desc)
            print(f"  Converted {len(raw_datapoints)} rows from BigSet CSV")

            if len(raw_datapoints) == 0:
                print("No data points found. Aborting iteration.")
                return {"iteration": iteration, "status": "failed", "reason": "empty_data"}

            # Step 1: Consensus verification
            verifier = ConsensusVerifier(self.config)
            verified_dps, rejected_dps, report = verifier.verify(raw_datapoints[:max_rows])

            if len(verified_dps) == 0:
                print("No data points passed consensus. Aborting iteration.")
                return {"iteration": iteration, "status": "failed", "reason": "no_consensus"}

            data_dir = os.path.join(self.project_root, "data", niche_name)
            save_verified_dataset(verified_dps, rejected_dps, report, data_dir)
            verified_path = os.path.join(data_dir, "verified_train.jsonl")

            consensus_rate = report["summary"]["verification_rate"]

        # Step 2: Prepare training data (creates train/valid split)
        niche_data_dir = os.path.join(self.project_root, "data", niche_name, f"iter_{iteration}")
        prepare_training_data(
            verified_path=verified_path,
            output_dir=niche_data_dir,
            test_split=self.config.get("eval", {}).get("test_split", 0.2),
        )

        # Count training rows
        with open(os.path.join(niche_data_dir, "train.jsonl")) as f:
            num_train = sum(1 for _ in f if _.strip())

        # Step 3: Run baseline eval if this is iteration 1
        baseline_model = self.config.get("base_model", "")
        niche_leaderboard = self.leaderboard.get_niche_leaderboard(niche_name)

        if iteration == 1 and niche_leaderboard.get("baseline") is None:
            print(f"\n{'='*60}")
            print("Establishing baseline benchmark...")
            print(f"{'='*60}")

            base_model_id = self.config.get("consensus_models", [None])[0]
            if base_model_id:
                baseline_score = self.eval_harness.evaluate(
                    model_id=base_model_id,
                    test_set_path=os.path.join(niche_data_dir, "valid.jsonl"),
                    model_type="cmd",
                    num_repeats=1,
                )
                self.leaderboard.set_baseline(niche_name, base_model_id, baseline_score)
                print(f"  Baseline set: accuracy={baseline_score['accuracy']:.1%}")

        # Step 4: Fine-tune
        adapter_path = os.path.join(
            self.project_root,
            self.config.get("paths", {}).get("adapter_path", "models/adapters"),
            f"{niche_name}-iter{iteration}",
        )

        fine_tune(
            niche=f"{niche_name}-iter{iteration}",
            data_dir=niche_data_dir,
            config=self.config,
            epochs=epochs,
            adapter_path=adapter_path,
        )

        # Step 5: Export
        merged_path = export_to_gguf(
            niche=f"{niche_name}-iter{iteration}",
            adapter_path=adapter_path,
            export_dir=os.path.join(
                self.project_root,
                self.config.get("paths", {}).get("export_path", "models/gguf"),
            ),
            config=self.config,
        )

        # Step 6: Register with Ollama
        register_with_ollama(
            niche=f"{niche_name}-iter{iteration}",
            model_path=merged_path,
        )

        # Step 7: Eval the fine-tuned model (via Ollama)
        ollama_model = f"{niche_name}-iter{iteration}-v1"
        eval_score = self.eval_harness.evaluate(
            model_id=ollama_model,
            test_set_path=os.path.join(niche_data_dir, "valid.jsonl"),
            model_type="ollama",
        )

        # Step 8: Record to leaderboard
        consensus_agreement = report["summary"]["verification_rate"] if not skip_data_generation else 1.0
        self.leaderboard.add_iteration(
            niche=niche_name,
            score=eval_score,
            training_rows=num_train,
            consensus_agreement=consensus_agreement,
        )

        # Step 9: Analyze gaps (find failure cases for next iteration)
        failed_eval_path = os.path.join(niche_data_dir, "eval_failures.jsonl")
        # In a full implementation, we'd collect actual eval failures here.
        # For now, mark the gap analysis as pending.

        print(f"\n{'='*60}")
        print(f"Iteration {iteration} complete!")
        print(f"  Model: {ollama_model}")
        print(f"  Accuracy: {eval_score.get('accuracy', '?'):.1%}")
        self.leaderboard.print_leaderboard(niche_name)
        print(f"{'='*60}")

        return {
            "iteration": iteration,
            "status": "completed",
            "model_name": ollama_model,
            "accuracy": eval_score.get("accuracy"),
            "training_rows": num_train,
            "consensus_rate": consensus_agreement if not skip_data_generation else None,
        }

    def run_recursive(
        self,
        niche_name: str,
        niche_desc: str,
        max_iterations: int = 5,
        epochs_per_iter: int = None,
        max_rows: int = 50,
        improvement_threshold: float = 0.01,  # Stop if improvement < 1%
        patience: int = 2,  # Stop after N iterations without meaningful improvement
        skip_data_generation: bool = False,
        existing_data_path: str = None,
    ):
        """
        Run the full recursive fine-tuning loop.

        Stops when:
        - max_iterations reached
        - improvement drops below threshold for `patience` consecutive iterations
        - accuracy reaches 95%+
        """
        results = []
        no_improvement_count = 0
        best_score = 0.0

        config = load_config()
        epochs_per_iter = epochs_per_iter or config.get("training", {}).get("epochs", 3)
        baseline_model = config.get("base_model", "").split("/")[-1]

        print(f"\n{'#'*60}")
        print(f"# RECURSIVE FINE-TUNING LOOP")
        print(f"# Niche: {niche_name}")
        print(f"# Description: {niche_desc}")
        print(f"# Base Model: {baseline_model}")
        print(f"# Max Iterations: {max_iterations}")
        print(f"# Epochs per Iter: {epochs_per_iter}")
        print(f"# Improvement Threshold: {improvement_threshold:.1%}")
        print(f"{'#'*60}")

        for i in range(1, max_iterations + 1):
            result = self.run_iteration(
                niche_name=niche_name,
                niche_desc=niche_desc,
                iteration=i,
                epochs=epochs_per_iter,
                max_rows=max_rows,
                skip_data_generation=skip_data_generation and i > 1,
                existing_data_path=existing_data_path if i == 1 else None,
            )

            results.append(result)

            if result.get("status") != "completed":
                print(f"Iteration {i} failed: {result.get('reason', 'unknown')}")
                break

            current_score = result.get("accuracy", 0)
            improvement = current_score - best_score
            best_score = max(best_score, current_score)

            # Check termination conditions
            if current_score >= 0.95:
                print(f"\n✓ Accuracy reached 95%+! Stopping loop.")
                break

            if improvement < improvement_threshold:
                no_improvement_count += 1
                print(f"\n  Minor improvement ({improvement:.1%}) — "
                      f"{no_improvement_count}/{patience}")
                if no_improvement_count >= patience:
                    print(f"\n  No significant improvement for {patience} iterations. Stopping.")
                    break
            else:
                no_improvement_count = 0

            # Between iterations: the gap analysis feeds into the next data generation
            # In a full implementation, we'd use eval failures to craft a better BigSet prompt
            if i < max_iterations:
                niche_desc = self._refine_description(niche_desc, results)

        # Final summary
        self._print_final_summary(niche_name, baseline_model, results)
        return results

    def _refine_description(self, original_desc: str, results: list) -> str:
        """Refine the BigSet description based on previous iteration results."""
        # In v1, just append "with verified sources" to encourage better data
        # In future versions, this would use eval failure analysis
        return original_desc

    def _print_final_summary(self, niche_name: str, base_model: str, results: list):
        """Print the final summary across all iterations."""
        completed = [r for r in results if r.get("status") == "completed"]
        if not completed:
            return

        first_acc = completed[0].get("accuracy", 0)
        last_acc = completed[-1].get("accuracy", 0)
        total_improvement = last_acc - first_acc

        print(f"\n{'#'*60}")
        print(f"# RECURSIVE LOOP COMPLETE")
        print(f"{'#'*60}")
        print(f"  Niche: {niche_name}")
        print(f"  Base: {base_model}")
        print(f"  Iterations: {len(completed)}")
        print(f"  Start accuracy: {first_acc:.1%}")
        print(f"  Final accuracy: {last_acc:.1%}")
        print(f"  Total improvement: {total_improvement:+.1%}")
        print(f"\n  Models available in Ollama:")
        for r in completed:
            print(f"    - {r.get('model_name', '?')}: {r.get('accuracy', '?'):.1%}")
        print(f"{'#'*60}")

        # Print leaderboard
        self.leaderboard.print_leaderboard(niche_name)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Recursive Fine-Tuning Loop")
    parser.add_argument("--niche-name", type=str, required=True,
                        help="Short name (e.g. medical-coding)")
    parser.add_argument("--niche-desc", type=str, required=True,
                        help="Natural language description for data generation")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip BigSet data generation, use existing data only")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to existing verified jsonl data")

    args = parser.parse_args()

    loop = RecursiveLoop()
    loop.run_recursive(
        niche_name=args.niche_name,
        niche_desc=args.niche_desc,
        max_iterations=args.max_iterations,
        epochs_per_iter=args.epochs,
        max_rows=args.max_rows,
        skip_data_generation=args.skip_data,
        existing_data_path=args.data_path,
    )


if __name__ == "__main__":
    main()
