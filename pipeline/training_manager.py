"""
Training Manager — orchestrates training subprocess, SSE streaming, and history.

Manages the lifecycle:
  idle → starting → loading_model → training → complete/error/stopped
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from pipeline.train_qlora import prepare_training_data
from pipeline.export_gguf import export_model
from pipeline.logging_util import training_log_path


def mlx_available() -> bool:
    """True only on Apple Silicon with a working MLX/Metal backend.

    On Linux mlx isn't installed, so the import raises ImportError (caught below)
    and this returns False — training/export dispatch then falls back to the
    HuggingFace CPU backend.
    """
    try:
        import mlx.core as mx
        return bool(mx.metal.is_available())
    except Exception:
        return False


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


class TrainingManager:
    """Manages a single training job with subprocess isolation and SSE streaming."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.process: Optional[subprocess.Popen] = None
        self.stop_file: Optional[str] = None
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.current_run: Optional[dict] = None
        self.history_path = os.path.join(
            self.config.get("paths", {}).get("data", "data"),
            "training_history.json",
        )
        self._running = False
        self._event_listener_task: Optional[asyncio.Task] = None
        self._worker_log_fh = None

    @property
    def is_training(self) -> bool:
        return self._running and self.process is not None and self.process.poll() is None

    def _load_history(self) -> list:
        if os.path.exists(self.history_path):
            with open(self.history_path) as f:
                return json.load(f)
        return []

    def _save_history(self, history: list):
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)

    async def start_training(self, params: dict) -> dict:
        """Start a training job in a subprocess. Returns the run info."""
        if self.is_training:
            return {"error": "Training already in progress"}

        run_id = str(uuid.uuid4())[:8]
        niche = params.get("niche", f"run-{run_id}")
        self.stop_file = os.path.join(tempfile.gettempdir(), f"train_stop_{run_id}")

        # Clean up any old stop file
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)

        # Build paths
        data_dir = params.get("data_dir") or os.path.join(
            "data", niche, f"run_{run_id}"
        )
        adapter_path = params.get("adapter_path") or os.path.join(
            "models", "adapters", niche
        )

        self.current_run = {
            "run_id": run_id,
            "niche": niche,
            "status": "starting",
            "phase": "initializing",
            "progress_percent": 0,
            "current_step": 0,
            "total_steps": 0,
            "loss": None,
            "learning_rate": None,
            "epoch": 0,
            "elapsed_seconds": 0,
            "eta_seconds": 0,
            "message": "Starting...",
            "final_loss": None,
            "output_dir": adapter_path,
            "duration_seconds": None,
            "num_train_examples": 0,
            "started_at": datetime.now().isoformat(),
            "params": params,
            "loss_history": [],
            "log_path": training_log_path(niche, run_id),
        }

        # Prepare training data if needed
        if params.get("dataset_type") == "bigset":
            # We'd call BigSet here — placeholder
            pass

        verified_path = params.get("verified_data_path")
        if verified_path and os.path.exists(verified_path):
            prepare_training_data(
                verified_path=verified_path,
                output_dir=data_dir,
                test_split=self.config.get("eval", {}).get("test_split", 0.2),
                max_examples=params.get("max_rows"),
            )

        # Build worker config
        training_cfg = self.config.get("training", {})
        worker_config = {
            "niche": niche,
            "data_dir": data_dir,
            "adapter_path": adapter_path,
            "stop_file": self.stop_file,
            "base_model": params.get("base_model") or self.config.get("base_model"),
            "lora_rank": params.get("lora_rank") or training_cfg.get("lora_rank", 16),
            "lora_alpha": params.get("lora_alpha") or training_cfg.get("lora_alpha", 32),
            "learning_rate": params.get("learning_rate") or training_cfg.get("learning_rate", 1e-4),
            "batch_size": params.get("batch_size") or training_cfg.get("batch_size", 4),
            "epochs": params.get("epochs") or training_cfg.get("epochs", 3),
            "max_seq_length": params.get("max_seq_length") or training_cfg.get("max_seq_length", 2048),
            "grad_checkpoint": params.get("grad_checkpoint", True),
            "max_rows": params.get("max_rows", 50),
        }

        # Spawn worker subprocess
        venv_python = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".venv", "bin", "python",
        )
        worker_name = "training_worker.py" if mlx_available() else "training_worker_hf.py"
        worker_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            worker_name,
        )

        try:
            # Stream the worker's stderr (tracebacks, transformers/MLX logs) straight
            # to the per-run log file; events on stdout are still read for SSE and are
            # mirrored into the same file below. Previously stderr=PIPE was never read,
            # so worker errors were lost.
            self._worker_log_fh = open(self.current_run["log_path"], "w", buffering=1)
            self._worker_log_fh.write(
                f"=== training run {run_id} ({niche}) — worker: {worker_name} ===\n"
                f"=== config: {json.dumps(worker_config)} ===\n"
            )
            self.process = subprocess.Popen(
                [venv_python, worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._worker_log_fh,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )

            # Send config via stdin
            self.process.stdin.write(json.dumps(worker_config) + "\n")
            self.process.stdin.flush()
            self.process.stdin.close()

        except Exception as e:
            self.current_run["status"] = "error"
            self.current_run["message"] = f"Failed to start worker: {e}"
            return self.current_run

        # Start background event listener
        self._running = True
        self._event_listener_task = asyncio.create_task(
            self._listen_to_worker()
        )

        self.current_run["status"] = "running"
        return self.current_run

    async def _listen_to_worker(self):
        """Background task: read JSON events from worker stdout, put on asyncio queue."""
        loop = asyncio.get_event_loop()

        try:
            while self.process and self.process.poll() is None:
                # Read line from stdout in a thread to not block
                line = await loop.run_in_executor(
                    None, self._read_line_with_timeout, self.process.stdout, 0.5
                )
                if line is None:
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Mirror the event into the per-run log alongside the worker's stderr.
                if self._worker_log_fh:
                    try:
                        self._worker_log_fh.write(line + "\n")
                    except Exception:
                        pass

                event_type = event.get("event", "")

                if event_type == "status":
                    self.current_run["phase"] = event.get("phase", "")
                    self.current_run["message"] = event.get("message", "")
                    if "total_steps" in event:
                        self.current_run["total_steps"] = event["total_steps"]

                elif event_type == "progress":
                    self.current_run["current_step"] = event.get("step", 0)
                    self.current_run["total_steps"] = event.get("total_steps", 0)
                    self.current_run["loss"] = event.get("loss")
                    self.current_run["learning_rate"] = event.get("lr")
                    self.current_run["epoch"] = event.get("epoch", 0)
                    self.current_run["progress_percent"] = event.get("progress_percent", 0)
                    self.current_run["elapsed_seconds"] = event.get("elapsed_seconds", 0)
                    self.current_run["eta_seconds"] = event.get("eta_seconds", 0)
                    self.current_run["tokens_per_sec"] = event.get("tokens_per_sec", 0)
                    self.current_run["grad_norm"] = event.get("grad_norm")
                    if event.get("loss") is not None:
                        self.current_run["loss_history"].append({
                            "step": event.get("step", 0),
                            "loss": event.get("loss"),
                        })

                elif event_type == "complete":
                    self.current_run["status"] = "completed"
                    self.current_run["phase"] = "completed"
                    self.current_run["final_loss"] = event.get("final_loss")
                    self.current_run["duration_seconds"] = event.get("duration_seconds")
                    self.current_run["total_steps"] = event.get("total_steps")
                    self.current_run["num_train_examples"] = event.get("num_train_examples", 0)
                    self._save_to_history()

                elif event_type == "error":
                    self.current_run["status"] = "error"
                    self.current_run["phase"] = "error"
                    self.current_run["message"] = event.get("message", "Unknown error")
                    self._save_to_history()

                elif event_type == "checkpoint_saved":
                    self.current_run["checkpoint_path"] = event.get("path")

                # Put on queue for SSE consumers
                await self.event_queue.put(event)

        except Exception as e:
            self.current_run["status"] = "error"
            self.current_run["message"] = f"Event listener error: {e}"
            self._save_to_history()
        finally:
            self._running = False
            # Drain any remaining stdout
            if self.process:
                remaining = self._read_remaining(self.process.stdout)
                for line in remaining.split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            await self.event_queue.put(event)
                        except json.JSONDecodeError:
                            pass

            # Mark complete if not already
            if self.current_run["status"] == "running":
                self.current_run["status"] = "error"
                self.current_run["message"] = "Process terminated unexpectedly"
                self._save_to_history()

            # Close the per-run log file (also closes the worker's stderr target).
            if self._worker_log_fh:
                try:
                    self._worker_log_fh.write(
                        f"=== run ended: status={self.current_run['status']} ===\n"
                    )
                    self._worker_log_fh.close()
                except Exception:
                    pass
                self._worker_log_fh = None

    def _read_line_with_timeout(self, stream, timeout: float):
        """Read a line from a stream with a soft timeout (non-blocking for asyncio)."""
        import select

        if hasattr(stream, 'fileno'):
            try:
                r, _, _ = select.select([stream], [], [], timeout)
                if r:
                    return stream.readline()
            except (ValueError, TypeError, OSError):
                pass
        return None

    def _read_remaining(self, stream):
        """Read all remaining content from a stream."""
        try:
            return stream.read()
        except Exception:
            return ""

    async def stop_training(self, save_checkpoint: bool = False):
        """Stop the current training job."""
        if not self.is_training:
            return {"status": "no_training_active"}

        # Touch the stop file
        if self.stop_file:
            with open(self.stop_file, "w") as f:
                f.write("stop")

        if save_checkpoint:
            self.current_run["message"] = "Stopping and saving checkpoint..."
        else:
            self.current_run["message"] = "Stopping training..."

        # Wait for process to finish
        if self.process:
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()

        self._running = False
        self.current_run["status"] = "stopped"
        self.current_run["phase"] = "stopped"
        self._save_to_history()

        return {"status": "stopped"}

    def get_status(self) -> dict:
        """Get the current training status."""
        if self.current_run:
            return self.current_run
        return {"status": "idle"}

    def _save_to_history(self):
        """Save the current run to training history."""
        history = self._load_history()

        entry = {
            "run_id": self.current_run["run_id"],
            "niche": self.current_run["niche"],
            "status": self.current_run["status"],
            "phase": self.current_run["phase"],
            "final_loss": self.current_run.get("final_loss"),
            "duration_seconds": self.current_run.get("duration_seconds"),
            "total_steps": self.current_run.get("total_steps"),
            "num_train_examples": self.current_run.get("num_train_examples"),
            "started_at": self.current_run.get("started_at"),
            "params": self.current_run.get("params", {}),
            "output_dir": self.current_run.get("output_dir"),
            # Persist the per-step loss curve so the UI can redraw the chart after
            # a server restart (current_run is in-memory only and is lost on restart).
            "loss_history": self.current_run.get("loss_history", []),
            "log_path": self.current_run.get("log_path"),
        }

        # Add to front, limit to 50 entries
        history.insert(0, entry)
        if len(history) > 50:
            history = history[:50]

        self._save_history(history)

    def get_history(self, limit: int = 20) -> list:
        """Get training history."""
        return self._load_history()[:limit]

    async def event_stream(self):
        """Async generator for SSE events."""
        try:
            while self._running or not self.event_queue.empty():
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(), timeout=1.0
                    )
                    yield event
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    if self.current_run:
                        yield {"event": "heartbeat", "status": self.current_run.get("status")}
                    continue
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        """Clean shutdown — stop training if running."""
        if self.is_training:
            await self.stop_training()
        if self._event_listener_task:
            self._event_listener_task.cancel()
            try:
                await self._event_listener_task
            except asyncio.CancelledError:
                pass
