"""GEPA callback utilities for LSE baselines."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class LSEGEPARecorderCallback:
    """Record GEPA iteration metrics to JSONL + optional wandb.

    Writes `metrics.jsonl` into `out_dir` with one line per GEPA iteration end.
    Also writes `best_instructions.txt` whenever the best program improves on D_pareto.
    """

    out_dir: str
    wandb_run: Any | None = None
    adapter: Any | None = None
    testset: list[dict[str, Any]] | None = None
    file_name: str = "metrics.jsonl"
    max_instruction_chars: int = 50000

    # Filled during run
    best_val_history: list[float] = field(default_factory=list, init=False)
    last_best_candidate_idx: Optional[int] = field(default=None, init=False)
    last_best_val: float = field(default=float("-inf"), init=False)
    # Mirror LSE simulator logging keys:
    # - dev/*: dev-holdout (GEPA valset / D_pareto) metrics
    # - test/*: optional true test metrics; when no testset is provided, alias dev metrics
    # - test/selected: test performance at the best-dev-selected iteration so far
    best_dev_acc: float = field(default=float("-inf"), init=False)
    best_test_acc: float = field(default=float("-inf"), init=False)
    test_selected_acc: Optional[float] = field(default=None, init=False)

    _iter_cache: dict[int, dict[str, Any]] = field(default_factory=dict, init=False)
    _fh: Any | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        self._fh = open(os.path.join(self.out_dir, self.file_name), "a", encoding="utf-8")

    # ----------------------------
    # GEPA callback hooks
    # ----------------------------
    def on_optimization_start(self, event: dict[str, Any]) -> None:
        self._write({"event": "optimization_start", **event})

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        # D_pareto evaluation snapshot
        avg = _safe_float(event.get("average_score"), default=float("nan"))
        cand_idx = event.get("candidate_idx")
        is_best = bool(event.get("is_best_program", False))

        self._write({"event": "valset_evaluated", **event})

        if is_best:
            self.last_best_val = avg
            self.last_best_candidate_idx = int(cand_idx) if cand_idx is not None else None
            best_text = ""
            try:
                best_text = str(event["candidate"].get("instructions", ""))
            except Exception:
                best_text = ""
            best_text = best_text[: self.max_instruction_chars]
            with open(os.path.join(self.out_dir, "best_instructions.txt"), "w", encoding="utf-8") as f:
                f.write(best_text)

    def on_minibatch_sampled(self, event: dict[str, Any]) -> None:
        it = int(event.get("iteration", -1))
        rec = self._iter_cache.setdefault(it, {})
        rec["minibatch_ids"] = event.get("minibatch_ids")
        rec["trainset_size"] = event.get("trainset_size")

    def on_candidate_selected(self, event: dict[str, Any]) -> None:
        # Candidate selected for mutation this iteration (its score is on the full valset).
        it = int(event.get("iteration", -1))
        rec = self._iter_cache.setdefault(it, {})
        rec["selected_candidate_idx"] = event.get("candidate_idx")
        rec["selected_val_avg"] = _safe_float(event.get("score"), default=float("nan"))

    def on_evaluation_end(self, event: dict[str, Any]) -> None:
        it = int(event.get("iteration", -1))
        rec = self._iter_cache.setdefault(it, {})
        scores = list(event.get("scores") or [])

        # Two key evaluations per iteration in reflective mutation:
        # - current candidate on minibatch with traces (has_trajectories=True)
        # - proposed candidate on minibatch without traces (candidate_idx=None, has_trajectories=False)
        if bool(event.get("has_trajectories")):
            rec["minibatch_scores_before"] = scores
        elif event.get("candidate_idx") is None:
            rec["minibatch_scores_after"] = scores

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        it = int(event.get("iteration", -1))
        rec = self._iter_cache.setdefault(it, {})
        rec["proposal_accepted"] = True

    def on_candidate_rejected(self, event: dict[str, Any]) -> None:
        it = int(event.get("iteration", -1))
        rec = self._iter_cache.setdefault(it, {})
        rec["proposal_accepted"] = False
        rec["rejection_reason"] = event.get("reason")

    def on_iteration_end(self, event: dict[str, Any]) -> None:
        it = int(event.get("iteration", -1))
        state = event.get("state")
        proposal_accepted = bool(event.get("proposal_accepted", False))

        rec = self._iter_cache.setdefault(it, {})
        rec["proposal_accepted"] = proposal_accepted

        # Summaries
        before_scores = rec.get("minibatch_scores_before") or []
        after_scores = rec.get("minibatch_scores_after") or []
        rec["minibatch_sum_before"] = float(sum(before_scores)) if before_scores else None
        rec["minibatch_mean_before"] = float(sum(before_scores) / len(before_scores)) if before_scores else None
        rec["minibatch_sum_after"] = float(sum(after_scores)) if after_scores else None
        rec["minibatch_mean_after"] = float(sum(after_scores) / len(after_scores)) if after_scores else None

        # Best D_pareto score so far (full-eval policy => avg over valset ids)
        best_val = float("nan")
        best_idx = None
        try:
            scores = list(getattr(state, "program_full_scores_val_set"))
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_val = float(scores[best_idx])
        except Exception:
            pass

        rec["best_val_avg"] = best_val
        rec["best_candidate_idx"] = best_idx
        rec["total_metric_calls"] = int(getattr(state, "total_num_evals", -1)) if state is not None else None
        rec["num_candidates"] = int(len(getattr(state, "program_candidates", []))) if state is not None else None

        # LSE-compatible dev metrics:
        # Use the selected candidate's full-valset accuracy as dev/avg_acc.
        dev_avg_acc = _safe_float(rec.get("selected_val_avg"), default=float("nan"))
        if dev_avg_acc != dev_avg_acc:  # NaN
            dev_avg_acc = best_val
        rec["dev_avg_acc"] = dev_avg_acc

        dev_improved = False
        if not (dev_avg_acc != dev_avg_acc) and dev_avg_acc > self.best_dev_acc:
            self.best_dev_acc = dev_avg_acc
            dev_improved = True
        rec["dev_best_acc"] = self.best_dev_acc

        # Evaluate testset on the same selected-by-dev candidate.
        candidate_idx = rec.get("selected_candidate_idx")
        try:
            candidate_idx = int(candidate_idx) if candidate_idx is not None else None
        except Exception:
            candidate_idx = None
        if candidate_idx is None:
            candidate_idx = best_idx

        candidate_for_test = None
        try:
            candidates = list(getattr(state, "program_candidates"))
            if candidate_idx is not None and 0 <= candidate_idx < len(candidates):
                candidate_for_test = candidates[candidate_idx]
        except Exception:
            candidate_for_test = None

        if self.testset is not None and self.adapter is not None and candidate_for_test is not None:
            try:
                test_eval = self.adapter.evaluate(
                    batch=[x for x in self.testset],
                    candidate=candidate_for_test,
                    capture_traces=False,
                )
                test_scores = list(test_eval.scores or [])
                test_avg_acc = (
                    float(sum(test_scores) / len(test_scores)) if test_scores else float("nan")
                )
            except Exception:
                test_avg_acc = float("nan")
        else:
            # Legacy fallback: no dedicated testset -> alias dev metrics
            test_avg_acc = dev_avg_acc

        rec["test_avg_acc"] = test_avg_acc
        if not (test_avg_acc != test_avg_acc) and test_avg_acc > self.best_test_acc:
            self.best_test_acc = test_avg_acc
        rec["test_best_acc"] = self.best_test_acc

        if dev_improved and not (test_avg_acc != test_avg_acc):
            self.test_selected_acc = test_avg_acc
        if self.test_selected_acc is not None:
            rec["test_selected"] = self.test_selected_acc

        payload = {"event": "iteration_end", "iteration": it, **rec}
        self._write(payload)

        # Track history for simulator return value
        if not (best_val != best_val):  # not NaN
            self.best_val_history.append(best_val)
        else:
            self.best_val_history.append(self.last_best_val)

        # Optional wandb logging (match LSE style)
        if self.wandb_run is not None:
            try:
                # Align with LSE's 0-indexed `round_idx` steps:
                # GEPA's first "real" iteration_end is typically iteration=1.
                step = max(it - 1, 0)
                log_payload = {
                    # Match simulator metric keys
                    "dev/avg_acc": rec["dev_avg_acc"],
                    "dev/best_acc": rec["dev_best_acc"],
                    "test/avg_acc": rec["test_avg_acc"],
                    "test/best_acc": rec["test_best_acc"],
                    "gepa/minibatch_mean_before": rec["minibatch_mean_before"],
                    "gepa/minibatch_mean_after": rec["minibatch_mean_after"],
                    "gepa/best_val_avg": rec["best_val_avg"],
                    "gepa/proposal_accepted": int(proposal_accepted),
                    "gepa/num_candidates": rec["num_candidates"],
                    "gepa/total_metric_calls": rec["total_metric_calls"],
                }
                if "test_selected" in rec:
                    log_payload["test/selected"] = rec["test_selected"]
                self.wandb_run.log(log_payload, step=step)
            except Exception:
                pass

        # Free per-iteration cache to cap memory usage
        self._iter_cache.pop(it, None)

    def on_optimization_end(self, event: dict[str, Any]) -> None:
        # `event["final_state"]` is a GEPAState object (not JSON-serializable).
        safe_event = {k: v for k, v in event.items() if k != "final_state"}
        self._write({"event": "optimization_end", **safe_event})
        self.close()

    # ----------------------------
    # Utilities
    # ----------------------------
    def _write(self, obj: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None


__all__ = ["LSEGEPARecorderCallback"]

