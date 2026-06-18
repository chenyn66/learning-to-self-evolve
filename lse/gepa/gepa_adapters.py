"""GEPA adapters for running baselines on LSE tasks.

We integrate vendored GEPA (./gepa/) via its `GEPAAdapter` interface:
`gepa/src/gepa/core/adapter.py`.

Baseline contract:
- Optimize **instructions only** (candidate is {"instructions": "..."}).
- Use `n_sims` as minibatch size (handled by GEPA's batch sampler).
- Use `task.hold_out_test_size` items as the valset / D_pareto.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

from .gepa_import import ensure_gepa_on_path

ensure_gepa_on_path()

from gepa.core.adapter import EvaluationBatch  # noqa: E402

from lse.envs.bird_env import BirdEnv  # noqa: E402
from lse.envs.mmlu_env import MMLUEnv  # noqa: E402


def _truncate(text: str, max_chars: int = 800) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + " ... (truncated)"


@dataclass
class BirdGEPAAdapter:
    """GEPAAdapter for BIRD text-to-SQL using existing LSE agent/env logic."""

    args: Any
    agent: Any  # BirdAgent
    # IMPORTANT: GEPA's reflective proposer expects adapters to have a `propose_new_texts`
    # attribute (even if None). If missing, GEPA will raise AttributeError.
    propose_new_texts = None

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        # Defensive copies to avoid accidental mutation by downstream code.
        items = [copy.deepcopy(x) for x in batch]

        # Candidate -> instructions
        instr = str(candidate.get("instructions", "")).strip()
        self.agent.instructions = instr

        # Determine db_id for this batch (must be homogeneous)
        db_ids = {it.get("db_id") for it in items}
        if len(db_ids) != 1:
            raise ValueError(f"BIRD GEPAAdapter expects a single db_id per batch, got {sorted(db_ids)}")
        (db_id,) = tuple(db_ids)

        # Ensure agent prompt matches schema + new instructions
        self.agent.reset_schema(db_id=db_id)
        self.agent.update_agent_prompt(schema=self.agent.schema, instructions=self.agent.instructions)

        # Build envs + problems
        envs: List[BirdEnv] = [BirdEnv(self.args, item=it) for it in items]
        for e in envs:
            e.reset()
        problems = [e.get_problem() for e in envs]

        # Run agent
        self.agent.reset(n_sims=len(problems))
        out = self.agent.act(problems, n_samples=1)
        predictions_batch = out["selected"]  # List[List[str]]

        # Evaluate each env
        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for i, (it, env, preds) in enumerate(zip(items, envs, predictions_batch, strict=False)):
            env.evaluate(preds)
            summ = env.get_summary()[0]
            score = float(summ.get("accuracy", 0.0))
            scores.append(score)

            raw_response = ""
            try:
                raw_response = self.agent.history[i][-1]["content"]
            except Exception:
                raw_response = ""

            outputs.append(
                {
                    "task_id": summ.get("task_id"),
                    "db_id": summ.get("db_id", it.get("db_id")),
                    "question_id": it.get("question_id"),
                    "question": summ.get("test_inputs", it.get("full_question", "")),
                    "pred_sql": summ.get("pred_outputs", ""),
                    "gold_sql": summ.get("gold_outputs", it.get("SQL", "")),
                    "accuracy": score,
                    "error": summ.get("error", ""),
                    "raw_response": raw_response,
                }
            )

            if trajectories is not None:
                trajectories.append(
                    {
                        "input": {
                            "db_id": it.get("db_id"),
                            "question_id": it.get("question_id"),
                            "question": it.get("full_question", ""),
                        },
                        "output": {
                            "pred_sql": summ.get("pred_outputs", ""),
                            "raw_response": raw_response,
                        },
                        "feedback": {
                            "gold_sql": summ.get("gold_outputs", it.get("SQL", "")),
                            "accuracy": score,
                            "error": summ.get("error", ""),
                        },
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        # We only support optimizing instructions.
        if "instructions" not in components_to_update:
            return {}

        records: list[dict[str, Any]] = []
        for out, score in zip(eval_batch.outputs, eval_batch.scores, strict=False):
            ok = bool(score >= 1.0)
            feedback = [
                f"Correct: {ok}",
                f"Gold SQL: {_truncate(out.get('gold_sql', ''), 400)}",
                f"Pred SQL: {_truncate(out.get('pred_sql', ''), 400)}",
            ]
            err = out.get("error", "")
            if err:
                feedback.append(f"Error: {_truncate(err, 600)}")
            records.append(
                {
                    "Inputs": {
                        "db_id": out.get("db_id"),
                        "question_id": out.get("question_id"),
                        "question": _truncate(out.get("question", ""), 800),
                    },
                    "Generated Outputs": {
                        "answer": _truncate(out.get("pred_sql", ""), 800),
                        "raw_response": _truncate(out.get("raw_response", ""), 1200),
                    },
                    "Feedback": "\n".join(feedback),
                }
            )

        return {"instructions": records}


@dataclass
class MMLUGEPAAdapter:
    """GEPAAdapter for MMLU using existing LSE agent/env logic.

    DataInst format:
      {"item": <original dataset dict>, "subject": <str>}
    """

    args: Any
    agent: Any  # MMLUAgent
    # IMPORTANT: GEPA's reflective proposer expects adapters to have a `propose_new_texts`
    # attribute (even if None). If missing, GEPA will raise AttributeError.
    propose_new_texts = None

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        insts = [copy.deepcopy(x) for x in batch]

        instr = str(candidate.get("instructions", "")).strip()
        self.agent.instructions = instr
        self.agent.update_agent_prompt(instructions=self.agent.instructions)

        envs: list[MMLUEnv] = []
        for inst in insts:
            item = inst.get("item", inst)
            subject = inst.get("subject") or self.args.task.subject or "unknown"
            env = MMLUEnv(self.args, item=item, subject=str(subject))
            env.reset()
            envs.append(env)

        problems = [e.get_problem() for e in envs]

        self.agent.reset(n_sims=len(problems))
        out = self.agent.act(problems, n_samples=1)
        predictions_batch = out["selected"]

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for i, (inst, env, preds) in enumerate(zip(insts, envs, predictions_batch, strict=False)):
            env.evaluate(preds)
            summ = env.get_summary()[0]
            score = float(summ.get("accuracy", 0.0))
            scores.append(score)

            raw_response = ""
            try:
                raw_response = self.agent.history[i][-1]["content"]
            except Exception:
                raw_response = ""

            item = inst.get("item", inst)
            subject = inst.get("subject") or summ.get("subject")

            outputs.append(
                {
                    "task_id": summ.get("task_id"),
                    "subject": subject,
                    "question": summ.get("question", item.get("question", "")),
                    "choices": summ.get("choices", item.get("choices", [])),
                    "pred": summ.get("pred_outputs", ""),
                    "gold": summ.get("gold_outputs", ""),
                    "accuracy": score,
                    "error": summ.get("error", ""),
                    "error_type": item.get("error_type", ""),
                    "potential_reason": item.get("potential_reason", ""),
                    "raw_response": raw_response,
                }
            )

            if trajectories is not None:
                trajectories.append(
                    {
                        "input": {
                            "subject": subject,
                            "question": item.get("question", ""),
                            "choices": item.get("choices", []),
                        },
                        "output": {
                            "pred": summ.get("pred_outputs", ""),
                            "raw_response": raw_response,
                        },
                        "feedback": {
                            "gold": summ.get("gold_outputs", ""),
                            "accuracy": score,
                            "error": summ.get("error", ""),
                            "error_type": item.get("error_type", ""),
                            "potential_reason": item.get("potential_reason", ""),
                        },
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if "instructions" not in components_to_update:
            return {}

        records: list[dict[str, Any]] = []
        for out, score in zip(eval_batch.outputs, eval_batch.scores, strict=False):
            ok = bool(score >= 1.0)
            feedback_lines = [
                f"Correct: {ok}",
                f"Gold: {out.get('gold', '')}",
                f"Pred: {out.get('pred', '')}",
            ]
            if out.get("error_type"):
                feedback_lines.append(f"Error type: {out.get('error_type')}")
            if out.get("potential_reason"):
                feedback_lines.append(f"Potential reason: {_truncate(out.get('potential_reason'), 400)}")
            if out.get("error"):
                feedback_lines.append(f"Error: {_truncate(out.get('error'), 500)}")

            records.append(
                {
                    "Inputs": {
                        "subject": out.get("subject"),
                        "question": _truncate(out.get("question", ""), 1200),
                        "choices": out.get("choices", []),
                    },
                    "Generated Outputs": {
                        "answer": out.get("pred", ""),
                        "raw_response": _truncate(out.get("raw_response", ""), 1200),
                    },
                    "Feedback": "\n".join(feedback_lines),
                }
            )

        return {"instructions": records}


__all__ = ["BirdGEPAAdapter", "MMLUGEPAAdapter"]

