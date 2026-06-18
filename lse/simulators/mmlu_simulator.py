from __future__ import annotations

import copy
import hashlib
from typing import Optional, List, Dict, Any
import os
import json
import random
import numpy as np
import wandb
from omegaconf import OmegaConf

from lse.core.tree import EvolutionTree, TreeNode
from lse.envs.mmlu_env import (
    BatchMMLU,
    TRAIN_SUBJECTS,
    DEV_SUBJECTS,
    GPQA_SUBJECTS,
    GPQA_SUBFIELD_SUBJECTS,
)
from lse.agents.mmlu_agent import MMLUAgent
from lse.simulators.base_simulator import BaseSimulator
from lse.wandb_utils import login_if_configured, wandb_mode


def _stable_seed(*parts: Any) -> int:
    payload = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _subjects_for_split(split: str) -> List[str]:
    if split == "train":
        return list(TRAIN_SUBJECTS)
    if split == "dev":
        return list(DEV_SUBJECTS)
    if split == "gpqa":
        return list(GPQA_SUBJECTS)
    if split == "gpqa-subfield":
        return list(GPQA_SUBFIELD_SUBJECTS)
    raise ValueError(f"Invalid split: {split}")


def _sample_indices(pool_size: int, k: int, seed: int) -> List[int]:
    if k <= 0 or pool_size <= 0:
        return []
    rng = random.Random(seed)
    return rng.sample(list(range(pool_size)), k=min(k, pool_size))


class MMLUSimulator(BaseSimulator):
    def __init__(self, args):
        super().__init__(args)

        self.dev_envs = None
        self.test_envs = None

        # Resolve deterministic in-domain subject and its pool.
        temp_batch = BatchMMLU(args)
        train_subject = temp_batch._fixed_subject
        self._train_subject = train_subject
        train_pool_len = len(temp_batch._all_items)
        if self.cv_enabled and self.dev_size > train_pool_len:
            raise ValueError("k-fold CV requires task.dev_size <= available pool size for the sampled subject")
        dev_indices: List[int] = []
        test_indices: List[int] = []
        test_subject = train_subject

        # Dev holdout (in-domain)
        if self.dev_size >= 1:
            # Backward compatibility:
            # - Legacy single-holdout mode (test_size <= 0): use plain args.seed,
            #   matching historical runs exactly.
            # - New dual-holdout mode (test_size > 0): use isolated stable seed.
            if self.test_size <= 0:
                dev_seed = self.seed
            else:
                dev_seed = _stable_seed(
                    "mmlu",
                    self.args.task.split,
                    train_subject,
                    self.seed,
                    self.dev_size,
                    self.test_size,
                    "dev",
                )
            dev_indices = _sample_indices(train_pool_len, self.dev_size, dev_seed)
            self.dev_envs = BatchMMLU.create_test_envs(
                args,
                {"subject": train_subject, "holdout_indices": dev_indices},
            )

        # Test holdout
        if self.test_size >= 1:
            if bool(self.args.task.test_ood):
                candidate_subjects = sorted(
                    s for s in _subjects_for_split(self.args.task.split) if s != train_subject
                )
                if not candidate_subjects:
                    raise ValueError(
                        f"No OOD MMLU subject available for split={self.args.task.split} "
                        f"when in-domain subject is {train_subject}"
                    )
                test_subject_seed = _stable_seed(
                    "mmlu",
                    self.args.task.split,
                    train_subject,
                    self.seed,
                    self.dev_size,
                    self.test_size,
                    "test_subject",
                )
                test_subject = random.Random(test_subject_seed).choice(candidate_subjects)
                test_batch = BatchMMLU(args, subject=test_subject)
                test_pool_len = len(test_batch._all_items)
                test_indices_seed = _stable_seed(
                    "mmlu",
                    self.args.task.split,
                    test_subject,
                    self.seed,
                    self.dev_size,
                    self.test_size,
                    "test_indices",
                )
                test_indices = _sample_indices(test_pool_len, self.test_size, test_indices_seed)
            else:
                remaining_indices = [i for i in range(train_pool_len) if i not in set(dev_indices)]
                if not remaining_indices:
                    raise ValueError(
                        "No remaining in-domain examples for test holdout. "
                        "Decrease task.dev_size or disable task.test_size."
                    )
                test_indices_seed = _stable_seed(
                    "mmlu",
                    self.args.task.split,
                    train_subject,
                    self.seed,
                    self.dev_size,
                    self.test_size,
                    "test_indices",
                )
                rng = random.Random(test_indices_seed)
                test_indices = rng.sample(remaining_indices, k=min(self.test_size, len(remaining_indices)))
            if not test_indices:
                raise ValueError(
                    "Requested task.test_size > 0 but failed to sample any test holdout example."
                )
            self.test_envs = BatchMMLU.create_test_envs(
                args,
                {"subject": test_subject, "holdout_indices": test_indices},
            )

        # Main envs:
        # - If test_size > 0 and test_ood=true: only test holdout is OOD.
        # - If test_size <= 0 and test_ood=true: keep legacy behavior.
        if bool(self.args.task.test_ood) and self.test_size >= 1:
            self.envs = BatchMMLU(args, subject=train_subject, exclude_indices=set(dev_indices))
        elif bool(self.args.task.test_ood):
            self.envs = BatchMMLU(args, exclude_subjects=[train_subject])
        else:
            exclude_indices = set(dev_indices)
            if test_subject == train_subject:
                exclude_indices.update(test_indices)
            self.envs = BatchMMLU(args, subject=train_subject, exclude_indices=exclude_indices)

        os.makedirs(f"{self.args.log_dir}/{self.args.run_name}/env", exist_ok=True)

        self.agent = MMLUAgent(args)

        # Keep info.json schema unchanged and dev-only.
        with open(f"{self.args.log_dir}/{self.args.run_name}/env/info.json", "w") as f:
            json.dump(
                {
                    "subject": train_subject,
                    "holdout_indices": dev_indices,
                    "test_ood": self.args.task.test_ood,
                },
                f,
                indent=2,
            )

        # New test-only metadata file.
        if self.test_size >= 1:
            with open(f"{self.args.log_dir}/{self.args.run_name}/env/test_info.json", "w") as f:
                json.dump(
                    {
                        "subject": test_subject,
                        "holdout_indices": test_indices,
                        "test_ood": self.args.task.test_ood,
                    },
                    f,
                    indent=2,
                )

        with open(f"{self.args.log_dir}/{self.args.run_name}/env/config.yaml", "w") as f:
            OmegaConf.save(config=args, f=f)

        run_name = args.run_name
        if train_subject not in run_name:
            run_name = f"{run_name}-{train_subject}"

        login_if_configured(args.wandb_key)
        self.logger = wandb.init(
            project=args.project_name,
            name=run_name,
            mode=wandb_mode(bool(args.debug)),
        )
        wandb.config.update(OmegaConf.to_container(args, resolve=True))

        # Tree search state
        try:
            tree_cfg = getattr(args, "tree", None)
        except Exception:
            tree_cfg = None
        self.tree_enabled = bool(getattr(tree_cfg, "enabled", False))
        self.tree_selection = getattr(tree_cfg, "selection", "deepest")
        self.tree_ucb_c = float(getattr(tree_cfg, "ucb_c", 2.0))
        
        if self.tree_enabled:
            # Note: Base instructions are used for tree root
            self.tree = EvolutionTree(selection=self.tree_selection, ucb_c=self.tree_ucb_c, base_instructions=self.agent.instructions)
            self.current_parent_node: Optional[TreeNode] = self.tree.root
        else:
            self.tree = None
            self.current_parent_node = None
            
        self.best_acc = float("-inf")
        if self.args.use_ref:
            self.best_delta = float("-inf")

        if self.dev_envs is not None:
            self.best_dev_acc = float("-inf")
        if self.test_envs is not None or self.dev_envs is not None:
            self.best_test_acc = float("-inf")
        self.textgrad_revert_on_worse_holdout = bool(
            getattr(getattr(self.args, "textgrad", None), "revert_on_worse_holdout", True)
        )
        self.textgrad_best_holdout_acc = float("-inf")
        self.textgrad_best_instructions = self.agent.instructions
        # Test performance at the round selected by best dev performance.
        self.test_selected_acc: Optional[float] = None

    def _append_performance_delta(self, round_idx: int, delta: dict):
        evolve_chat_path = f"{self.args.log_dir}/{self.args.run_name}/round_{round_idx}/evolve_chat.txt"
        delta_json = {
            "performance_delta": {
                "accuracy_delta": round(delta["accuracy_delta"], 4),
            },
            "evaluation": {
                "round_from": delta["round_from"],
                "round_to": delta["round_to"],
                "improvement": str(delta["accuracy_delta"] > 0),
            },
        }
        try:
            with open(evolve_chat_path, "a") as f:
                f.write("\n\n[PERFORMANCE DELTA]\n")
                f.write(json.dumps(delta_json, indent=2))
        except FileNotFoundError:
            print(f"Warning: Could not append delta to {evolve_chat_path} - file not found")

    def _evaluate_on_holdout(
        self,
        holdout_envs: BatchMMLU,
        round_idx: int,
        *,
        prefix: str,
        best_attr: str,
        log: bool = True,
        details_out: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        avg_acc_list: List[float] = []
        per_example_acc_sum = None
        per_example_acc_count = 0
        orig_sample_agent = self.agent.args.model.sample_agent
        if self.args.test_n_eval > 1:
            self.agent.args.model.sample_agent = True

        # Disable reference comparison during hold-out eval
        orig_use_ref = self.args.use_ref
        self.agent.args.use_ref = False

        holdout_envs.reset()
        self.agent.reset(n_sims=holdout_envs.n_sims)

        holdout_problems = holdout_envs.get_batch()
        holdout_predictions = self.agent.act(holdout_problems, n_samples=self.args.test_n_eval)["all"]

        assert len(holdout_predictions) == self.args.test_n_eval
        for i in range(self.args.test_n_eval):
            holdout_envs.reset()
            holdout_envs.evaluate(holdout_predictions[i])
            holdout_summary = holdout_envs.get_summary()
            avg_acc_list.append(sum(s["accuracy"] for s in holdout_summary) / len(holdout_summary))
            if details_out is not None:
                if per_example_acc_sum is None:
                    per_example_acc_sum = np.zeros(len(holdout_summary), dtype=np.float32)
                per_example_acc_sum += np.array(
                    [float(s.get("accuracy", 0.0)) for s in holdout_summary],
                    dtype=np.float32,
                )
                per_example_acc_count += 1

        # Restore flags
        self.agent.args.use_ref = orig_use_ref
        self.agent.args.model.sample_agent = orig_sample_agent

        if details_out is not None and per_example_acc_sum is not None and per_example_acc_count > 0:
            details_out["per_example_acc"] = (
                per_example_acc_sum / float(per_example_acc_count)
            ).tolist()

        avg_acc = float(np.mean(avg_acc_list)) if avg_acc_list else None
        if log and avg_acc is not None:
            self._log_holdout_metric(prefix=prefix, round_idx=round_idx, avg_acc=avg_acc, best_attr=best_attr)
        return avg_acc_list

    def _apply_textgrad_holdout_selection(self, round_idx: int) -> int:
        """Evaluate candidate instruction on dev holdout and accept/revert."""
        if self.dev_envs is None:
            self.textgrad_best_instructions = self.agent.instructions
            self.logger.log({"textgrad/accepted": 1}, step=round_idx)
            return 1

        prev_best_holdout = float(self.textgrad_best_holdout_acc)

        # Evaluate candidate on dev holdout only.
        dev_details = None
        if self.cv_enabled:
            dev_details = dict()
        candidate_acc_values = self._evaluate_on_holdout(
            self.dev_envs,
            round_idx,
            prefix="dev",
            best_attr="best_dev_acc",
            log=False,
            details_out=dev_details,
        )
        candidate_acc = float(np.mean(candidate_acc_values)) if candidate_acc_values else float("-inf")
        candidate_per_example = None
        if self.cv_enabled and dev_details is not None:
            candidate_per_example = dev_details.get("per_example_acc")
        dev_improved = candidate_acc > prev_best_holdout
        accepted = 1
        if candidate_acc > self.textgrad_best_holdout_acc:
            self.textgrad_best_holdout_acc = candidate_acc
            self.textgrad_best_instructions = self.agent.instructions
            if self.cv_enabled and candidate_per_example is not None:
                self._textgrad_best_dev_per_example_acc = candidate_per_example
        else:
            accepted = 0
            if self.textgrad_revert_on_worse_holdout:
                self.agent.instructions = self.textgrad_best_instructions
                self.agent.update_agent_prompt(instructions=self.agent.instructions)
                accepted = 0

        if self.cv_enabled and self._dev_per_example_acc_by_round is not None:
            if accepted == 1:
                if candidate_per_example is not None:
                    self._dev_per_example_acc_by_round[round_idx] = candidate_per_example
            else:
                if self.textgrad_revert_on_worse_holdout and self._textgrad_best_dev_per_example_acc is not None:
                    self._dev_per_example_acc_by_round[round_idx] = list(self._textgrad_best_dev_per_example_acc)
                elif candidate_per_example is not None:
                    self._dev_per_example_acc_by_round[round_idx] = candidate_per_example

        # Log accepted (post-revert) dev holdout performance.
        accepted_acc = float(self.textgrad_best_holdout_acc)
        self._log_holdout_metric(
            prefix="dev",
            round_idx=round_idx,
            avg_acc=accepted_acc,
            best_attr="best_dev_acc",
        )

        # Evaluate and log test holdout after final (accepted/reverted) instructions.
        avg_test_acc: Optional[float] = None
        if self.test_envs is not None:
            test_acc_values = self._evaluate_on_holdout(
                self.test_envs,
                round_idx,
                prefix="test",
                best_attr="best_test_acc",
                log=False,
            )
            if test_acc_values:
                avg_test_acc = float(np.mean(test_acc_values))
                self._log_holdout_metric(
                    prefix="test",
                    round_idx=round_idx,
                    avg_acc=avg_test_acc,
                    best_attr="best_test_acc",
                )
        else:
            avg_test_acc = accepted_acc
            self._maybe_log_legacy_test_alias(round_idx=round_idx, avg_dev_acc=accepted_acc)

        # Log test performance of the best-dev-selected round.
        if dev_improved and avg_test_acc is not None:
            self.test_selected_acc = float(avg_test_acc)
        if self.test_selected_acc is not None:
            self.logger.log({"test/selected": float(self.test_selected_acc)}, step=round_idx)

        self.logger.log(
            {
                "textgrad/accepted": accepted,
                "textgrad/best_holdout_acc": self.textgrad_best_holdout_acc,
            },
            step=round_idx,
        )
        return accepted

    def run_round(self, round_idx=0):
        avg_dev_acc = None
        avg_test_acc = None
        # Evaluate holdouts each round. In TextGrad mode, dev/test are logged after evolve
        # to avoid duplicate logs.
        if self.args.evolve_mode != "textgrad":
            prev_best_dev = float(getattr(self, "best_dev_acc", float("-inf")))
            dev_improved = False
            if self.dev_envs is not None:
                dev_details = None
                if self.cv_enabled:
                    dev_details = dict()
                dev_acc_values = self._evaluate_on_holdout(
                    self.dev_envs,
                    round_idx,
                    prefix="dev",
                    best_attr="best_dev_acc",
                    log=True,
                    details_out=dev_details,
                )
                if self.cv_enabled and self._dev_per_example_acc_by_round is not None and dev_details is not None:
                    per_example = dev_details.get("per_example_acc")
                    if per_example is not None:
                        self._dev_per_example_acc_by_round[round_idx] = per_example
                if dev_acc_values:
                    avg_dev_acc = float(np.mean(dev_acc_values))
                    dev_improved = avg_dev_acc > prev_best_dev
            if self.test_envs is not None:
                test_acc_values = self._evaluate_on_holdout(
                    self.test_envs,
                    round_idx,
                    prefix="test",
                    best_attr="best_test_acc",
                    log=True,
                )
                if test_acc_values:
                    avg_test_acc = float(np.mean(test_acc_values))
            else:
                self._maybe_log_legacy_test_alias(round_idx=round_idx, avg_dev_acc=avg_dev_acc)
                if avg_dev_acc is not None:
                    avg_test_acc = avg_dev_acc

            # Log test performance of the best-dev-selected round.
            if dev_improved and avg_test_acc is not None:
                self.test_selected_acc = float(avg_test_acc)
            if self.test_selected_acc is not None:
                self.logger.log({"test/selected": float(self.test_selected_acc)}, step=round_idx)

        # Main round
        self.envs.reset()
        self.agent.reset()

        problems = self.envs.get_batch()
        predictions_batch = self.agent.act(problems)['selected']

        # If comparing with reference, predictions_batch contains [current (n), reference (n)]
        if self.args.use_ref:
            n = self.envs.n_sims
            assert len(predictions_batch) == 2 * n, (
                f"Expected 2*n predictions when use_ref=true. "
                f"Got {len(predictions_batch)} for n={n}."
            )
            sim_predictions = predictions_batch[:n]
            reference_predictions = predictions_batch[n:]
        else:
            sim_predictions = predictions_batch

        # Evaluate current policy on main envs
        self.envs.evaluate(sim_predictions)
        summary = self.envs.get_summary()

        current_histories_text = self.agent.get_history()
        avg_acc = sum(s['accuracy'] for s in summary) / len(summary)

        # Optional: compute reference accuracy using cloned envs
        if self.args.use_ref:
            cloned_batch = self.envs.clone_with_same_problems()
            assert len(reference_predictions) == len(cloned_batch.envs)
            cloned_batch.evaluate(reference_predictions)
            ref_summary = cloned_batch.get_summary()

            for sim_s, ref_s in zip(summary, ref_summary):
                sim_s["ref_accuracy"] = ref_s["accuracy"]
                sim_s["ref_delta"] = sim_s["accuracy"] - ref_s["accuracy"]

            ref_delta = sum(s['ref_delta'] for s in summary) / len(summary)

        if self.tree_enabled and self.tree is not None:
            metric_name = str(getattr(self.args.tree, "metric", "train")).lower()
            if metric_name in {"dev", "test"}:
                performance = avg_dev_acc if avg_dev_acc is not None else avg_acc
            elif self.args.use_ref:
                performance = ref_delta
            else:
                performance = avg_acc

            evolve_prompt = self.agent.build_evolve_prompt(summary)
                
            self.tree.update_node(self.current_parent_node.id,
                                  performance=performance,
                                  history=copy.deepcopy(self.agent.history),
                                  summary=summary,
                                  self_evolve_prompt=[{"role": "user", "content": evolve_prompt}])
            # TextGrad evolves from the current round rather than a newly selected node.
            if self.args.evolve_mode != "textgrad":
                selected = self.tree.select(self.tree_selection)
                if selected is not None and selected.summary is not None and selected.history is not None:
                    self.agent.instructions = selected.instructions
                    self.agent.update_agent_prompt(instructions=self.agent.instructions)
                    self.agent.history = copy.deepcopy(selected.history)
                    parent_summary = selected.summary
                    self.current_parent_node = selected
                else:
                    parent_summary = summary
            else:
                parent_summary = summary
        else:
            parent_summary = summary

        # Evolve
        if self.args.evolve_mode == "dummy":
            evolve_chat = self.agent.dummy_self_evolve(parent_summary)
        elif self.args.evolve_mode == "textgrad":
            evolve_chat = self.agent.textgrad_evolve(parent_summary)
        else:
            # Default to self evolve
            evolve_chat = self.agent.self_evolve(parent_summary)

        if self.args.evolve_mode == "textgrad":
            self._apply_textgrad_holdout_selection(round_idx)
            stats = getattr(self.agent, "_last_textgrad_stats", {}) or {}
            self.logger.log(
                {
                    "textgrad/critic_chars": int(stats.get("critic_chars", 0)),
                    "textgrad/instruction_chars": int(stats.get("instruction_chars", len(self.agent.instructions))),
                    "textgrad/n_failures_used": int(stats.get("n_failures_used", 0)),
                },
                step=round_idx,
            )

        # Add the prompt to the tree
        if self.tree_enabled and self.tree is not None:
            new_node = self.tree.add_node(
                parent=self.current_parent_node,
                instructions=self.agent.instructions,
                round_idx=round_idx+1,
                conversation=evolve_chat,
            )
            self.current_parent_node = new_node

        # Logs
        os.makedirs(f"{self.args.log_dir}/{self.args.run_name}/round_{round_idx}", exist_ok=True)
        for i, h in enumerate(current_histories_text):
            with open(
                f"{self.args.log_dir}/{self.args.run_name}/round_{round_idx}/agent_{i}.txt",
                "w",
            ) as f:
                f.write(h)

        with open(
            f"{self.args.log_dir}/{self.args.run_name}/round_{round_idx}/evolve_chat.txt",
            "w",
        ) as f:
            f.write(self.agent.format_history(evolve_chat))
            
        # Separate folders for chat/response
        os.makedirs(f"{self.args.log_dir}/{self.args.run_name}/evolve/chat", exist_ok=True)
        os.makedirs(f"{self.args.log_dir}/{self.args.run_name}/evolve/response", exist_ok=True)

        with open(
            f"{self.args.log_dir}/{self.args.run_name}/evolve/chat/round_{round_idx}.txt",
            "w",
        ) as f:
            f.write(self.agent.format_history(evolve_chat))

        with open(
            f"{self.args.log_dir}/{self.args.run_name}/evolve/response/round_{round_idx}.txt",
            "w",
        ) as f:
            f.write(evolve_chat[-1]["content"])

        return summary

    def run_self_evolve(self):
        average_accuracy_history = []
        previous_performance = None

        for i in range(self.args.n_round):
            summary = self.run_round(round_idx=i)
            avg_acc = self._log_round(summary, i)
            current_perf = {"average_accuracy": avg_acc}

            if previous_performance is not None:
                delta = {
                    "accuracy_delta": avg_acc - previous_performance["average_accuracy"],
                    "round_from": i - 1,
                    "round_to": i,
                }
                self._append_performance_delta(i - 1, delta)

            average_accuracy_history.append(avg_acc)
            previous_performance = current_perf

            if self.tree_enabled and self.tree is not None:
                try:
                    out_dir = f"{self.args.log_dir}/{self.args.run_name}/tree"
                    self.tree.save_visualizations(out_dir)
                    self.tree.save_tree(out_dir)
                except Exception as e:
                    print(f"Warning: Failed to save tree visualizations: {e}")

        self._maybe_log_kfold_cv()
        return {"average_accuracy": average_accuracy_history}

