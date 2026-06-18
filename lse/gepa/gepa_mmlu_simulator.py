from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from typing import Any, Dict, List, Optional

import wandb
from omegaconf import OmegaConf

from lse.envs.mmlu_env import (
    BatchMMLU,
    TRAIN_SUBJECTS,
    DEV_SUBJECTS,
    GPQA_SUBJECTS,
    GPQA_SUBFIELD_SUBJECTS,
)
from lse.agents.mmlu_agent import MMLUAgent
from lse.wandb_utils import login_if_configured, wandb_mode

from .gepa_import import ensure_gepa_on_path
from .gepa_stoppers import MaxIterationsStopper
from .gepa_adapters import MMLUGEPAAdapter
from .gepa_logging import LSEGEPARecorderCallback


def _stable_seed(*parts: Any) -> int:
    payload = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _subjects_for_split(split: str) -> list[str]:
    if split == "train":
        return list(TRAIN_SUBJECTS)
    if split == "dev":
        return list(DEV_SUBJECTS)
    if split == "gpqa":
        return list(GPQA_SUBJECTS)
    if split == "gpqa-subfield":
        return list(GPQA_SUBFIELD_SUBJECTS)
    raise ValueError(f"Invalid split: {split}")


class GEPAMMLUSimulator:
    """Run a GEPA baseline on the MMLU task (instructions-only)."""

    def __init__(self, args):
        self.args = args

        # Backward compatible parsing:
        # - Prefer task.dev_size when present
        # - Fall back to legacy task.hold_out_test_size
        dev_k = int(getattr(args.task, "dev_size", getattr(args.task, "hold_out_test_size", -1)) or -1)
        test_k = int(getattr(args.task, "test_size", -1) or -1)
        if dev_k < 1:
            raise ValueError(
                "GEPA baseline requires task.dev_size >= 1 "
                "(or legacy task.hold_out_test_size >= 1) "
                "(this is D_pareto)."
            )
        if test_k >= 1 and dev_k < 1:
            raise ValueError("task.test_size > 0 requires task.dev_size > 0")

        # Resolve a fixed subject + pool from the current split logic
        temp_batch = BatchMMLU(args)
        self.val_subject = temp_batch._fixed_subject
        pool = list(temp_batch._all_items)
        dataset_len = len(pool)
        if dataset_len == 0:
            raise ValueError(f"Empty MMLU pool for subject={self.val_subject}")

        seed = int(getattr(args, "seed", 0))
        rng = random.Random(int(getattr(args, "seed", 0)))
        dev_indices = rng.sample(list(range(dataset_len)), k=min(dev_k, dataset_len))
        dev_set = set(dev_indices)

        # Val set (D_pareto): fixed holdout indices from the resolved subject
        self.val_items: List[dict[str, Any]] = [{"item": pool[i], "subject": self.val_subject} for i in dev_indices]

        # Train set: either in-domain (same subject, exclude holdout) or OOD (different subject)
        if bool(getattr(self.args.task, "test_ood", False)) and test_k <= 0:
            # Legacy behavior: in old single-holdout mode, test_ood means train is OOD.
            train_batch = BatchMMLU(args, exclude_subjects=[self.val_subject])
            self.train_subject = train_batch._fixed_subject
            train_pool = list(train_batch._all_items)
            self.train_items = [{"item": it, "subject": self.train_subject} for it in train_pool]
        else:
            # New behavior: train/dev stay in-domain; only test holdout can be OOD.
            self.train_subject = self.val_subject
            self.train_items = [
                {"item": pool[i], "subject": self.train_subject}
                for i in range(dataset_len)
                if i not in dev_set
            ]

        # Optional test holdout set for reporting.
        self.test_items: Optional[List[dict[str, Any]]] = None
        test_subject = self.val_subject
        test_indices: List[int] = []
        if test_k >= 1:
            if bool(getattr(self.args.task, "test_ood", False)):
                candidate_subjects = sorted(
                    s for s in _subjects_for_split(self.args.task.split) if s != self.val_subject
                )
                if not candidate_subjects:
                    raise ValueError(
                        f"No OOD MMLU subject available for split={self.args.task.split} "
                        f"when val subject is {self.val_subject}"
                    )
                subj_rng = random.Random(
                    _stable_seed(
                        "gepa-mmlu",
                        self.args.task.split,
                        self.val_subject,
                        seed,
                        dev_k,
                        test_k,
                        "test_subject",
                    )
                )
                test_subject = subj_rng.choice(candidate_subjects)
                test_batch = BatchMMLU(args, subject=test_subject)
                test_pool = list(test_batch._all_items)
                if not test_pool:
                    raise ValueError(f"Empty MMLU test pool for subject={test_subject}")
                idx_rng = random.Random(
                    _stable_seed(
                        "gepa-mmlu",
                        self.args.task.split,
                        test_subject,
                        seed,
                        dev_k,
                        test_k,
                        "test_indices",
                    )
                )
                test_indices = idx_rng.sample(list(range(len(test_pool))), k=min(test_k, len(test_pool)))
                self.test_items = [{"item": test_pool[i], "subject": test_subject} for i in test_indices]
            else:
                remaining = [i for i in range(dataset_len) if i not in dev_set]
                if not remaining:
                    raise ValueError(
                        "No remaining in-domain examples for test holdout. "
                        "Decrease task.dev_size or disable task.test_size."
                    )
                idx_rng = random.Random(
                    _stable_seed(
                        "gepa-mmlu",
                        self.args.task.split,
                        self.val_subject,
                        seed,
                        dev_k,
                        test_k,
                        "test_indices",
                    )
                )
                test_indices = idx_rng.sample(remaining, k=min(test_k, len(remaining)))
                self.test_items = [{"item": pool[i], "subject": self.val_subject} for i in test_indices]

        # Env/config snapshots (match existing protocol)
        os.makedirs(f"{self.args.log_dir}/{self.args.run_name}/env", exist_ok=True)
        with open(f"{self.args.log_dir}/{self.args.run_name}/env/info.json", "w") as f:
            json.dump(
                {
                    "train_subject": self.train_subject,
                    "val_subject": self.val_subject,
                    "holdout_indices": dev_indices,
                    "holdout_size": len(self.val_items),
                    "test_ood": bool(getattr(self.args.task, "test_ood", False)),
                },
                f,
                indent=2,
            )
        if self.test_items is not None:
            with open(f"{self.args.log_dir}/{self.args.run_name}/env/test_info.json", "w") as f:
                json.dump(
                    {
                        "subject": test_subject,
                        "holdout_indices": test_indices,
                        "holdout_size": len(self.test_items),
                        "test_ood": bool(getattr(self.args.task, "test_ood", False)),
                    },
                    f,
                    indent=2,
                )
        with open(f"{self.args.log_dir}/{self.args.run_name}/env/config.yaml", "w") as f:
            OmegaConf.save(config=args, f=f)

        # W&B (match LSE style)
        run_name = self.args.run_name
        if self.val_subject not in run_name:
            run_name = f"{run_name}-{self.val_subject}"
        run_name = f"{run_name}-gepa"

        login_if_configured(self.args.wandb_key)
        self.logger = wandb.init(
            project=self.args.project_name,
            name=run_name,
            mode=wandb_mode(bool(self.args.debug)),
        )
        wandb.config.update(OmegaConf.to_container(self.args, resolve=True))

        # GEPA run dir
        self.gepa_dir = f"{self.args.log_dir}/{self.args.run_name}/gepa"
        # Always start GEPA from a clean run dir; never resume prior state.
        if os.path.isdir(self.gepa_dir):
            shutil.rmtree(self.gepa_dir)
        os.makedirs(self.gepa_dir, exist_ok=True)

        # Agent + adapter
        self.agent = MMLUAgent(args)
        self.agent.args.use_ref = False
        self.adapter = MMLUGEPAAdapter(args=self.args, agent=self.agent)
        self.callback = LSEGEPARecorderCallback(
            out_dir=self.gepa_dir,
            wandb_run=self.logger,
            adapter=self.adapter,
            testset=self.test_items,
        )

    def _make_reflection_lm(self):
        from vllm import SamplingParams

        sampling = self.args.model.sampling
        params = SamplingParams(
            temperature=float(getattr(sampling, "temperature", 0.7)),
            top_p=float(getattr(sampling, "top_p", 0.8)),
            top_k=int(getattr(sampling, "top_k", 20)),
            min_p=float(getattr(sampling, "min_p", 0.0)),
            presence_penalty=float(getattr(sampling, "presence_penalty", 0.5)),
            max_tokens=int(getattr(sampling, "max_tokens", 4096)),
        )

        max_tries = int(getattr(self.args, "max_tries", 3))
        no_think = bool(getattr(self.args, "no_think", False))

        def lm(prompt: str) -> str:
            self.agent.mode = "self_evolve"
            out = self.agent._batch_chat(
                [[{"role": "user", "content": prompt}]],
                params,
                checker=None,
                max_tries=max_tries,
                no_think=no_think,
                n_samples=1,
            )
            return out[0][0]

        return lm

    def run_gepa(self) -> Dict[str, Any]:
        ensure_gepa_on_path()
        from gepa.api import optimize

        reflection_lm = self._make_reflection_lm()
        seed_candidate = {"instructions": str(self.agent.prompt_class.base_instructions)}
        stopper = MaxIterationsStopper(int(self.args.n_round))

        _ = optimize(
            seed_candidate=seed_candidate,
            trainset=self.train_items,
            valset=self.val_items,
            adapter=self.adapter,
            reflection_lm=reflection_lm,
            callbacks=[self.callback],
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            batch_sampler="epoch_shuffled",
            reflection_minibatch_size=int(self.args.n_sims),
            stop_callbacks=stopper,
            run_dir=self.gepa_dir,
            use_wandb=False,
            display_progress_bar=False,
            seed=int(getattr(self.args, "seed", 0)),
            raise_on_exception=True,
            cache_evaluation=False,
            track_best_outputs=False,
        )

        return {"average_accuracy": list(self.callback.best_val_history)}


__all__ = ["GEPAMMLUSimulator"]

