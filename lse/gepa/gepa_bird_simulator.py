from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List

import wandb
from omegaconf import OmegaConf

from lse.envs import BatchBird
from lse.agents import BirdAgent
from lse.wandb_utils import login_if_configured, wandb_mode

from .gepa_import import ensure_gepa_on_path
from .gepa_stoppers import MaxIterationsStopper
from .gepa_adapters import BirdGEPAAdapter
from .gepa_logging import LSEGEPARecorderCallback


class GEPABirdSimulator:
    """Run a GEPA baseline on the BIRD task (instructions-only)."""

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

        # Build fixed D_pareto holdout set on a single db_id
        dev_envs = BatchBird(args, n_sims=dev_k, resample_problem=False)
        dev_envs.reset()
        holdout_qids = [env.get_problem()["meta"].get("question_id") for env in dev_envs.envs]
        self.val_db_id = dev_envs._fixed_db_id

        # Training db_id: either same DB (ID) or a different DB (OOD)
        if bool(getattr(self.args.task, "test_ood", False)) and test_k <= 0:
            # Legacy behavior: in old single-holdout mode, test_ood means train is OOD.
            train_batch = BatchBird(args, exclude_db_id=self.val_db_id, db_id=None, resample_problem=False)
            # no need to reset; _fixed_db_id is chosen in __init__
            self.train_db_id = train_batch._fixed_db_id
            all_items = train_batch._all_items
            self.train_items = [it for it in all_items if it.get("db_id") == self.train_db_id]
        else:
            # New behavior: train/dev stay in-domain; only test holdout can be OOD.
            self.train_db_id = self.val_db_id
            all_items = dev_envs._all_items
            self.train_items = [
                it
                for it in all_items
                if it.get("db_id") == self.train_db_id and it.get("question_id") not in set(holdout_qids)
            ]

        # Val items are always from val_db_id and only the holdout qids
        all_items_val = dev_envs._all_items
        holdout_set = set(holdout_qids)
        self.val_items = [
            it for it in all_items_val if it.get("db_id") == self.val_db_id and it.get("question_id") in holdout_set
        ]

        # Optional test holdout set for reporting.
        self.test_items: list[dict[str, Any]] | None = None
        test_db_id = self.val_db_id
        test_holdout_qids: list[Any] = []
        if test_k >= 1:
            if bool(getattr(self.args.task, "test_ood", False)):
                test_envs = BatchBird(
                    args,
                    n_sims=test_k,
                    resample_problem=False,
                    db_id=None,
                    exclude_db_id=self.val_db_id,
                )
            else:
                test_envs = BatchBird(
                    args,
                    n_sims=test_k,
                    resample_problem=False,
                    db_id=self.val_db_id,
                    exclude_question_ids=holdout_qids,
                )
            test_envs.reset()
            test_db_id = test_envs._fixed_db_id
            test_holdout_qids = [env.get_problem()["meta"].get("question_id") for env in test_envs.envs]
            test_holdout_set = set(test_holdout_qids)
            self.test_items = [
                it
                for it in test_envs._all_items
                if it.get("db_id") == test_db_id and it.get("question_id") in test_holdout_set
            ]

        # Logging dirs (match existing protocol)
        os.makedirs(f"{self.args.log_dir}/{self.args.run_name}/env", exist_ok=True)

        # Arctic special-case (kept for parity with BirdSimulator)
        if "arctic" in str(self.args.model.name).lower():
            os.environ["USE_ARCTIC"] = "1"

        # Agent (task LM) — initialize on training db_id for OOD case
        self.agent = BirdAgent(args, db_id=self.train_db_id)
        # Ensure no reference comparisons inside GEPA baseline.
        self.agent.args.use_ref = False

        # Persist env metadata/config
        with open(f"{self.args.log_dir}/{self.args.run_name}/env/info.json", "w") as f:
            json.dump(
                {
                    "train_db_id": self.train_db_id,
                    "val_db_id": self.val_db_id,
                    "holdout_qids": holdout_qids,
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
                        "db_id": test_db_id,
                        "holdout_qids": test_holdout_qids,
                        "holdout_size": len(self.test_items),
                        "test_ood": bool(getattr(self.args.task, "test_ood", False)),
                    },
                    f,
                    indent=2,
                )

        with open(f"{self.args.log_dir}/{self.args.run_name}/env/config.yaml", "w") as f:
            OmegaConf.save(config=args, f=f)

        # W&B run naming: make it obvious this is GEPA
        run_name = self.args.run_name
        if self.val_db_id not in run_name:
            run_name = f"{run_name}-{self.val_db_id}"
        run_name = f"{run_name}-gepa"

        login_if_configured(self.args.wandb_key)
        self.logger = wandb.init(
            project=self.args.project_name,
            name=run_name,
            mode=wandb_mode(bool(self.args.debug)),
        )
        wandb.config.update(OmegaConf.to_container(self.args, resolve=True))

        # GEPA run dir (stores GEPA state + our JSONL metrics)
        self.gepa_dir = f"{self.args.log_dir}/{self.args.run_name}/gepa"
        # Always start GEPA from a clean run dir; never resume prior state.
        if os.path.isdir(self.gepa_dir):
            shutil.rmtree(self.gepa_dir)
        os.makedirs(self.gepa_dir, exist_ok=True)

        # Adapter + callback
        self.adapter = BirdGEPAAdapter(args=self.args, agent=self.agent)
        self.callback = LSEGEPARecorderCallback(
            out_dir=self.gepa_dir,
            wandb_run=self.logger,
            adapter=self.adapter,
            testset=self.test_items,
        )

    def _make_reflection_lm(self):
        # Reflection LM callable: prompt(str) -> completion(str)
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

        # Seed instructions come from the task prompt class
        seed_instructions = str(self.agent.prompt_class.base_instructions)
        seed_candidate = {"instructions": seed_instructions}

        # Match LSE: n_round == number of GEPA iterations
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


__all__ = ["GEPABirdSimulator"]

