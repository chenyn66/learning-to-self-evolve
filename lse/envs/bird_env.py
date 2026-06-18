from __future__ import annotations

import os
import json
import random
import copy
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor

from lse.envs.base import BaseEnv
from lse.bird.eval import eval_bird_ex
from lse.paths import bird_prompt_path


def get_cpu_quota():
    try:
        quota = int(open("/sys/fs/cgroup/cpu.max").read().split()[0])
        period = int(open("/sys/fs/cgroup/cpu.max").read().split()[1])
        if quota > 0 and period > 0:
            return quota / period
    except Exception:
        pass
    return os.cpu_count()

class BirdEnv(BaseEnv):
    """Single-problem environment for BIRD text-to-SQL.

    Initializes with exactly one sampled problem from a dataset JSON and
    evaluates the predicted SQL string using the tested eval in lse.bird.eval.
    """

    def __init__(self, args, item: Dict[str, Any]):
        self.args = copy.deepcopy(args)
        self._original_item = copy.deepcopy(item)
        self._problem: Optional[Dict[str, Any]] = None
        self._summary: List[Dict[str, Any]] = []

    def reset(self) -> None:
        item = self._original_item
        db_id = item.get("db_id", "")
        qid = item.get("question_id", -1)
        gold_sql = item.get("SQL", "")

        self._problem = {
            "task_id": f"bird_{db_id}_{qid}",
            "train": [],
            "test": [
                {"question": item.get("full_question", "")}
            ],
            "meta": {
                "db_id": db_id,
                "gold_sql": gold_sql,
                "question_id": qid,
                "difficulty": item.get("difficulty", "")
            },
        }
        self._summary = []

    def get_problem(self) -> Dict[str, Any]:
        assert self._problem is not None, "Environment not reset. Call reset() first."
        return self._problem

    def evaluate(self, predictions: List[str]) -> None:
        assert self._problem is not None, "Environment not reset. Call reset() first."
        # We expect a single prediction
        pred = predictions[0] if predictions else ""
        split = str(getattr(self.args.task, "split", "dev"))
        ok, err = eval_bird_ex(pred, self._original_item, split=split)

        self._summary = [
            {
                "task_id": self._problem["task_id"],
                "observations": self._problem["train"],
                "train_count": 0,
                "test_count": 1,
                "test_inputs": self._problem["test"][0]["question"],
                "gold_outputs": self._problem["meta"]["gold_sql"],
                "pred_outputs": pred,
                "accuracy": 1 if ok else 0,
                "db_id": self._problem["meta"]["db_id"],
                "error": err,
            }
        ]

    def get_summary(self) -> List[Dict[str, Any]]:
        return self._summary

    # ---- Cloning helpers ----
    def clone_with_same_problem(self) -> "BirdEnv":
        cloned = BirdEnv(self.args, item=self._original_item)
        cloned._problem = copy.deepcopy(self._problem) if self._problem is not None else None
        cloned._summary = []
        return cloned


class BatchBird:
    """Batched wrapper holding one BirdEnv per simulation.

    Enforces that all problems in the batch share the same db_id.
    """

    def __init__(self, args, exclude_question_ids: Optional[List[Any]] = None, exclude_db_id: Optional[str] = None, **kwargs):
        self.args = copy.deepcopy(args)
        for k, v in kwargs.items():
            if hasattr(self.args, k):
                setattr(self.args, k, v)
            elif hasattr(self.args.task, k):
                setattr(self.args.task, k, v)

        self.n_sims = self.args.n_sims
        self._rng = random.Random(getattr(self.args, "seed", 0))

        self.resample_problem = bool(self.args.task.resample_problem)

        split = str(getattr(self.args.task, "split", "dev"))
        data_root = getattr(self.args.task, "data_root", None)
        if data_root:
            data_path = Path(str(data_root)).expanduser().resolve() / f"{split}_data" / f"{split}_prompt.json"
        else:
            data_path = bird_prompt_path(split)
        assert os.path.exists(data_path), f"BIRD data file not found: {data_path}"
        with open(data_path, "r", encoding="utf-8") as f:
            self._all_items: List[Dict[str, Any]] = json.load(f)

        self.envs: List[BirdEnv] = []
        self._summary: List[Dict[str, Any]] = []
        # Persist the chosen db_id for this BatchBird if configured; else sample on first reset
        self._fixed_db_id = self.args.task.db_id

        # Use sorted db ids so sampling is deterministic across runs/processes.
        all_db_ids = sorted({item.get("db_id", "") for item in self._all_items})
        if self._fixed_db_id is None:
            if exclude_db_id:
                all_db_ids = [db_id for db_id in all_db_ids if db_id != exclude_db_id]
            if not all_db_ids:
                raise ValueError("No available db_id candidates after applying exclusions")
            self._fixed_db_id = self._rng.choice(all_db_ids)

                # Group items by db_id
        by_db = defaultdict(list)
        for it in self._all_items:
            by_db[it["db_id"]].append(it)




        self._exclude_qids: Set[Any] = set(exclude_question_ids) if exclude_question_ids else set()

        self._pool = [i for i in by_db[self._fixed_db_id] if i["question_id"] not in self._exclude_qids]

    def reset(self) -> None:
        


        assert self.n_sims <= len(self._pool), "Not enough problems in self._pool to sample from"

        if self.resample_problem:
            self.problems = self._rng.sample(self._pool, k=self.n_sims)
        elif not hasattr(self, 'problems'):
            self.problems = self._rng.sample(self._pool, k=self.n_sims)

        self.envs = [BirdEnv(self.args, item=it) for it in self.problems]
        for env in self.envs:
            env.reset()

    def get_batch(self) -> List[Dict[str, Any]]:
        return [env.get_problem() for env in self.envs]

    def evaluate(self, predictions_batch: List[List[str]]) -> None:
        assert len(predictions_batch) == len(self.envs), "Mismatched batch sizes"

        max_workers = min(len(self.envs), int(get_cpu_quota()) - 4)
        if os.environ.get("NUM_SQL_WORKERS"):
            max_workers = min(max_workers, int(os.environ.get("NUM_SQL_WORKERS")))

            

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(env.evaluate, preds) for env, preds in zip(self.envs, predictions_batch)]
                for future in futures:
                    future.result()
        else:
            for env, preds in zip(self.envs, predictions_batch):
                env.evaluate(preds)
        self._summary = [s for env in self.envs for s in env.get_summary()]

    def get_summary(self) -> List[Dict[str, Any]]:
        return self._summary


    def clone_with_same_problems(self) -> "BatchBird":
        cloned: BatchBird = object.__new__(BatchBird)
        cloned.args = self.args
        cloned.n_sims = self.n_sims
        cloned._rng = self._rng
        cloned._all_items = self._all_items
        cloned.resample_problem = self.resample_problem
        cloned.problems = self.problems
        cloned.envs = [env.clone_with_same_problem() for env in self.envs]
        cloned._summary = []
        return cloned

    @classmethod
    def create_test_envs(self, args, test_info) -> "BatchBird":
        test_envs = BatchBird(args)
        test_envs.resample_problem = False
        test_envs.n_sims = len(test_info["holdout_qids"])
        test_envs._fixed_db_id = test_info["db_id"]


        by_db = defaultdict(list)
        for it in test_envs._all_items:
            by_db[it["db_id"]].append(it)

        test_envs.problems = [i for i in by_db[test_info["db_id"]] if i["question_id"] in test_info["holdout_qids"]]

        test_envs.reset()
        return test_envs



