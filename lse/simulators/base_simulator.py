from __future__ import annotations

import random
from typing import Optional, List, Dict, Any

import numpy as np


class BaseSimulator:
    """Shared helpers for task simulators (MMLU/BIRD)."""

    def __init__(self, args):
        self.args = args
        self.seed = int(getattr(args, "seed", 0) or 0)

        # Holdout/CV config (tasks that don't use these can ignore by not inheriting).
        self.dev_size = int(getattr(args.task, "dev_size", -1) or -1)
        self.test_size = int(getattr(args.task, "test_size", -1) or -1)
        self.k_fold = int(getattr(args.task, "k_fold", 1) or 1)
        if self.k_fold < 1:
            raise ValueError("task.k_fold must be >= 1")
        self.cv_enabled = self.k_fold > 1

        if self.test_size >= 1 and self.dev_size < 1:
            raise ValueError("task.test_size > 0 requires task.dev_size > 0")
        if self.cv_enabled:
            if self.dev_size < 1:
                raise ValueError("k-fold CV requires task.dev_size > 0")
            if self.test_size != -1:
                raise ValueError("k-fold CV requires task.test_size == -1")
            if self.dev_size % self.k_fold != 0:
                raise ValueError("k-fold CV requires task.dev_size % task.k_fold == 0")

        self._dev_per_example_acc_by_round: Optional[List[Optional[List[float]]]] = None
        self._textgrad_best_dev_per_example_acc: Optional[List[float]] = None
        if self.cv_enabled:
            n_round = int(getattr(self.args, "n_round", 0) or 0)
            self._dev_per_example_acc_by_round = [None] * n_round

    def _log_holdout_metric(self, *, prefix: str, round_idx: int, avg_acc: float, best_attr: str):
        if avg_acc > getattr(self, best_attr):
            setattr(self, best_attr, avg_acc)
        self.logger.log(
            {
                f"{prefix}/avg_acc": avg_acc,
                f"{prefix}/best_acc": getattr(self, best_attr),
            },
            step=round_idx,
        )

    def _maybe_log_legacy_test_alias(self, *, round_idx: int, avg_dev_acc: Optional[float]):
        # Backward compatibility: when test_size is not enabled, keep legacy test/* as alias of dev/*.
        if self.test_envs is None and avg_dev_acc is not None:
            self._log_holdout_metric(
                prefix="test",
                round_idx=round_idx,
                avg_acc=avg_dev_acc,
                best_attr="best_test_acc",
            )

    def _log_round(self, summary, round_idx: int):
        avg_acc = sum(s["accuracy"] for s in summary) / len(summary)
        if avg_acc > self.best_acc:
            self.best_acc = avg_acc
        metrics = {
            "average_accuracy": avg_acc,
            "best_acc": self.best_acc,
        }
        if self.args.use_ref:
            ref_acc = sum(s.get("ref_accuracy", 0.0) for s in summary) / len(summary)
            ref_delta = sum(s.get("ref_delta", 0.0) for s in summary) / len(summary)
            if ref_delta > getattr(self, "best_delta", float("-inf")):
                self.best_delta = ref_delta
            metrics["ref_acc"] = ref_acc
            metrics["ref_delta"] = ref_delta
            metrics["best_delta"] = self.best_delta
        self.logger.log(metrics, step=round_idx)
        return avg_acc

    def _maybe_log_kfold_cv(self) -> None:
        if not self.cv_enabled:
            return
        if self._dev_per_example_acc_by_round is None:
            return
        if not self._dev_per_example_acc_by_round:
            return
        if any(v is None for v in self._dev_per_example_acc_by_round):
            raise RuntimeError("Missing per-round dev per-example accuracies for k-fold CV")

        dev_acc_by_round = [
            np.asarray(v, dtype=np.float32) for v in self._dev_per_example_acc_by_round  # type: ignore[arg-type]
        ]
        dev_size = int(dev_acc_by_round[0].shape[0])

        cv_seed = int(self.seed)
        indices = list(range(dev_size))
        rng = random.Random(cv_seed)
        rng.shuffle(indices)
        fold_size = dev_size // self.k_fold
        folds = []
        for fold_id in range(self.k_fold):
            folds.append(indices[fold_id * fold_size : (fold_id + 1) * fold_size])

        metrics: Dict[str, Any] = dict()
        metrics["cv/enabled"] = 1
        metrics["cv/k_fold"] = int(self.k_fold)
        metrics["cv/fold_size"] = int(fold_size)
        metrics["cv/seed"] = int(cv_seed)

        val_accs = []
        for fold_id, val_idx in enumerate(folds):
            val_set = set(val_idx)
            train_idx = [i for i in range(dev_size) if i not in val_set]

            best_round = 0
            best_train_acc = float("-inf")
            for round_idx, vec in enumerate(dev_acc_by_round):
                train_acc = float(np.mean(vec[train_idx]))
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                    best_round = round_idx

            val_acc = float(np.mean(dev_acc_by_round[best_round][val_idx]))
            val_accs.append(val_acc)

            metrics["cv/fold_" + str(fold_id) + "/best_round"] = int(best_round)
            metrics["cv/fold_" + str(fold_id) + "/train_acc"] = float(best_train_acc)
            metrics["cv/fold_" + str(fold_id) + "/val_acc"] = float(val_acc)

        metrics["cv/val_acc_mean"] = float(np.mean(val_accs))
        metrics["cv/val_acc_std"] = float(np.std(val_accs))

        step = int(getattr(self.args, "n_round", 0) or 0)
        self.logger.log(metrics, step=step)

