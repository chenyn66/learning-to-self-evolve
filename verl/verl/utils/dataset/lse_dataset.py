import json
import math
import os
import random
from collections import defaultdict
from typing import Any

import torch
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import Dataset

from lse.core.tree import EvolutionTree, TreeNode


class LSEDataset(Dataset):
    """LSE dataset loader compatible with verl RL rollout."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer,
        processor,
        config,
        max_samples: int = -1,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples

        if isinstance(data_files, (list, tuple, ListConfig)):
            data_paths = list(data_files)
        else:
            data_paths = [data_files]

        all_paths = []
        for path in data_paths:
            if not os.path.isdir(path):
                raise ValueError(f"Dataset path is not a directory: {path}")
            for d in os.listdir(path):
                full_path = os.path.join(path, d)
                if os.path.isdir(full_path):
                    all_paths.append(full_path)

        trees = {path: EvolutionTree.load_tree(os.path.join(path, "tree")) for path in all_paths}
        envs = {path: json.load(open(os.path.join(path, "env", "info.json"))) for path in all_paths}
        configs = {path: OmegaConf.load(os.path.join(path, "env", "config.yaml")) for path in all_paths}

        db2nodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen_prompts = set()

        for name, tree in trees.items():
            tree.calculate_improve_potential()
            for node in tree.node_list:
                if not self._node_is_usable_for_training(node):
                    continue

                env_info = envs[name]
                if "db_id" in env_info:
                    data_source = "bird"
                    db_name = env_info["db_id"]
                elif "subject" in env_info:
                    data_source = "mmlu"
                    db_name = env_info["subject"]
                else:
                    raise ValueError(f"Invalid env info: {env_info}")

                messages = node.self_evolve_prompt
                if not messages:
                    continue

                prompt_key = (
                    messages[0].get("content")
                    if isinstance(messages, list) and isinstance(messages[0], dict)
                    else str(messages)
                )
                if prompt_key in seen_prompts:
                    continue
                seen_prompts.add(prompt_key)

                extra_info = {
                    "depth": int(node.depth),
                    "visits": int(node.visits),
                    "performance": float(node.performance),
                    "round_idx": (-1 if node.round_idx is None else int(node.round_idx)),
                    "id": int(node.id),
                    "test_env": env_info,
                    "db_id": db_name,
                    "improve_potential": float(node.improve_potential),
                    "config": OmegaConf.to_container(configs[name], resolve=True),
                }

                db2nodes[db_name].append(
                    {
                        "messages": messages,
                        "extra_info": extra_info,
                        "data_source": data_source,
                    }
                )

        for _, data_list in db2nodes.items():
            random.shuffle(data_list)

        self.data = sum(db2nodes.values(), [])
        self.data = [
            data
            for data in self.data
            if data["extra_info"]["improve_potential"] >= self.config.min_improve_potential
        ]

        ordered_by = self.config.get("ordered_by", "random")
        if ordered_by == "random":
            random.shuffle(self.data)
        elif ordered_by == "improve_potential":
            self.data.sort(key=lambda x: x["extra_info"]["improve_potential"], reverse=True)
        elif ordered_by == "reverse_potential":
            self.data.sort(key=lambda x: x["extra_info"]["improve_potential"], reverse=False)
        else:
            raise ValueError(f"Invalid ordered_by: {ordered_by}")

        if self.max_samples > 0 and len(self.data) > self.max_samples:
            if self.config.get("shuffle", False):
                seed = self.config.get("seed")
                rng = random.Random(seed)
                indices = rng.sample(range(len(self.data)), k=self.max_samples)
                self.data = [self.data[i] for i in indices]
            else:
                self.data = self.data[: self.max_samples]

        for idx, item in enumerate(self.data):
            item["uid"] = self._make_uid(item, idx)
            item["extra_info"]["index"] = idx

    @staticmethod
    def _make_uid(item: dict[str, Any], idx: int) -> str:
        extra_info = item.get("extra_info", {})
        db_id = extra_info.get("db_id", "unknown")
        node_id = extra_info.get("id", idx)
        return f"{item.get('data_source', 'unknown')}:{db_id}:{node_id}"

    @staticmethod
    def _node_is_usable_for_training(node: TreeNode) -> bool:
        if math.isnan(node.performance):
            return False
        if node.self_evolve_prompt is None:
            return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            # `raw_prompt` is a list of chat messages (OpenAI-style dicts).
            # We intentionally do NOT apply `tokenizer.apply_chat_template(...)` here.
            # AgentLoop does that right before generation so it can inject tool schemas / multimodal inputs.
            "raw_prompt": ex["messages"],
            "extra_info": ex["extra_info"],
            "data_source": ex["data_source"],
            "reward_model": {"ground_truth": None, "style": "function"},
            "uid": ex["uid"],
            "index": ex["extra_info"]["index"],
            # Ensure DataProto has at least one tensor field.
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }
