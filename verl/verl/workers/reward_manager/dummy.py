# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dummy")
class DummyRewardManager(AbstractRewardManager):
    """Return a random reward in [0, 1) for each sample."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        # NOTE: This is intentionally minimal and deterministic-in-shape:
        # - Used only to validate the PPO training loop end-to-end without a real reward model.
        # - Writes reward at the last valid response token (matching verl's token-level reward convention).
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i in range(len(data)):
            data_item = data[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            if valid_response_length <= 0:
                continue

            reward = torch.rand((), device=reward_tensor.device).item()
            reward_tensor[i, valid_response_length - 1] = reward

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        return reward_tensor
