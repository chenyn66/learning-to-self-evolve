# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import hydra
import ray

from verl.trainer.main_ppo import TaskRunner as BaseTaskRunner, run_ppo
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device


class LSETaskRunner(BaseTaskRunner):
    """TaskRunner variant that allocates reward_pool without spawning RewardModelWorker."""

    def add_reward_model_worker(self, config):
        if config.reward_model.enable_resource_pool:
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            self.mapping[Role.RewardModel] = "global_pool"


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    task_runner_class = ray.remote(num_cpus=1)(LSETaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
