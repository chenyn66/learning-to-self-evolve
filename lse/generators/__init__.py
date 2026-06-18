from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
import copy

from lse.core.tree import EvolutionTree, TreeNode
from lse.envs.bird_env import BatchBird, BirdEnv
from lse.agents.bird_agent import BirdAgent


class RLGenerator:
    """On-policy data generator for the inner loop of RL over prompts.

    Maintains a task environment batch, an agent, and an evolution tree of
    prompts. Exposes two main methods:
      - sample(k): select k prompts from the tree, run on-policy rollouts H,
        and return per-rollout handles with avg reward.
      - update(rollout_id, new_prompt, batch_option): evaluate a new prompt
        against either the parent rollout batch, a fresh batch, or an explicit
        batch, compute the reward delta, and add a new node to the tree.
    """

    def __init__(self, args):
        self.args = args

        # Initialize batched environment and agent
        self.envs = BatchBird(args)
        self.agent = BirdAgent(args, db_id=self.envs._fixed_db_id)
        # Ensure pure on-policy reward in inner loop
        self.agent.args.use_ref = False

        # Initialize evolution tree
        tree_cfg = getattr(args, "tree", None)
        selection = getattr(tree_cfg, "selection", "deepest") if tree_cfg is not None else "deepest"
        ucb_c = float(getattr(tree_cfg, "ucb_c", 2.0)) if tree_cfg is not None else 2.0
        self.tree = EvolutionTree(selection=selection, ucb_c=ucb_c, base_instructions=self.agent.instructions)

        # Rollout registry for delta computation and reproducible A/B
        self._next_rollout_id: int = 1
        self.rollouts: Dict[int, Dict[str, Any]] = {}

    # ----------------------------
    # Public API
    # ----------------------------
    def sample(self, k: int, selection: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run k on-policy rollouts using prompts selected from the tree.

        Args:
            k: Number of rollouts to perform (one rollout per selected node).
            selection: Optional selection strategy override (deepest|best|ucb).

        Returns:
            List of dict payloads, each containing:
              - rollout_id: int
              - node_id: int (the selected node)
              - prompt: str (instructions used)
              - H: List[dict] (environment summary per sim)
              - histories: List[str] (stringified chat histories)
              - avg_reward: float (mean accuracy over sims)
        """
        results: List[Dict[str, Any]] = []
        chosen_ids: set[int] = set()

        for _ in range(max(0, int(k))):
            node = self._select_node(selection)
            # Best-effort dedupe, fallback to reuse if cannot dedupe
            attempts = 0
            while node is not None and node.id in chosen_ids and attempts < 5:
                node = self._select_node(selection)
                attempts += 1
            if node is None:
                break
            chosen_ids.add(node.id)

            # Configure agent with the node's instructions
            self._set_agent_instructions(node.instructions)

            # Run a single inner-loop rollout on a fresh batch
            self.envs.reset()
            self.agent.reset(n_sims=self.envs.n_sims)
            problems = self.envs.get_batch()
            predictions = self.agent.act(problems)
            self.envs.evaluate(predictions)
            summary = self.envs.get_summary()
            avg_reward = self._mean_accuracy(summary)

            # Update node statistics
            self.tree.update_node(
                node,
                performance=avg_reward,
                summary=copy.deepcopy(summary),
                history=copy.deepcopy(self.agent.history),
            )

            # Persist rollout with an environment clone for exact A/B reuse
            rollout_id = self._alloc_rollout_id()
            self.rollouts[rollout_id] = {
                "parent_node_id": node.id,
                "prompt": node.instructions,
                "avg_reward": avg_reward,
                "summary": copy.deepcopy(summary),
                "histories": self.agent.get_history(),
                "env_clone": self.envs.clone_with_same_problems(),
            }

            results.append(
                {
                    "rollout_id": rollout_id,
                    "node_id": node.id,
                    "prompt": node.instructions,
                    "H": copy.deepcopy(summary),
                    "histories": self.agent.get_history(),
                    "avg_reward": float(avg_reward),
                }
            )

        return results

    def update(
        self,
        rollout_id: int,
        new_prompt: str,
        batch_option: Literal["parent", "fresh", "explicit"] = "parent",
        explicit_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a new prompt z' and add a new node under the parent node.

        Args:
            rollout_id: The id returned by sample() to compare against.
            new_prompt: The candidate prompt to evaluate.
            batch_option: Which batch to evaluate on: "parent" (A/B on same),
                "fresh" (new sampled batch), or "explicit" (provided items).
            explicit_items: When batch_option=="explicit", a list of dataset
                item dicts compatible with BirdEnv.

        Returns:
            Dict with fields: {new_rollout_id, parent_rollout_id, parent_node_id,
                new_node_id, avg_reward_parent, avg_reward_new, delta,
                H_new, histories_new}
        """
        if rollout_id not in self.rollouts:
            raise ValueError(f"Unknown rollout_id: {rollout_id}")

        parent_payload = self.rollouts[rollout_id]
        parent_node_id: int = parent_payload["parent_node_id"]
        parent_avg: float = float(parent_payload["avg_reward"])
        parent_node: TreeNode = self.tree.nodes[parent_node_id]

        # Configure agent with new prompt
        self._set_agent_instructions(new_prompt)

        # Prepare evaluation batch
        if batch_option == "parent":
            eval_envs = parent_payload["env_clone"].clone_with_same_problems()
            problems = eval_envs.get_batch()
        elif batch_option == "fresh":
            self.envs.reset()
            eval_envs = self.envs
            problems = eval_envs.get_batch()
        elif batch_option == "explicit":
            if not explicit_items:
                raise ValueError("explicit_items must be provided when batch_option='explicit'")
            eval_envs = None  # ephemeral path
            env_list: List[BirdEnv] = [BirdEnv(self.args, item=it) for it in explicit_items]
            for e in env_list:
                e.reset()
            problems = [e.get_problem() for e in env_list]
        else:
            raise ValueError(f"Invalid batch_option: {batch_option}")

        # Run rollout
        self.agent.reset(n_sims=len(problems))
        predictions = self.agent.act(problems)
        if eval_envs is not None:
            eval_envs.evaluate(predictions)
            summary_new = eval_envs.get_summary()
        else:
            # explicit path
            summary_new: List[Dict[str, Any]] = []
            for env, preds in zip(env_list, predictions):
                env.evaluate(preds)
                summary_new.extend(env.get_summary())

        avg_new = self._mean_accuracy(summary_new)
        delta = float(avg_new - parent_avg)

        # Add new node to the tree under the parent
        round_idx = (parent_node.round_idx or 0) + 1
        new_node = self.tree.add_node(
            parent=parent_node,
            instructions=new_prompt,
            round_idx=round_idx,
            conversation=[],
        )
        self.tree.update_node(
            new_node,
            performance=avg_new,
            summary=copy.deepcopy(summary_new),
            history=copy.deepcopy(self.agent.history),
        )

        # Persist new rollout handle (use concrete env clone if available)
        new_rollout_id = self._alloc_rollout_id()
        stored_env = (
            eval_envs.clone_with_same_problems() if eval_envs is not None else None
        )
        self.rollouts[new_rollout_id] = {
            "parent_node_id": new_node.id,
            "prompt": new_prompt,
            "avg_reward": avg_new,
            "summary": copy.deepcopy(summary_new),
            "histories": self.agent.get_history(),
            "env_clone": stored_env,
        }

        return {
            "new_rollout_id": new_rollout_id,
            "parent_rollout_id": rollout_id,
            "parent_node_id": parent_node_id,
            "new_node_id": new_node.id,
            "avg_reward_parent": float(parent_avg),
            "avg_reward_new": float(avg_new),
            "delta": float(delta),
            "H_new": copy.deepcopy(summary_new),
            "histories_new": self.agent.get_history(),
        }

    # ----------------------------
    # Helpers
    # ----------------------------
    def _alloc_rollout_id(self) -> int:
        rid = self._next_rollout_id
        self._next_rollout_id += 1
        return rid

    def _select_node(self, selection: Optional[str]) -> Optional[TreeNode]:
        crit = selection or self.tree.selection
        return self.tree.select(crit)

    def _set_agent_instructions(self, instructions: str) -> None:
        self.agent.instructions = instructions
        self.agent.update_agent_prompt(schema=self.agent.schema, instructions=self.agent.instructions)

    @staticmethod
    def _mean_accuracy(summary: List[Dict[str, Any]]) -> float:
        if not summary:
            return 0.0
        return float(sum(s.get("accuracy", 0.0) for s in summary) / len(summary))


__all__ = ["RLGenerator"]