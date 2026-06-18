from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
from vllm import SamplingParams
from openai import OpenAI

from lse.agents.base import BaseAgent
from lse.helpers import extract_from_tag
from lse.prompts.mmlu import get_prompt_class
from lse.textgrad_baseline import (
    build_textgrad_critic_prompt,
    build_textgrad_optimizer_prompt,
    history_to_textgrad_conversation,
    select_representative_failures,
    textgrad_backward_system_prompt,
    textgrad_optimizer_system_prompt,
)
import random


class MMLUAgent(BaseAgent):
    """Self-evolving agent for the MMLU-Redux task.

    Expects problems with 'question' and 'choices'.
    The agent must return ONLY the answer letter (A, B, C, D) in <answer>...</answer>.
    """

    def __init__(self, args, **kwargs):
        self.args = copy.deepcopy(args)
        self.n_sims = self.args.n_sims
        
        # Model runtime
        try:
            self.model_path = self.args.model.name
        except Exception:
            self.model_path = getattr(self.args, "model_path", None)
        self.use_api = self.args.model.url is not None
        self.no_think = getattr(self.args, "no_think", False)
        self.max_tries = getattr(self.args, "max_tries", 3)

        self.act_model = self.model_path
        self.evolve_model = self.args.model.evolve_model

        if self.use_api:
            self.llm = OpenAI(
                **self._build_openai_client_kwargs(
                    base_url=self.args.model.url,
                    api_key=self.args.model.api_key,
                )
            )
        else:
            self._init_vllm_models()

        # Sampling knobs
        sampling = self.args.model.sampling
        self.temperature = float(getattr(sampling, "temperature", 0.7))
        self.top_p = float(getattr(sampling, "top_p", 0.8))
        self.top_k = int(getattr(sampling, "top_k", 20))
        self.min_p = float(getattr(sampling, "min_p", 0.0))
        self.presence_penalty = float(getattr(sampling, "presence_penalty", 0.5))
        self.max_tokens = int(getattr(sampling, "max_tokens", 4096))

        self.prompt_class = get_prompt_class(self.args.task.prompt_style)

        # Prompt class
        self.base_instructions: str = self.prompt_class.base_instructions
        self.instructions = self.base_instructions
        self.update_agent_prompt(instructions=self.instructions)
        self.reset()

    def reset(self, n_sims: int = None) -> None:
        # Chat history per sim
        self.history: List[List[Dict[str, str]]] = []
        for _ in range(n_sims if n_sims is not None else self.n_sims):
            self.history.append([
                {"role": "system", "content": self.agent_prompt},
            ])

    def update_agent_prompt(self, **kwargs):
        self.agent_prompt = self.prompt_class.base_agent_prompt.format(**kwargs)
        return self.agent_prompt

    # ---------- Task-specific helpers ----------
    def _check_valid_answer(self, text: str) -> bool:
        content = extract_from_tag(text, "answer")
        if content is None:
            return False
        content = content.strip().upper()
        return content in ["A", "B", "C", "D"]

    def _format_choices(self, choices: List[str]) -> str:
        labels = ["A", "B", "C", "D"]
        formatted = []
        for label, choice in zip(labels, choices):
            formatted.append(f"{label}. {choice}")
        return "\n".join(formatted)

    def _build_user_prompt(self, problem: Dict[str, Any]) -> str:
        question = problem["question"]
        choices = problem["choices"]
        formatted_choices = self._format_choices(choices)
        
        return self.prompt_class.problem_message_template.format(
            question=question,
            choices=formatted_choices,
            instructions=self.instructions,
        )

    def _build_reference_prompt(self, problem: Dict[str, Any]) -> List[Dict[str, str]]:
        base_instructions = self.prompt_class.base_instructions
        question = problem["question"]
        choices = problem["choices"]
        formatted_choices = self._format_choices(choices)
        
        user_msg = self.prompt_class.problem_message_template.format(
            question=question,
            choices=formatted_choices,
            instructions=base_instructions,
        )
        # Use a temporary prompt with base instructions
        temp_agent_prompt = self.prompt_class.base_agent_prompt.format(
            instructions=base_instructions
        )
        return [
            {"role": "system", "content": temp_agent_prompt},
            {"role": "user", "content": user_msg},
        ]

    def _act_impl(self, problems_batch: List[Dict[str, Any]], n_samples=1):
        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            max_tokens=self.max_tokens,
        )
        if not self.args.model.sample_agent:
            params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                max_tokens=self.max_tokens,
                presence_penalty=self.presence_penalty,
            )

        assert all(h[-1]["role"] != "assistant" for h in self.history), (
            "The agent has already responded; call reset() before new act()"
        )

        # Build per-problem message and append to history
        for i, problem in enumerate(problems_batch):
            user_msg = self._build_user_prompt(problem)
            self.history[i].append({"role": "user", "content": user_msg})

        num_responses = len(self.history)

        # Optional: reference histories using base instructions
        do_compare = self.args.use_ref
        reference_histories: List[List[Dict[str, str]]] = []
        if do_compare:
            for problem in problems_batch:
                reference_histories.append(self._build_reference_prompt(problem))

        chat_histories = self.history + reference_histories

        responses = self._batch_chat(
            chat_histories,
            params,
            checker=lambda t: self._check_valid_answer(t),
            max_tries=self.max_tries,
            no_think=self.no_think,
            n_samples=n_samples,
        ) 

        selected_responses = [random.choice(r) for r in responses]

        # Only log the current responses to history
        for i in range(num_responses):
            r = selected_responses[i]
            self.history[i].append({"role": "assistant", "content": r})
    
        # parse selected responses
        selected_batch: List[List[str]] = []
        for r in selected_responses[:num_responses]:
            content = extract_from_tag(r, "answer") or ""
            content = content.strip().upper()
            selected_batch.append([content] if content else [])

        # Append reference predictions if needed
        if do_compare:
            reference_predictions: List[List[str]] = []
            for r in selected_responses[num_responses:]:
                content = extract_from_tag(r, "answer") or ""
                content = content.strip().upper()
                reference_predictions.append([content] if content else [])

            selected_batch += reference_predictions

        # parse all responses
        all_responses = []
        for i in range(n_samples):
            this_response = [r[i] for r in responses]
            this_batch: List[List[str]] = []
            for r in this_response[:num_responses]:
                content = extract_from_tag(r, "answer") or ""
                content = content.strip().upper()
                this_batch.append([content] if content else [])

            if do_compare:
                reference_predictions: List[List[str]] = []
                for r in this_response[num_responses:]:
                    content = extract_from_tag(r, "answer") or ""
                    content = content.strip().upper()
                    reference_predictions.append([content] if content else [])

                this_batch += reference_predictions

            all_responses.append(this_batch)

        return {'selected': selected_batch, 'all': all_responses}

    def update(self, *args, **kwargs) -> None:
        return None

    def format_history(self, history):
        if isinstance(history, list) and history and isinstance(history[0], dict):
            return ("\n" + "-" * 50 + "\n").join(
                [f"[{m['role'].upper()}]:\n{m['content']}" for m in history]
            )
        return str(history)

    def get_history(self) -> List[str]:
        return [self.format_history(h) for h in self.history]

    def dummy_self_evolve(self, summary: List[Dict[str, Any]], **kwargs):
        return [
            {"role": "system", "content": "Dummy evolve for mmlu"},
            {"role": "assistant", "content": f"<answer>{self.instructions}</answer>"},
        ]

    def textgrad_evolve(self, summary: List[Dict[str, Any]], **kwargs):
        """TextGrad-style two-step evolve: critic feedback then optimizer rewrite."""
        cfg = getattr(self.args, "textgrad", None)
        max_failures = int(getattr(cfg, "max_failures", 6)) if cfg is not None else 6
        max_example_chars = int(getattr(cfg, "max_example_chars", 1200)) if cfg is not None else 1200
        max_instruction_chars = int(getattr(cfg, "max_instruction_chars", 4000)) if cfg is not None else 4000
        forbid_fewshot = bool(getattr(cfg, "forbid_fewshot", True)) if cfg is not None else True

        failures = select_representative_failures(summary, task="mmlu", max_failures=max_failures)
        output_requirement = self.base_instructions

        selected_indices: List[int] = []
        for f in failures:
            for i, s in enumerate(summary or []):
                if s is f:
                    selected_indices.append(i)
                    break

        conversations: List[str] = []
        for i in selected_indices:
            if 0 <= i < len(self.history):
                conversations.append(
                    history_to_textgrad_conversation(self.history[i], max_chars=max_example_chars).strip()
                )

        critic_prompt = build_textgrad_critic_prompt(
            task="mmlu",
            current_instruction=self.instructions,
            conversations=conversations,
            failures=failures,
            output_format_requirement=output_requirement,
            max_example_chars=max_example_chars,
            max_instruction_chars=max_instruction_chars,
        )

        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            max_tokens=self.max_tokens,
        )

        self.mode = "self_evolve"

        critic_messages = [
            {"role": "system", "content": textgrad_backward_system_prompt()},
            {"role": "user", "content": critic_prompt},
        ]
        critic_response = self._batch_chat(
            [critic_messages],
            params,
            checker=lambda t: bool((t or "").strip()),
            max_tries=self.max_tries,
            n_samples=1,
        )[0][0].strip()
        critic_messages.append({"role": "assistant", "content": critic_response})
        critic_feedback = critic_response

        optimizer_prompt = build_textgrad_optimizer_prompt(
            task="mmlu",
            current_instruction=self.instructions,
            critic_feedback=critic_feedback.strip(),
            conversations=conversations,
            output_format_requirement=output_requirement,
            max_instruction_chars=max_instruction_chars,
            forbid_fewshot=forbid_fewshot,
        )
        optimizer_messages = [
            {"role": "system", "content": textgrad_optimizer_system_prompt()},
            {"role": "user", "content": optimizer_prompt},
        ]
        optimizer_response = self._batch_chat(
            [optimizer_messages],
            params,
            checker=lambda t: (extract_from_tag(t, "IMPROVED_VARIABLE") or "").strip() != "",
            max_tries=self.max_tries,
            n_samples=1,
        )[0][0].strip()
        optimizer_messages.append({"role": "assistant", "content": optimizer_response})
        new_instr = extract_from_tag(optimizer_response, "IMPROVED_VARIABLE") or self.instructions
        self.instructions = new_instr.strip()
        if len(self.instructions) > max_instruction_chars and max_instruction_chars > 0:
            self.instructions = self.instructions[:max_instruction_chars].rstrip()
        self.update_agent_prompt(instructions=self.instructions)

        self._last_textgrad_stats = {
            "critic_chars": len(critic_feedback.strip()),
            "instruction_chars": len(self.instructions),
            "n_failures_used": len(failures),
        }

        return critic_messages + optimizer_messages

    def build_evolve_prompt(self, summary: List[Dict[str, Any]]):
        text_summary = []
        for i, s in enumerate(summary):
            lines = []
            lines.append(f"Problem {i+1}: accuracy={s['accuracy']:.2f}")
            
            # We don't want to dump the whole question text if it's huge, but MMLU questions are usually reasonable.
            lines.append(f"Question: {s['question']}")
            choices = self._format_choices(s['choices'])
            lines.append(f"Choices:\n{choices}")

            ok = 'True' if s['accuracy'] == 1 else 'False'
            lines.append(
                f"Model's response:\n{s['pred_outputs']}\nGround truth: {s['gold_outputs']}\nCorrect: {ok}"
            )
            
            if self.args.task.include_full_response:
                model_response = self.history[i][-1]["content"]
                lines.append("Model's full thinking process:")
                lines.append(model_response)
            text_summary.append("\n".join(lines).strip())

        text_summary = ("\n" + "=" * 50 + "\n").join(text_summary)

        evolve_prompt = self.prompt_class.self_evolve_prompt.format(
            old_prompt=self.agent_prompt,
            n_problems=self.n_sims,
            summary=text_summary.strip(),
        )
        return evolve_prompt

    def _self_evolve_impl(self, summary: List[Dict[str, Any]], **kwargs):
        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            max_tokens=self.max_tokens,
        )

        evolve_prompt = self.build_evolve_prompt(summary)

        messages = [{"role": "user", "content": evolve_prompt}]

        response_text = self._batch_chat(
            [messages],
            params,
            checker=lambda t: extract_from_tag(t, "prompt") is not None,
            max_tries=self.max_tries,
            n_samples=1,
        )[0][0].strip()

        messages.append({"role": "assistant", "content": response_text})
        new_instr = extract_from_tag(response_text, "prompt") or self.instructions
        self.instructions = new_instr.strip()
        self.update_agent_prompt(instructions=self.instructions)
        return messages

