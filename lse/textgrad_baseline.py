from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _truncate(text: Any, max_chars: int) -> str:
    raw = _safe_str(text)
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw
    if max_chars <= 20:
        return raw[:max_chars]
    return raw[: max_chars - 15] + " ...[truncated]"


def _normalize_error_key(error_text: Any) -> str:
    err = _safe_str(error_text).strip().lower()
    if not err:
        return "no_error_text"
    # Coarse bucket to promote diversity while staying deterministic.
    return err[:120]


def select_representative_failures(
    summary: List[Dict[str, Any]],
    *,
    task: str,
    max_failures: int = 6,
) -> List[Dict[str, Any]]:
    """Pick a small representative subset of examples for critic prompts.

    Prefer incorrect examples. For BIRD, diversify by error text first.
    If there are no failures in the round, fall back to the first examples.
    """
    if not summary:
        return []

    max_failures = max(1, int(max_failures))
    failures = [s for s in summary if float(s.get("accuracy", 0.0)) < 1.0]
    pool = failures if failures else list(summary)

    if task.lower() != "bird":
        return pool[:max_failures]

    selected: List[Dict[str, Any]] = []
    seen_error_keys = set()

    # First pass: one sample per unique error key.
    for item in pool:
        key = _normalize_error_key(item.get("error", ""))
        if key in seen_error_keys:
            continue
        selected.append(item)
        seen_error_keys.add(key)
        if len(selected) >= max_failures:
            return selected

    # Second pass: fill remainder with earliest examples.
    for item in pool:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= max_failures:
            break

    return selected


def _format_mmlu_choices(choices: Any) -> str:
    if not isinstance(choices, list):
        return _safe_str(choices)
    labels = ["A", "B", "C", "D"]
    lines = []
    for label, choice in zip(labels, choices):
        lines.append(f"{label}. {_safe_str(choice)}")
    return "\n".join(lines)


def _build_bird_failure_block(
    failures: List[Dict[str, Any]],
    *,
    max_example_chars: int,
) -> str:
    if not failures:
        return "No failures in this batch."

    blocks = []
    for idx, f in enumerate(failures, start=1):
        blocks.append(
            (
                f"Failure {idx}\n"
                f"- Accuracy: {f.get('accuracy', 0)}\n"
                f"- Question: {_truncate(f.get('test_inputs', ''), max_example_chars)}\n"
                f"- Predicted SQL: {_truncate(f.get('pred_outputs', ''), max_example_chars)}\n"
                f"- Gold SQL: {_truncate(f.get('gold_outputs', ''), max_example_chars)}\n"
                f"- Error evidence: {_truncate(f.get('error', ''), max_example_chars)}"
            )
        )
    return "\n\n".join(blocks)


def _build_mmlu_failure_block(
    failures: List[Dict[str, Any]],
    *,
    max_example_chars: int,
) -> str:
    if not failures:
        return "No failures in this batch."

    blocks = []
    for idx, f in enumerate(failures, start=1):
        blocks.append(
            (
                f"Failure {idx}\n"
                f"- Accuracy: {f.get('accuracy', 0)}\n"
                f"- Question: {_truncate(f.get('question', ''), max_example_chars)}\n"
                f"- Choices:\n{_truncate(_format_mmlu_choices(f.get('choices', [])), max_example_chars)}\n"
                f"- Predicted option: {_truncate(f.get('pred_outputs', ''), max_example_chars)}\n"
                f"- Gold option: {_truncate(f.get('gold_outputs', ''), max_example_chars)}\n"
                f"- Error evidence: {_truncate(f.get('error', ''), max_example_chars)}"
            )
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# TextGrad native prompt templates (verbatim from the cloned `textgrad` repo)
# ---------------------------------------------------------------------------
#
# We keep these templates to align LSE's TextGrad baseline prompts with TextGrad's
# native backward (critic) + TGD (optimizer) prompts and tag structure.

# From: textgrad/textgrad/autograd/llm_backward_prompts.py
GLOSSARY_TEXT_BACKWARD = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <OBJECTIVE_FUNCTION>: The objective of the optimization task.
# - <VARIABLE>: Specifies the span of the variable.
# - <ROLE>: The role description of the variable."""

BACKWARD_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves a given text (i.e. the variable). You are the gradient (feedback) engine. "
    "Your only responsibility is to give intelligent and creative feedback and constructive criticism to variables, given an objective specified in <OBJECTIVE_FUNCTION> </OBJECTIVE_FUNCTION> tags. "
    "The variables may be solutions to problems, prompts to language models, code, or any other text-based variable. "
    "Pay attention to the role description of the variable, and the context in which it is used. You should assume that the variable will be used in a similar context in the future. "
    "Only provide strategies, explanations, and methods to change in the variable. DO NOT propose a new version of the variable, that will be the job of the optimizer. Your only job is to send feedback and criticism (compute 'gradients'). "
    "For instance, feedback can be in the form of 'Since language models have the X failure mode...', 'Adding X can fix this error because...', 'Removing X can improve the objective function because...', 'Changing X to Y would fix the mistake ...', that gets at the downstream objective.\n"
    "If a variable is already working well (e.g. the objective function is perfect, an evaluation shows the response is accurate), you should not give feedback.\n"
    f"{GLOSSARY_TEXT_BACKWARD}"
)

CONVERSATION_TEMPLATE = (
    "<LM_SYSTEM_PROMPT> {system_prompt} </LM_SYSTEM_PROMPT>\n\n"
    "<LM_INPUT> {prompt} </LM_INPUT>\n\n"
    "<LM_OUTPUT> {response_value} </LM_OUTPUT>\n\n"
)

CONVERSATION_START_INSTRUCTION_CHAIN = (
    "You will give feedback to a variable with the following role: <ROLE> {variable_desc} </ROLE>. "
    "Here is a conversation with a language model (LM):\n\n"
    "{conversation}"
)
OBJECTIVE_INSTRUCTION_CHAIN = (
    "This conversation is part of a larger system. The <LM_OUTPUT> was later used as {response_desc}.\n\n"
    "<OBJECTIVE_FUNCTION>Your goal is to give feedback to the variable to address the following feedback on the LM_OUTPUT: {response_gradient} </OBJECTIVE_FUNCTION>\n\n"
)

EVALUATE_VARIABLE_INSTRUCTION = (
    "We are interested in giving feedback to the {variable_desc} "
    "for this conversation. Specifically, give feedback to the following span "
    "of text:\n\n<VARIABLE> "
    "{variable_short} </VARIABLE>\n\n"
    "Given the above history, describe how the {variable_desc} "
    "could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.\n\n"
)

# From: textgrad/textgrad/optimizer/optimizer_prompts.py
GLOSSARY_TEXT = """
### Glossary of tags that will be sent to you:
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <FEEDBACK>: The feedback to the variable.
# - <CONVERSATION>: The conversation history.
# - <FOCUS>: The focus of the optimization.
# - <ROLE>: The role description of the variable."""

OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve prompts, solutions to problems, code, or any other text-based variable. "
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy, identify what is important and what is correct. "
    "Pay attention to the role description of the variable, and the context in which it is used. "
    "This is very important: You MUST give your response by sending the improved variable between {new_variable_start_tag} {{improved variable}} {new_variable_end_tag} tags. "
    "The text you send between the tags will directly replace the variable.\n\n"
    f"{GLOSSARY_TEXT}"
)

TGD_PROMPT_PREFIX = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {variable_short} </VARIABLE>\n\n"
    "Here is the context and feedback we got for the variable:\n\n"
    "<CONTEXT>{variable_grad}</CONTEXT>\n\n"
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags.\n"
)

TGD_MULTIPART_PROMPT_INIT = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The variable is the text within the following span: <VARIABLE> {variable_short} </VARIABLE>\n\n"
    "Here is the context and feedback we got for the variable:\n\n"
)

TGD_MULTIPART_PROMPT_PREFIX = (
    "Improve the variable ({variable_desc}) using the feedback provided in <FEEDBACK> tags.\n"
)

TGD_PROMPT_SUFFIX = (
    "Send the improved variable "
    "in the following format:\n\n{new_variable_start_tag}{{the improved variable}}{new_variable_end_tag}\n\n"
    "Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else."
)

MOMENTUM_PROMPT_ADDITION = (
    "Here are the past iterations of this variable:\n\n"
    "<PAST_ITERATIONS>{past_values}</PAST_ITERATIONS>\n\n"
    "Similar feedbacks across different steps suggests that the modifications to the variable are insufficient."
    "If this is the case, please make more significant changes to the variable.\n\n"
)

CONSTRAINT_PROMPT_ADDITION = (
    "You must follow the following constraints:\n\n"
    "<CONSTRAINTS>{constraint_text}</CONSTRAINTS>\n\n"
)

IN_CONTEXT_EXAMPLE_PROMPT_ADDITION = (
    "You must base on the following examples when modifying the {variable_desc}:\n\n"
    "<EXAMPLES>{in_context_examples}</EXAMPLES>\n\n"
)


def construct_tgd_prompt(
    do_momentum: bool = False,
    do_constrained: bool = False,
    do_in_context_examples: bool = False,
    **optimizer_kwargs: Any,
):
    """
    Construct the textual gradient descent prompt.

    :param do_momentum: Whether to include momentum in the prompt.
    :type do_momentum: bool, optional
    :param do_constrained: Whether to include constraints in the prompt.
    :type do_constrained: bool, optional
    :param do_in_context_examples: Whether to include in-context examples in the prompt.
    :type do_in_context_examples: bool, optional
    :param optimizer_kwargs: Additional keyword arguments for formatting the prompt. These will be things like the variable description, gradient, past values, constraints, and in-context examples.
    :return: The TGD update prompt.
    :rtype: str
    """

    if isinstance(optimizer_kwargs["variable_grad"], str):
        multipart = False
        prompt = TGD_PROMPT_PREFIX.format(**optimizer_kwargs)

    else:
        gradient_context = optimizer_kwargs["variable_grad"]
        gradient_context = [TGD_MULTIPART_PROMPT_INIT.format(**optimizer_kwargs)] + gradient_context
        multipart = True
        prompt = TGD_MULTIPART_PROMPT_PREFIX.format(**optimizer_kwargs)

    if do_momentum:
        prompt += MOMENTUM_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_constrained:
        prompt += CONSTRAINT_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_in_context_examples:
        prompt += IN_CONTEXT_EXAMPLE_PROMPT_ADDITION.format(**optimizer_kwargs)

    prompt += TGD_PROMPT_SUFFIX.format(**optimizer_kwargs)

    if not multipart:
        return prompt

    return gradient_context + [prompt]


GRADIENT_TEMPLATE = (
    "Here is a conversation:\n\n<CONVERSATION>{context}</CONVERSATION>\n\n"
    "This conversation is potentially part of a larger system. The output is used as {response_desc}\n\n"
    "Here is the feedback we got for {variable_desc} in the conversation:\n\n<FEEDBACK>{feedback}</FEEDBACK>\n\n"
)


def textgrad_backward_system_prompt() -> str:
    return BACKWARD_SYSTEM_PROMPT


def textgrad_optimizer_system_prompt(
    *,
    new_variable_start_tag: str = "<IMPROVED_VARIABLE>",
    new_variable_end_tag: str = "</IMPROVED_VARIABLE>",
) -> str:
    return OPTIMIZER_SYSTEM_PROMPT.format(
        new_variable_start_tag=new_variable_start_tag,
        new_variable_end_tag=new_variable_end_tag,
    )


def history_to_textgrad_conversation(
    history: Sequence[Dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    """Convert a single agent chat history into TextGrad's <LM_*> conversation template."""
    system_prompt = ""
    user_prompt = ""
    assistant_output = ""

    for m in history or []:
        role = _safe_str(m.get("role"))
        if role == "system" and not system_prompt:
            system_prompt = _safe_str(m.get("content"))
        elif role == "user":
            user_prompt = _safe_str(m.get("content"))

    for m in reversed(list(history or [])):
        if _safe_str(m.get("role")) == "assistant":
            assistant_output = _safe_str(m.get("content"))
            break

    return CONVERSATION_TEMPLATE.format(
        system_prompt=_truncate(system_prompt, max_chars),
        prompt=_truncate(user_prompt, max_chars),
        response_value=_truncate(assistant_output, max_chars),
    )


def _default_variable_desc(task: str) -> str:
    task_name = (task or "").strip().lower()
    if task_name == "bird":
        return "the instruction that guides the LM to solve text-to-SQL problems and format the final SQL answer"
    return "the instruction that guides the LM to solve multiple-choice QA problems and format the final answer option"


def _build_response_gradient_text(
    *,
    task: str,
    failures: List[Dict[str, Any]],
    output_format_requirement: str,
    max_example_chars: int,
) -> str:
    task_name = (task or "").strip().lower()
    if task_name == "bird":
        failure_block = _build_bird_failure_block(failures, max_example_chars=max_example_chars)
        metric_desc = "execution correctness / exact match (accuracy)"
    else:
        failure_block = _build_mmlu_failure_block(failures, max_example_chars=max_example_chars)
        metric_desc = "multiple-choice accuracy"

    fmt_req = _truncate(output_format_requirement, max_example_chars)
    return (
        f"Metric: {metric_desc}.\n"
        f"Observed failures:\n{failure_block}\n\n"
        f"Must preserve this output format requirement exactly:\n{fmt_req}"
    ).strip()


def _build_constraint_text(
    *,
    output_format_requirement: str,
    max_instruction_chars: int,
    forbid_fewshot: bool,
) -> str:
    constraints = [
        "Constraint 1: Keep the instruction generalizable across future problems.",
        "Constraint 2: Preserve and reinforce the output format requirement exactly.",
        f"Constraint 3: Keep the final instruction under {int(max_instruction_chars)} characters.",
        "Constraint 4: Do not include chain-of-thought examples or dataset-specific memorization.",
    ]
    if forbid_fewshot:
        constraints.insert(
            3,
            "Constraint 4: Do not add demonstrations, few-shot examples, or copied training questions.",
        )
        constraints[-1] = "Constraint 5: Do not include chain-of-thought examples or dataset-specific memorization."

    constraints.append(f"Output format requirement:\n{output_format_requirement}")
    return "\n".join(constraints).strip()


def build_textgrad_critic_prompt(
    *,
    task: str,
    current_instruction: str,
    conversations: List[str],
    failures: List[Dict[str, Any]],
    output_format_requirement: str,
    max_example_chars: int = 1200,
    max_instruction_chars: int = 4000,
) -> str:
    """Build TextGrad-native backward (critic) user prompt (feedback only, no rewrite)."""
    variable_desc = _default_variable_desc(task)
    variable_short = _truncate(current_instruction, max_instruction_chars)
    conversation_text = "\n\n".join([_safe_str(c) for c in (conversations or []) if _safe_str(c).strip() != ""]).strip()

    response_desc = "the model's outputs that are scored for correctness"
    response_gradient = _build_response_gradient_text(
        task=task,
        failures=failures,
        output_format_requirement=output_format_requirement,
        max_example_chars=max_example_chars,
    )

    prompt = CONVERSATION_START_INSTRUCTION_CHAIN.format(
        variable_desc=variable_desc,
        conversation=conversation_text,
    )
    prompt += OBJECTIVE_INSTRUCTION_CHAIN.format(
        response_desc=response_desc,
        response_gradient=response_gradient,
    )
    prompt += EVALUATE_VARIABLE_INSTRUCTION.format(
        variable_desc=variable_desc,
        variable_short=variable_short,
    )
    return prompt.strip()


def build_textgrad_optimizer_prompt(
    *,
    task: str,
    current_instruction: str,
    critic_feedback: str,
    conversations: List[str],
    output_format_requirement: str,
    max_instruction_chars: int = 4000,
    forbid_fewshot: bool = True,
) -> str:
    """Build TextGrad-native TGD (optimizer) user prompt."""
    variable_desc = _default_variable_desc(task)
    variable_short = _truncate(current_instruction, max_instruction_chars)
    conversation_text = "\n\n".join([_safe_str(c) for c in (conversations or []) if _safe_str(c).strip() != ""]).strip()
    response_desc = "the model's outputs that are scored for correctness"

    variable_grad = GRADIENT_TEMPLATE.format(
        context=conversation_text,
        response_desc=response_desc,
        variable_desc=variable_desc,
        feedback=_safe_str(critic_feedback).strip(),
    )

    constraint_text = _build_constraint_text(
        output_format_requirement=output_format_requirement,
        max_instruction_chars=max_instruction_chars,
        forbid_fewshot=forbid_fewshot,
    )

    return construct_tgd_prompt(
        do_momentum=False,
        do_constrained=True,
        do_in_context_examples=False,
        variable_desc=variable_desc,
        variable_short=variable_short,
        variable_grad=variable_grad,
        constraint_text=constraint_text,
        in_context_examples="",
        new_variable_start_tag="<IMPROVED_VARIABLE>",
        new_variable_end_tag="</IMPROVED_VARIABLE>",
    ).strip()


__all__ = [
    "build_textgrad_critic_prompt",
    "build_textgrad_optimizer_prompt",
    "history_to_textgrad_conversation",
    "select_representative_failures",
    "textgrad_backward_system_prompt",
    "textgrad_optimizer_system_prompt",
]
