from dataclasses import dataclass


@dataclass
class MMLUPromptClass:
    base_agent_prompt: str
    self_evolve_prompt: str
    mem_inst: str
    obs_line_template: str
    problem_message_template: str
    obs_closer: str
    base_instructions: str


def get_prompt_class(prompt_style: str) -> MMLUPromptClass:
    if prompt_style != "instruction":
        raise ValueError(
            f"Only prompt_style='instruction' is supported in this paper release. Got: {prompt_style}"
        )

    import lse.prompts.mmlu.instruction_prompt as MP

    return MMLUPromptClass(
        base_agent_prompt=MP.BASE_AGENT_PROMPT,
        self_evolve_prompt=MP.SELF_EVOLVE_PROMPT,
        mem_inst=MP.MEM_INST,
        obs_line_template=MP.OBS_LINE_TEMPLATE,
        problem_message_template=MP.PROBLEM_MESSAGE_TEMPLATE,
        obs_closer=MP.OBS_QUERY_CLOSER,
        base_instructions=MP.BASE_INSTRUCTIONS,
    )

