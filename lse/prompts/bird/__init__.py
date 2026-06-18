from dataclasses import dataclass


@dataclass
class BirdPromptClass:
    base_agent_prompt: str
    self_evolve_prompt: str
    mem_inst: str
    obs_line_template: str
    problem_message_template: str
    obs_closer: str
    base_instructions: str


def get_prompt_class(prompt_style: str) -> BirdPromptClass:
    if prompt_style != "instruction":
        raise ValueError(
            f"Only prompt_style='instruction' is supported in this paper release. Got: {prompt_style}"
        )

    from . import instruction_prompt as BP

    return BirdPromptClass(
        base_agent_prompt=BP.BASE_AGENT_PROMPT,
        self_evolve_prompt=BP.SELF_EVOLVE_PROMPT,
        mem_inst=BP.MEM_INST,
        obs_line_template=BP.OBS_LINE_TEMPLATE,
        problem_message_template=BP.PROBLEM_MESSAGE_TEMPLATE,
        obs_closer=BP.OBS_QUERY_CLOSER,
        base_instructions=BP.BASE_INSTRUCTIONS,
    )


__all__ = ["BirdPromptClass", "get_prompt_class"]
