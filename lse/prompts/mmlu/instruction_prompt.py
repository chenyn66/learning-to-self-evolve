"""
Prompts for the MMLU-Redux (multiple choice) task.
"""

# ---------- Agent (system) prompt used while solving the task ----------
BASE_AGENT_PROMPT = '''
Task Overview:
You are an expert taking a test. Below, you are provided with a question and a list of choices. Your task is to select the correct answer from the choices.

**Instructions**
{instructions}
'''.strip()

BASE_INSTRUCTIONS = (
    "Return only the letter of the correct choice (A, B, C, or D) in <answer>...</answer>."
)

# Kept for API compatibility
OBS_QUERY_CLOSER = "Return ONLY the answer letter in <answer>...</answer>."
OBS_LINE_TEMPLATE = "{input_list} -> {output_list}"

# ---------- Prompt for formatting the actual problem (user message) ----------
PROBLEM_MESSAGE_TEMPLATE = '''
Question:
{question}

Choices:
{choices}

Follow the instructions and show your work. When you are ready, return the answer letter in tags: <answer> ... </answer>
'''.strip()

PROBLEM_MESSAGE_TEMPLATE_OLD = '''
Question:
{question}

Choices:
{choices}

**Instructions**
{instructions}

Follow the instructions and show your work. When you are ready, return the answer letter in tags: <answer> ... </answer>
'''.strip()

# ---------- Meta‑agent evolve prompt ----------
SELF_EVOLVE_PROMPT = (
    """
You are an expert at designing agents for solving multiple-choice questions that involve both factual knowledge and reasoning.
Below is the current agent prompt and a summary of recent performance on a set of problmes.
Rewrite ONLY the instructions to improve accuracy while maintaining strict output format.

Current prompt:
{old_prompt}

Evaluation summary over {n_problems} problems and the agent's full thinking process:
{summary}

**How to write Instructions**
- The agent will continue to receive different questions from the same subjects. Don't make the instructions too specific to a single question.
- Keep it concise and practical.
- You may include rules, heuristics, strategies for multiple choice questions (e.g., elimination, careful reading), knowledge about the subjects (e.g., common misconceptions, important facts, etc.), and any information that you think can make the agent better.
- Organize however you like (bullets, headings, checklists).
- Be creative and think about the agent's behavior across iterations. Don't be confined by what I told you.
- Don't change the output format, the agent should still return the finalanswer letter in tags: <answer> ... </answer>.

Think step by step and show your work. Reason about the history of the model's behavior across iterations.

When you are ready, put your revised Instructions within <prompt>[your new instructions]</prompt> tags.
"""
    .strip()
)

MEM_INST = ''

