"""
Prompt definitions for the BIRD text-to-SQL task.
"""

BASE_AGENT_PROMPT = """
Task Overview:
You are a data science expert. Below, you are provided with a database schema and
a natural language question. Your task is to understand the schema and generate
a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{schema}
This schema describes the database's structure, including tables, columns,
primary keys, foreign keys, and any relevant relationships or constraints.

**Instructions**
{instructions}
""".strip()

BASE_INSTRUCTIONS = (
    "Return only a single valid SQLite SQL statement in <answer>...</answer>."
)

OBS_QUERY_CLOSER = "Return ONLY the SQL in <answer>...</answer>."
OBS_LINE_TEMPLATE = "{input_list} -> {output_list}"
MEM_INST = ""

PROBLEM_MESSAGE_TEMPLATE = """
Question:
{question}

**Instructions**
{instructions}

Follow the instructions and show your work. When you are ready, return the final
SQL query in tags: <answer> ... </answer>
""".strip()

SELF_EVOLVE_PROMPT = """
You are an expert at designing text-to-SQL agents. The agent is running on a
fixed database schema.
Below is the current agent prompt and a summary of recent performance.
Rewrite ONLY the instructions to improve execution accuracy while maintaining
strict output format.

Current prompt:
{old_prompt}

Evaluation summary over {n_problems} problems and the agent's full thinking process:
{summary}

**How to write Instructions**
- The agent will continue to receive different user queries, so do not make the
  instructions too specific to a single question.
- Keep it concise and practical.
- You may include rules, heuristics, knowledge about the database, low-level
  instructions/examples, high-level ideas/strategies, pitfalls, and any other
  information that can make the agent better.
- Organize however you like (bullets, headings, checklists).
- Do not change the output format; the agent should still return the final SQL
  query in tags: <answer> ... </answer>.

Think step by step and reason about the history of the model's behavior across
iterations.

When you are ready, put your revised Instructions within
<prompt>[your new instructions]</prompt> tags.
""".strip()
