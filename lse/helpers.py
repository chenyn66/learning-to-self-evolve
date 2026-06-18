"""Lightweight helpers that avoid heavy dependencies.

Provides simple tag extraction and LaTeX boxed answer parsing used by agents.
"""
from __future__ import annotations

from typing import Optional
import os

def truncate_repetitive_suffix(
    text: str,
    *,
    min_run_length: int = 200,
    lookback: int = 8192,
    add_marker: bool = False,
) -> tuple[str, dict]:
    """Truncate pathological repetitive suffixes (e.g. '0000...0', '!!!!...!').

    This is a lightweight safety valve for occasional model degeneration where the
    output enters a high-repetition loop near the end. It is intentionally conservative:
    we only truncate when we see a *suffix* repetition that is both long and obvious.

    Returns:
        (new_text, info) where info includes:
            - truncated: bool
            - reason: str | None
            - cut_idx: int | None  (index in original string)
            - repeat_char: str | None
            - repeat_len: int | None
            - block_len: int | None
            - block_repeats: int | None
    """
    info = {
        "truncated": False,
        "reason": None,
        "cut_idx": None,
        "repeat_char": None,
        "repeat_len": None,
        "block_len": None,
        "block_repeats": None,
    }

    if not isinstance(text, str) or not text:
        return ("" if text is None else str(text)), info

    if min_run_length <= 0 or len(text) < min_run_length:
        return text, info

    # Limit scanning to tail window for speed.
    lookback = int(lookback) if isinstance(lookback, int) else 8192
    lookback = max(min(len(text), max(1, lookback)), min_run_length)
    tail = text[-lookback:]
    offset = len(text) - len(tail)

    # 1) Trailing single-character run.
    last = tail[-1]
    # Avoid truncating pure whitespace tails.
    if last not in {" ", "\n", "\r", "\t"}:
        i = len(tail) - 1
        while i >= 0 and tail[i] == last:
            i -= 1
        run_len = (len(tail) - 1) - i
        if run_len >= min_run_length:
            cut_idx = offset + i + 1
            out = text[:cut_idx].rstrip()
            if add_marker:
                out = out + "\n\n[TRUNCATED_REPETITIVE_SUFFIX]\n"
            info.update(
                {
                    "truncated": True,
                    "reason": "char_run",
                    "cut_idx": cut_idx,
                    "repeat_char": last,
                    "repeat_len": run_len,
                }
            )
            return out, info

    # 2) Trailing repeated block (captures e.g. 'abcabcabc...' or '0\\n0\\n0\\n...').
    # Keep the candidate lengths small to avoid false positives and heavy work.
    block_lens = (2, 3, 4, 5, 8, 16, 32, 64)
    for bl in block_lens:
        if bl > len(tail) // 2:
            continue
        block = tail[-bl:]
        if not block or block.strip() == "":
            continue
        j = len(tail)
        reps = 0
        while j >= bl and tail[j - bl : j] == block:
            reps += 1
            j -= bl
        if reps >= 3 and reps * bl >= min_run_length:
            cut_idx = offset + j
            out = text[:cut_idx].rstrip()
            if add_marker:
                out = out + "\n\n[TRUNCATED_REPETITIVE_SUFFIX]\n"
            info.update(
                {
                    "truncated": True,
                    "reason": "block_repeat",
                    "cut_idx": cut_idx,
                    "block_len": bl,
                    "block_repeats": reps,
                }
            )
            return out, info

    return text, info


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def remove_boxed(s: str) -> Optional[str]:
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def extract_boxed_answer(solution: str) -> Optional[str]:
    solution = last_boxed_only_string(solution)
    if solution is None:
        return None
    solution = remove_boxed(solution)
    return solution


def extract_answer(passage: str) -> Optional[str]:
    if "\\boxed" in passage or "\\fbox" in passage:
        return extract_boxed_answer(passage)
    return None





# def extract_from_tag(string: str, tag: str) -> Optional[str]:
#     start_idx = string.rfind(f"<{tag}>")
#     if start_idx == -1:
#         return None
#     end_idx = string.rfind(f"</{tag}>")
#     if end_idx == -1:
#         return None
#     return string[start_idx + len(f"<{tag}>") : end_idx]



def extract_from_sql(string: str) -> Optional[str]:
    '''

    ```sql
SELECT DISTINCT foreign_data.language
FROM foreign_data
INNER JOIN cards ON foreign_data.uuid = cards.uuid
WHERE cards.name = 'Annul' AND cards.number = '29';
```

    '''

    start_idx = string.rfind("```sql")
    if start_idx == -1:
        return None
    end_idx = string.rfind("```")
    if end_idx == -1:
        return None
    return string[start_idx + len("```sql") : end_idx]



def extract_from_tag(string: str, tag: str) -> Optional[str]:
    """Extract content enclosed by the opening tag that matches the last closing tag.

    Handles nested tags of the same name efficiently. For example,
    "<tag> a <tag> b </tag> </tag>" returns "a <tag> b </tag>".
    """

    if tag == "answer" and os.environ.get('USE_ARCTIC', '0') == '1': # Hardcoded for handling sql answers
        maybe_sql = extract_from_sql(string)
        if maybe_sql is not None:
            return maybe_sql
        
    open_pat = f"<{tag}>"
    close_pat = f"</{tag}>"

    last_close = string.rfind(close_pat)
    if last_close == -1:
        return ''

    stack = []  # positions of unmatched opening tags
    i = 0
    open_len = len(open_pat)
    close_len = len(close_pat)

    # Scan left-to-right only up to and including the last closing tag.
    while True:
        next_open = string.find(open_pat, i)
        next_close = string.find(close_pat, i)

        # Only consider events that occur at or before the last closing tag.
        pos_open = next_open if (next_open != -1 and next_open <= last_close) else float("inf")
        pos_close = next_close if (next_close != -1 and next_close <= last_close) else float("inf")

        if pos_open == float("inf") and pos_close == float("inf"):
            # No opening tag before the last closing tag -> unmatched.
            return ''

        if pos_open < pos_close:
            stack.append(pos_open)
            i = pos_open + open_len
        else:
            # Encounter a closing tag
            if not stack:
                # Unmatched close; if it's the last close, cannot extract.
                if pos_close == last_close:
                    return ''
                i = pos_close + close_len
                continue

            open_pos = stack.pop()
            if pos_close == last_close:
                # Found the opening that matches the last closing tag.
                return string[open_pos + open_len : pos_close]
            i = pos_close + close_len

    # Unreachable
    # return None
