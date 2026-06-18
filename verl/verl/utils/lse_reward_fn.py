import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from lse.agents import BirdAgent, MMLUAgent
from lse.envs import BatchBird, BatchMMLU
from lse.helpers import extract_from_tag


for name in ("httpx", "httpcore"):
    log = logging.getLogger(name)
    log.setLevel(logging.WARNING)
    log.propagate = False


INVALID_REWARD = -1.0
PRINT_RATE = float(os.environ.get("PRINT_RATE", 0.2))
# NOTE: Safety knob for reward-model generation.
# The reward function calls an OpenAI-compatible `/v1/chat/completions` endpoint (vLLM router).
# If `max_tokens` is too large for the reward model's context window, vLLM will 400 the request.
# Use `REWARD_MAX_TOKENS` to cap per-request generation length.
MAX_REWARD_TOKENS = int(os.environ.get("REWARD_MAX_TOKENS", 2048))


_GEN_SEM = asyncio.Semaphore(int(os.environ.get("SEM_MAX", 16)))
if os.environ.get("GEN_MAX"):
    _GEN_SEM = asyncio.Semaphore(int(os.environ.get("GEN_MAX")))
_EVAL_SEM = asyncio.Semaphore(int(os.environ.get("SEM_MAX", 16)))
if os.environ.get("EVAL_MAX"):
    _EVAL_SEM = asyncio.Semaphore(int(os.environ.get("EVAL_MAX")))

_WORKER_POOL = ThreadPoolExecutor(max_workers=200)


# set config overrides for evaluation environment

N_EVAL = int(os.environ.get("N_EVAL", -1))
MAX_TRIES = int(os.environ.get("MAX_TRIES", -1))


def _build_model_url(reward_router_address: str | None) -> str:
    if reward_router_address:
        return f"http://{reward_router_address}/v1"
    return os.environ.get("REWARD_URL", "http://0.0.0.0:30000/v1")


def _override_config(config, reward_router_address) :
    config.model.url = _build_model_url(reward_router_address)
    config.model.api_key = "None"
    config.use_ref = False
    config.model.sample_agent = True
    
    if config.model.sampling.temperature == 0.0:
        config.model.sampling.temperature = 0.7

    if MAX_TRIES > 0:
        config.max_tries = MAX_TRIES
    if N_EVAL > 0:
        config.test_n_eval = N_EVAL
    return config

async def _compute_bird_score(
    new_prompt: str,
    extra_info: dict,
    reward_router_address: str | None,
) -> float:
    config = OmegaConf.create(extra_info["config"])
    config = _override_config(config, reward_router_address)
    # NOTE: vLLM uses request-level max_tokens; clamp it to avoid context-length 400s.
    current_max_tokens = int(getattr(config.model.sampling, "max_tokens", MAX_REWARD_TOKENS))
    config.model.sampling.max_tokens = min(current_max_tokens, MAX_REWARD_TOKENS)

    test_envs = BatchBird.create_test_envs(config, extra_info["test_env"])
    test_agent = BirdAgent(config, extra_info["test_env"]["db_id"])
    test_agent.update_agent_prompt(schema=test_agent.schema, instructions=new_prompt)

    def do_generate():
        # Keep the problems stable for this scoring call.
        test_agent.reset(n_sims=test_envs.n_sims)
        test_envs.reset()
        test_problems = test_envs.get_batch()
        # BirdAgent.act returns:
        #   {"all": [n_samples][n_questions][1 sql str], "selected": ...}
        return test_agent.act(test_problems, n_samples=config.test_n_eval)["all"]

    start_time = time.time()
    async with _GEN_SEM:
        test_predictions = await asyncio.get_running_loop().run_in_executor(_WORKER_POOL, do_generate)
    assert len(test_predictions) == config.test_n_eval, (
        f"Expected {config.test_n_eval} samples, got {len(test_predictions)}"
    )
    generate_time = time.time() - start_time

    def do_eval(preds):
        accs = []
        for i in range(config.test_n_eval):
            test_envs.evaluate(preds[i])
            summary = test_envs.get_summary()
            accs.append(sum(s["accuracy"] for s in summary) / len(summary))
        return float(np.mean(accs))

    async with _EVAL_SEM:
        test_avg_acc = await asyncio.get_running_loop().run_in_executor(_WORKER_POOL, do_eval, test_predictions)

    evaluate_time = time.time() - start_time - generate_time
    total_time = time.time() - start_time
    logging.info("Generate time: %s seconds", generate_time)
    logging.info("Evaluate time: %s seconds", evaluate_time)
    logging.info("Total time: %s seconds", total_time)

    if np.random.random() < PRINT_RATE:
        print(f"prompt:\n{new_prompt}")
        print(f"test_avg_acc: {test_avg_acc}")
        print(f"extra_info['performance']: {extra_info['performance']}")
        print(f"reward: {test_avg_acc - extra_info['performance']}")
        print("=" * 100)

    return float(test_avg_acc)


async def _compute_mmlu_score(
    new_prompt: str,
    extra_info: dict,
    reward_router_address: str | None,
) -> float:
    config = OmegaConf.create(extra_info["config"])
    config = _override_config(config, reward_router_address)
    # NOTE: vLLM uses request-level max_tokens; clamp it to avoid context-length 400s.
    # current_max_tokens = int(getattr(config.model.sampling, "max_tokens", MAX_REWARD_TOKENS))
    # config.model.sampling.max_tokens = min(current_max_tokens, MAX_REWARD_TOKENS)

    test_envs = BatchMMLU.create_test_envs(config, extra_info["test_env"])
    test_agent = MMLUAgent(config)
    test_agent.update_agent_prompt(instructions=new_prompt)

    def do_generate():
        test_agent.reset(n_sims=test_envs.n_sims)
        test_envs.reset()
        test_problems = test_envs.get_batch()
        return test_agent.act(test_problems, n_samples=config.test_n_eval)["all"]

    start_time = time.time()
    async with _GEN_SEM:
        test_predictions = await asyncio.get_running_loop().run_in_executor(_WORKER_POOL, do_generate)
    assert len(test_predictions) == config.test_n_eval, (
        f"Expected {config.test_n_eval} samples, got {len(test_predictions)}"
    )
    generate_time = time.time() - start_time

    def do_eval(preds):
        accs = []
        for i in range(config.test_n_eval):
            test_envs.evaluate(preds[i])
            summary = test_envs.get_summary()
            accs.append(sum(s["accuracy"] for s in summary) / len(summary))
        return float(np.mean(accs))

    async with _EVAL_SEM:
        test_avg_acc = await asyncio.get_running_loop().run_in_executor(_WORKER_POOL, do_eval, test_predictions)

    evaluate_time = time.time() - start_time - generate_time
    total_time = time.time() - start_time
    logging.info("Generate time: %s seconds", generate_time)
    logging.info("Evaluate time: %s seconds", evaluate_time)
    logging.info("Total time: %s seconds", total_time)

    if np.random.random() < PRINT_RATE:
        print(f"prompt:\n{new_prompt}")
        print(f"test_avg_acc: {test_avg_acc}")
        print(f"extra_info['performance']: {extra_info['performance']}")
        print(f"reward: {test_avg_acc - extra_info['performance']}")
        print("=" * 100)

    return float(test_avg_acc)


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
    reward_router_address: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    del ground_truth, kwargs

    new_prompt = extract_from_tag(solution_str, "prompt")
    new_prompt = new_prompt.strip() if new_prompt is not None else ""

    if new_prompt is None or len(new_prompt) < 10:
        if np.random.random() < PRINT_RATE:
            print("INVALID PROMPT")
            print(f"new_prompt: {new_prompt}")
            print(f"extra_info['performance']: {extra_info['performance']}")
            print(f"reward: {INVALID_REWARD - extra_info['performance']}")
            print("=" * 100)
        return {"score": INVALID_REWARD, "reward": INVALID_REWARD, "performance": float(extra_info["performance"])}

    if data_source == "bird":
        score = await _compute_bird_score(new_prompt, extra_info, reward_router_address)
    elif data_source == "mmlu":
        score = await _compute_mmlu_score(new_prompt, extra_info, reward_router_address)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    reward = float(score - extra_info["performance"])

    return {
        "score": float(score),
        "reward": reward,
        "performance": float(extra_info["performance"]),
    }
