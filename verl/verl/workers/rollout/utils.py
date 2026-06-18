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
import asyncio
import logging
import os

import uvicorn
from fastapi import FastAPI

from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    """Start a uvicorn server in the current event loop.

    Notes:
    - We start `uvicorn.Server.serve()` in a background task so this coroutine
      can return without blocking the actor/event loop.
    - We bind a free port first and pass the bound socket into uvicorn to avoid
      port races (another process grabbing the port between selection and bind).
    """

    server_port: int | None = None
    server_task: asyncio.Task | None = None

    for i in range(max_retries):
        sock = None
        try:
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args

            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)

            # Run the server in background.
            server_task = asyncio.create_task(server.serve(sockets=[sock]))

            # Wait for the server to become ready (or fail fast).
            for _ in range(200):  # ~20s
                if server.started:
                    break
                if server_task.done():
                    # Propagate any startup exception.
                    server_task.result()
                await asyncio.sleep(0.1)

            if not server.started:
                raise RuntimeError(f"Uvicorn failed to start on {server_address}:{server_port}")

            logger.info(f"HTTP server started on port {server_port}")
            return server_port, server_task
        except (OSError, SystemExit, RuntimeError) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
            try:
                if server_task is not None and not server_task.done():
                    server_task.cancel()
            except Exception:
                pass
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
        except Exception as e:
            logger.exception(f"Unexpected error while starting HTTP server (try {i}): {e}")
            try:
                if server_task is not None and not server_task.done():
                    server_task.cancel()
            except Exception:
                pass
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
        finally:
            # Ensure we don't leak a bound socket on failure; on success uvicorn owns it.
            # (If the server started, `sock` is already passed to uvicorn and should not be closed here.)
            pass

    logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
    os._exit(-1)
