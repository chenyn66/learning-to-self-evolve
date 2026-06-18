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
import multiprocessing
import os
import random
import time
from typing import Any

import aiohttp
import ray
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


async def _read_async_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
    if resp.status == 204 or (resp.content_length == 0):
        return {}

    try:
        return await resp.json(content_type=None)
    except Exception:
        try:
            text = await resp.text()
        except Exception:
            return {}
        return {
            "content_type": (resp.headers.get("Content-Type") or ""),
            "text": text,
        }


def launch_router_process(
    worker_urls: list[str],
):
    router_ip = ray.util.get_node_ip_address().strip("[]")
    router_port, _ = get_free_port(router_ip)
    router_address = (
        f"[{router_ip}]:{router_port}" if is_valid_ipv6_address(router_ip) else f"{router_ip}:{router_port}"
    )

    router_process = multiprocessing.Process(
        target=run_router,
        args=(
            router_ip,
            router_port,
            worker_urls,
        ),
    )
    router_process.daemon = True
    router_process.start()
    time.sleep(3)
    assert router_process.is_alive()

    logger.info(f"Router is running on {router_address}")
    return router_address, router_process


def run_router(router_ip: str, router_port: int, worker_urls: list[str]):
    router = NaiveRouter(worker_urls=worker_urls, verbose=False)
    uvicorn.run(router.app, host=router_ip, port=router_port, log_level="warning")


class NaiveRouter:
    def __init__(
        self,
        worker_urls: list[str],
        max_connections: int = 1024,
        timeout: int = 60,
        max_attempts: int = 3,
        retry_delay: float = 2.0,
        verbose: bool = False,
    ) -> None:
        """A minimal async load-balancing router."""
        self.verbose = verbose
        self.app = FastAPI()
        self.worker_urls = worker_urls
        self.request_counts = {url: 0 for url in worker_urls}

        self.max_connections = max(max_connections, int(os.environ.get("REWARD_ROUTER_MAX_CONNECTIONS", "1024")))
        derived_per_host_limit = max(1, self.max_connections // max(len(self.worker_urls), 1))
        # Keep per-host parallelism sufficiently high under large fanout.
        # Historical profile used `max_connections // 4`; preserve that as a floor.
        derived_per_host_limit = max(derived_per_host_limit, max(1, self.max_connections // 4))
        per_host_limit_raw = os.environ.get("REWARD_ROUTER_MAX_CONNECTIONS_PER_HOST", "").strip()
        if per_host_limit_raw:
            try:
                derived_per_host_limit = max(1, int(per_host_limit_raw))
            except ValueError:
                logger.warning(
                    "Invalid REWARD_ROUTER_MAX_CONNECTIONS_PER_HOST=%r; using derived value=%d",
                    per_host_limit_raw,
                    derived_per_host_limit,
                )
        self.max_connections_per_host = derived_per_host_limit
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

        self.app = FastAPI()

        # Register startup / shutdown hooks
        self.app.on_event("startup")(self._on_startup)
        self.app.on_event("shutdown")(self._on_shutdown)

        # Catch-all proxy route
        self.app.api_route("/{endpoint:path}", methods=["GET", "POST"])(self._make_async_request)

        # Placeholder for aiohttp client
        self.client = None

    async def _on_startup(self):
        """Initialize aiohttp client safely inside the event loop"""
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections_per_host,
            ttl_dns_cache=300,
            use_dns_cache=True,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=None)
        self.client = aiohttp.ClientSession(connector=connector, timeout=timeout)
        if self.verbose:
            logger.info(
                "[router] aiohttp client initialized with max_connections=%s limit_per_host=%s workers=%s",
                self.max_connections,
                self.max_connections_per_host,
                len(self.worker_urls),
            )

    async def _on_shutdown(self):
        """Gracefully close aiohttp client"""
        if self.client and not self.client.closed:
            await self.client.close()
            if self.verbose:
                logger.info("[router] aiohttp client closed")

    async def _make_async_request(self, request: Request, endpoint: str):
        """Proxy single request to a worker URL."""
        if not self.worker_urls:
            return JSONResponse(status_code=503, content={"error": "No available workers"})

        # Copy request data
        body = await request.body()
        headers = dict(request.headers)
        attempted_workers: set[str] = set()

        for attempt in range(self.max_attempts):
            worker_url = self._select_worker(exclude=attempted_workers)
            attempted_workers.add(worker_url)
            target_url = f"{worker_url}/{endpoint}"

            if self.verbose:
                logger.debug(f"[router] Forwarding request → {target_url}")

            # Send request to worker
            try:
                # NOTE: Always read the upstream response body (even on non-2xx).
                # This makes vLLM / OpenAI server errors debuggable (instead of just "400 Bad Request").
                async with self.client.request(request.method, target_url, data=body, headers=headers) as response:
                    output = await _read_async_response(response)
                    status = response.status

                if status >= 400:
                    logger.error(
                        f"Upstream HTTP error for {endpoint} (status={status}) from {target_url}: {output}"
                    )
                    # Don't crash the router process; return the upstream error back to the caller.
                    # This helps the client see the real vLLM error payload.
                    is_retryable = status == 429 or status >= 500
                    if is_retryable and attempt < self.max_attempts - 1:
                        await asyncio.sleep(self.retry_delay * (2**attempt))
                        continue
                    return JSONResponse(status_code=status, content=output)
                # Success: return upstream payload to the caller.
                return JSONResponse(status_code=status, content=output)
            except asyncio.TimeoutError:
                logger.warning(f"Async request to {endpoint} timed out (attempt {attempt + 1})")
            except aiohttp.ClientConnectorError:
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP error for {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {e}")
                if attempt == self.max_attempts - 1:
                    raise
            finally:
                # Ensure we don't permanently "leak" request counts on exceptions.
                # NOTE: Release exactly once per attempt; double-releasing breaks load balancing.
                # (Release is idempotent due to max(0, ...).)
                self._release_worker(worker_url)

            if attempt < self.max_attempts - 1:
                await asyncio.sleep(self.retry_delay * (2**attempt))

        return JSONResponse(
            status_code=503,
            content={"error": f"Failed to proxy request for {endpoint} after {self.max_attempts} attempts"},
        )

    def _select_worker(self, exclude: set[str] | None = None) -> str:
        """Select the least-loaded worker (simple round-robin by request count)."""
        excluded = exclude or set()
        candidates = [url for url in self.worker_urls if url not in excluded]
        if not candidates:
            candidates = self.worker_urls

        min_inflight = min(self.request_counts[url] for url in candidates)
        least_loaded = [url for url in candidates if self.request_counts[url] == min_inflight]
        url = random.choice(least_loaded)
        self.request_counts[url] += 1
        return url

    def _release_worker(self, url: str) -> None:
        """Mark worker as free after request completes."""
        self.request_counts[url] = max(0, self.request_counts[url] - 1)
