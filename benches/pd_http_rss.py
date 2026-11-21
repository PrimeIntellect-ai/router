# /// script
# requires-python = ">=3.11"
# dependencies = ["aiohttp>=3.10,<4"]
# ///
"""Run on Linux with uv run benches/pd_http_rss.py /path/to/vllm-router."""

import argparse
import asyncio
import json
import socket
import subprocess
from contextlib import AsyncExitStack
from pathlib import Path

import aiohttp
from aiohttp import web

ROUTE = "/inference/v1/generate"


def rss_kib(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError(f"No VmRSS for PID {pid}")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def worker(stack, generate):
    app = web.Application(client_max_size=64 * 1024**2)

    async def health(request):
        return web.json_response({})

    async def models(request):
        return web.json_response({"data": [{"id": "bench"}]})

    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post(ROUTE, generate)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    stack.push_async_callback(runner.cleanup)
    await web.TCPSite(runner, "127.0.0.1", 0, backlog=1024).start()
    return f"http://127.0.0.1:{runner.addresses[0][1]}"


async def measure(args, stack):
    gate = asyncio.Event()
    gate.set()
    all_arrived = asyncio.Event()
    arrivals = 0

    async def prefill(request):
        await request.read()
        return web.json_response(
            {
                "choices": [{"token_ids": [1]}],
                "kv_transfer_params": {
                    "do_remote_prefill": True,
                    "remote_block_ids": [[1]],
                    "remote_engine_id": "p",
                    "remote_request_id": "p",
                    "remote_host": "localhost",
                    "remote_port": 5600,
                },
            }
        )

    async def decode(request):
        nonlocal arrivals
        await request.read()
        if not gate.is_set():
            arrivals += 1
            if arrivals == args.turns:
                all_arrived.set()
        await gate.wait()
        return web.json_response({"choices": [{"token_ids": [2]}]})

    prefill_url = await worker(stack, prefill)
    decode_url = await worker(stack, decode)
    port = free_port()
    router = await asyncio.create_subprocess_exec(
        str(args.binary.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--prometheus-port",
        str(free_port()),
        "--vllm-pd-disaggregation",
        "--prefill",
        prefill_url,
        "--decode",
        decode_url,
        "--policy",
        "sticky_least_loaded",
        "--pd-bidirectional-kv",
        stdout=subprocess.DEVNULL,
        stderr=None,
    )

    async def stop():
        gate.set()
        if router.returncode is None:
            router.kill()
        await router.wait()

    stack.push_async_callback(stop)
    session = await stack.enter_async_context(
        aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            raise_for_status=True,
        )
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(300):
        if router.returncode is not None:
            raise RuntimeError(f"Router exited with {router.returncode}")
        try:
            async with session.get(url + "/readiness"):
                break
        except aiohttp.ClientError:
            await asyncio.sleep(0.1)
    else:
        raise TimeoutError("Router did not become ready")

    payload = json.dumps(
        {
            "model": "bench",
            "token_ids": [(i * 7919) % 150000 for i in range(args.tokens)],
            "sampling_params": {"max_tokens": 2},
        },
        separators=(",", ":"),
    ).encode()

    async def turn(index):
        headers = {"Content-Type": "application/json", "x-session-id": str(index)}
        async with session.post(url + ROUTE, data=payload, headers=headers) as response:
            result = await response.json(content_type=None)
            if result["choices"][0]["token_ids"] != [2]:
                raise ValueError("Unexpected decode response")

    for index in range(8):
        await turn(-index - 1)
    warm = rss_kib(router.pid)
    gate.clear()
    async with asyncio.TaskGroup() as group:
        for index in range(args.turns):
            group.create_task(turn(index))
        await asyncio.wait_for(all_arrived.wait(), 90)
        await asyncio.sleep(1)
        held = rss_kib(router.pid)
        gate.set()
    await asyncio.sleep(1)
    print(
        json.dumps(
            dict(
                binary=args.binary.name,
                turns=arrivals,
                tokens=args.tokens,
                warm_kib=warm,
                held_kib=held,
                drained_kib=rss_kib(router.pid),
            )
        )
    )


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--turns", type=int, default=200)
    parser.add_argument("--tokens", type=int, default=100000)
    args = parser.parse_args()
    async with AsyncExitStack() as stack:
        await measure(args, stack)


if __name__ == "__main__":
    asyncio.run(main())
