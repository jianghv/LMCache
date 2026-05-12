# lmcache/v1/internal_api_server/memory/memory_api.py

"""
记忆系统 API。

上层记忆系统通过此 REST API 与 LMCache 交互：
- POST /memory/prefetch  : 预取 KV cache 到 DDR
- POST /memory/evict     : 驱逐 KV cache

chunk_hash 通过 vLLM 响应中的 kv_transfer_params 返回给记忆系统，
无需额外接口。

设计说明：
  vLLM v1 多进程模式下，scheduler 进程不创建 LMCacheEngine（节省内存），
  storage_manager 仅在 worker 进程中存在。当 scheduler 进程的 API server
  收到 /memory/* 请求时，会自动转发到 worker 进程的 API server。
"""

from typing import List, Optional, Tuple
import asyncio
import json
import urllib.request

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine

logger = init_logger(__name__)

router = APIRouter()


def _get_engine(
    request: Request,
) -> Tuple[Optional[LMCacheEngine], Optional[PlainTextResponse], bool]:
    """
    获取 LMCacheEngine。

    返回 (engine, error_response, forward_to_workers):
    - engine, None, False: 直接使用 engine
    - None, error_response, False: 返回错误
    - None, None, True: scheduler 进程，需转发到 worker
    """
    adapter = request.app.state.lmcache_adapter
    engine = getattr(adapter, "lmcache_engine", None)
    if engine:
        return engine, None, False

    port_offset = getattr(
        request.app.state, "internal_api_server_port_offset", None
    )
    if port_offset == 0:
        return None, None, True

    return None, PlainTextResponse(
        content=json.dumps({"error": "LMCache engine not available"}),
        media_type="application/json",
        status_code=503,
    ), False


def _hashes_to_keys(
    engine: LMCacheEngine, chunk_hashes: List[str]
) -> List[CacheEngineKey]:
    world_size = (
        1 if engine.save_only_first_rank
        else engine.metadata.world_size
    )
    return [
        CacheEngineKey(
            model_name=engine.metadata.model_name,
            world_size=world_size,
            worker_id=engine.metadata.worker_id,
            chunk_hash=int(h, 16),
            dtype=engine.metadata.kv_dtype,
        )
        for h in chunk_hashes
    ]


def _get_worker_ports(request: Request) -> List[int]:
    """获取所有 worker 的 API server 端口。
    
    非 MLA 模型 (save_only_first_rank=False) 中，每个 TP worker 各自存储
    KV cache 的不同 head 分片，因此需要广播到所有 worker。
    MLA 模型 (save_only_first_rank=True) 只有 worker 0 存储，但广播到所有
    worker 也是安全的（其余 worker 找不到数据，返回空结果）。
    """
    port_start = request.app.state.internal_api_server_port_start
    adapter = request.app.state.lmcache_adapter
    metadata = getattr(adapter, "lmcache_engine_metadata", None)
    world_size = metadata.world_size if metadata else 1
    return [port_start + 1 + i for i in range(world_size)]


async def _forward_to_all_workers(
    request: Request, path: str
) -> PlainTextResponse:
    """将请求转发到所有 worker 的 API server，返回合并结果。"""
    worker_ports = _get_worker_ports(request)
    body = await request.body()

    def _do_request(port: int):
        try:
            req = urllib.request.Request(
                f"http://localhost:{port}{path}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return port, resp.status, json.loads(resp.read())
        except Exception as e:
            return port, 503, {"error": str(e)}

    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, _do_request, port)
        for port in worker_ports
    ]
    results = await asyncio.gather(*tasks)

    failures = [(p, s, r) for p, s, r in results if s != 200]
    all_ok = len(failures) == 0

    if all_ok:
        first_result = results[0][2] if results else {}
        return PlainTextResponse(
            content=json.dumps(first_result),
            media_type="application/json",
            status_code=200,
        )

    logger.error(
        "Forward %s: %d/%d workers failed: %s",
        path, len(failures), len(worker_ports),
        {p: r for p, _, r in failures},
    )
    return PlainTextResponse(
        content=json.dumps({
            "error": "Some workers failed to process request",
            "failures": {str(p): r for p, _, r in failures},
        }),
        media_type="application/json",
        status_code=500,
    )


# ==================== prefetch ====================

@router.post("/memory/prefetch")
async def prefetch(request: Request):
    """
    预取 KV cache 到 DDR。

    记忆系统在推理前调用，将已命中的 chunk 提前加载到 CPU 内存。

    Request body (JSON):
    {
        "chunk_hashes": ["a1b2c3...", "d4e5f6..."],
        "lookup_id": "req_001"
    }

    Response:
    {
        "status": "prefetch_started",
        "lookup_id": "req_001",
        "num_chunks": 2
    }
    """
    engine, err, forward = _get_engine(request)
    if forward:
        return await _forward_to_all_workers(request, "/memory/prefetch")
    if err:
        return err

    if engine.storage_manager is None:
        return PlainTextResponse(
            content=json.dumps({"error": "Storage manager not available"}),
            media_type="application/json",
            status_code=503,
        )

    try:
        body = await request.json()
        chunk_hashes = body.get("chunk_hashes", [])
        lookup_id = body.get("lookup_id")

        if not chunk_hashes or not lookup_id:
            return PlainTextResponse(
                content=json.dumps({"error": "chunk_hashes and lookup_id are required"}),
                media_type="application/json",
                status_code=400,
            )

        keys = _hashes_to_keys(engine, chunk_hashes)
        chunk_size = engine.config.chunk_size
        cum_chunk_lengths = [i * chunk_size for i in range(len(keys) + 1)]

        asyncio.run_coroutine_threadsafe(
            engine.storage_manager.async_lookup_and_prefetch(
                lookup_id=lookup_id,
                keys=keys,
                cum_chunk_lengths=cum_chunk_lengths,
                search_range=engine.retrieve_locations,
                pin=True,
                log_timing=True,
            ),
            engine.storage_manager.loop,
        )

        return PlainTextResponse(
            content=json.dumps({
                "status": "prefetch_started",
                "lookup_id": lookup_id,
                "num_chunks": len(keys),
            }),
            media_type="application/json",
        )

    except Exception as e:
        logger.error("prefetch failed: %s", e)
        return PlainTextResponse(
            content=json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )


# ==================== evict ====================

@router.post("/memory/evict")
async def evict(request: Request):
    """
    驱逐 KV cache。

    记忆系统判断某些 chunk 不再需要时调用。

    Request body (JSON):
    {
        "chunk_hashes": ["a1b2c3..."],
        "locations": ["LocalCPUBackend"]   // 可选，默认所有 backend
    }

    Response:
    {
        "status": "success",
        "num_evicted": 1
    }
    """
    engine, err, forward = _get_engine(request)
    if forward:
        return await _forward_to_all_workers(request, "/memory/evict")
    if err:
        return err

    if engine.storage_manager is None:
        return PlainTextResponse(
            content=json.dumps({"error": "Storage manager not available"}),
            media_type="application/json",
            status_code=503,
        )

    try:
        body = await request.json()
        chunk_hashes = body.get("chunk_hashes", [])
        locations = body.get("locations")  # None = 所有 backend

        if not chunk_hashes:
            return PlainTextResponse(
                content=json.dumps({"error": "chunk_hashes is required"}),
                media_type="application/json",
                status_code=400,
            )

        keys = _hashes_to_keys(engine, chunk_hashes)
        num_evicted = engine.storage_manager.batched_remove(keys, locations=locations)

        return PlainTextResponse(
            content=json.dumps({
                "status": "success",
                "num_evicted": num_evicted,
            }),
            media_type="application/json",
        )

    except Exception as e:
        logger.error("evict failed: %s", e)
        return PlainTextResponse(
            content=json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )
