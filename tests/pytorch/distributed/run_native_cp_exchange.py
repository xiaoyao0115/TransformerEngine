# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed correctness probe for NativeCPTransport.exchange."""

import os

import torch
import torch.distributed as dist

from transformer_engine.pytorch.attention.native_cp_transport import (
    destroy_native_cp_transport,
    initialize_native_cp_transport,
)


def _bounded_peer_ring(world_size: int) -> tuple[int, ...]:
    return tuple([0, *range(1, world_size, 2), *reversed(range(2, world_size, 2))])


def _payload_bytes(message_bytes: int) -> int:
    return ((message_bytes + 255) // 256) * 256 + message_bytes


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 5:
        raise RuntimeError(f"This probe requires five ranks, got {world_size}")

    # Force lazy ProcessGroupNCCL initialization before measuring the transport,
    # so the arena measurement does not include the communicator warmup.
    torch.cuda.synchronize()
    free_before_nccl, _ = torch.cuda.mem_get_info()
    dist.barrier(device_ids=[local_rank])
    torch.cuda.synchronize()
    free_after_nccl, _ = torch.cuda.mem_get_info()
    nccl_warmup_physical_growth = free_before_nccl - free_after_nccl

    # Qwen3.5 TP1 GDN state: [num_value_heads, K, V + K], always fp32 in FLA.
    state = torch.full((32, 128, 256), rank, device="cuda", dtype=torch.float32)
    received = torch.empty_like(state)
    torch.cuda.synchronize()
    allocated_before_arena = torch.cuda.memory_allocated()
    reserved_before_arena = torch.cuda.memory_reserved()
    free_before_arena, _ = torch.cuda.mem_get_info()
    transport = initialize_native_cp_transport(dist.group.WORLD, _payload_bytes(state.nbytes))
    torch.cuda.synchronize()
    arena_allocated_growth = torch.cuda.memory_allocated() - allocated_before_arena
    arena_reserved_growth = torch.cuda.memory_reserved() - reserved_before_arena
    free_after_arena, _ = torch.cuda.mem_get_info()
    arena_physical_growth = free_before_arena - free_after_arena
    ranks = _bounded_peer_ring(world_size)
    cp_rank = ranks.index(rank)

    received_fwd = transport.exchange(
        state,
        ranks[(cp_rank + 1) % world_size],
        ranks[(cp_rank - 1) % world_size],
        channel=0,
        out=received,
    )
    assert received_fwd.data_ptr() == received.data_ptr()
    torch.testing.assert_close(
        received_fwd,
        torch.full_like(state, ranks[(cp_rank - 1) % world_size]),
        rtol=0,
        atol=0,
    )

    received_bwd = transport.exchange(
        state,
        ranks[(cp_rank - 1) % world_size],
        ranks[(cp_rank + 1) % world_size],
        channel=1,
        out=received,
    )
    assert received_bwd.data_ptr() == received.data_ptr()
    torch.testing.assert_close(
        received_bwd,
        torch.full_like(state, ranks[(cp_rank + 1) % world_size]),
        rtol=0,
        atol=0,
    )

    # Measure a steady-state exchange separately from correctness assertions,
    # whose expected-value tensors would otherwise pollute the allocator peak.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before_exchange = torch.cuda.memory_allocated()
    free_before_exchange, _ = torch.cuda.mem_get_info()
    transport.exchange(
        state,
        ranks[(cp_rank + 1) % world_size],
        ranks[(cp_rank - 1) % world_size],
        channel=0,
        out=received,
    )
    torch.cuda.synchronize()
    peak_exchange_growth = torch.cuda.max_memory_allocated() - allocated_before_exchange
    live_exchange_growth = torch.cuda.memory_allocated() - allocated_before_exchange
    free_after_exchange, _ = torch.cuda.mem_get_info()
    physical_exchange_growth = free_before_exchange - free_after_exchange

    if rank == 0:
        print(
            "NATIVE_CP_EXCHANGE_PASS "
            f"world_size={world_size} ring={ranks} message_bytes={state.nbytes} "
            f"arena_bytes={transport.payload_bytes} "
            f"nccl_warmup_physical_growth={nccl_warmup_physical_growth} "
            f"arena_allocated_growth={arena_allocated_growth} "
            f"arena_reserved_growth={arena_reserved_growth} "
            f"arena_physical_growth={arena_physical_growth} "
            f"peak_exchange_growth={peak_exchange_growth} "
            f"live_exchange_growth={live_exchange_growth} "
            f"physical_exchange_growth={physical_exchange_growth}"
        )
    destroy_native_cp_transport(dist.group.WORLD)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
