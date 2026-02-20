# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Instance-based service distribution across workers.

Total instances (addresses × n_concurrent) are distributed as a flat pool,
with each worker getting a contiguous slice, such that ideally one worker
repeatedly uses the same address as we will also try to assign multiple
rollouts of the same scenario to the same worker.
"""

from alpasim_runtime.config import SimulatorConfig
from alpasim_runtime.worker.ipc import ServiceAllocations


def compute_instance_allocation(
    addresses: list[str],
    n_concurrent: int,
    worker_id: int,
    num_workers: int,
) -> dict[str, int]:
    """
    Distribute total instances (addresses × n_concurrent) across workers.
    Returns a dict mapping address -> concurrency for this worker.

    Instances are assigned contiguously, so workers tend to get
    consecutive addresses (good for cache locality).

    Args:
        addresses: List of service addresses
        n_concurrent: Number of concurrent slots per address
        worker_id: This worker's ID (0-indexed)
        num_workers: Total number of workers

    Returns:
        Dict mapping address -> number of concurrent slots for this worker.
        Empty dict if worker has no allocation for this service.
    """
    total_instances = len(addresses) * n_concurrent

    if total_instances == 0 or num_workers == 0:
        return {}

    # Balanced distribution: some workers get ceil, others get floor
    base = total_instances // num_workers
    remainder = total_instances % num_workers

    # If total_instances / num_workers is not an integer, we distribute
    # the remainder evenly across the first `remainder` workers, i.e.
    # The workers 0, ..., remainder - 1 get `(base + 1)` instances and
    # The workers remainder, ..., num_workers - 1 get `base` instances
    if worker_id < remainder:
        start = worker_id * (base + 1)
        count = base + 1
    else:
        start = remainder * (base + 1) + (worker_id - remainder) * base
        count = base

    if count == 0:
        return {}

    end = start + count

    # Map instance range [start, end) to {address: concurrency}
    # Example: 2 addresses, n_concurrent=4
    # 3 Workers, 3 + 3 + 2 = 8 instances
    # Worker 2 gets instances [3, 6)
    #   Address 0: instances [0, 4) → overlap [3, 4) → 1 slot
    #   Address 1: instances [4, 8) → overlap [4, 6) → 2 slots
    #   Result: {addr0: 1, addr1: 2}
    # Worker 0 gets instances [7, 8]
    allocation: dict[str, int] = {}
    for addr_idx, addr in enumerate(addresses):
        addr_start = addr_idx * n_concurrent
        addr_end = addr_start + n_concurrent

        # Intersection of [start, end) with [addr_start, addr_end)
        overlap_start = max(start, addr_start)
        overlap_end = min(end, addr_end)

        if overlap_start < overlap_end:
            allocation[addr] = overlap_end - overlap_start

    return allocation


def compute_worker_allocations(
    config: SimulatorConfig,
    num_workers: int,
) -> list[ServiceAllocations]:
    """
    Compute service allocations for all workers.

    Args:
        config: Full simulator config with network addresses and endpoint settings
        num_workers: Number of workers to distribute across

    Returns:
        List of ServiceAllocations, one per worker (indexed by worker_id)
    """
    allocations = []

    for worker_id in range(num_workers):
        alloc = ServiceAllocations(
            driver=compute_instance_allocation(
                config.network.driver.addresses,
                config.user.endpoints.driver.n_concurrent_rollouts,
                worker_id,
                num_workers,
            ),
            sensorsim=compute_instance_allocation(
                config.network.sensorsim.addresses,
                config.user.endpoints.sensorsim.n_concurrent_rollouts,
                worker_id,
                num_workers,
            ),
            physics=compute_instance_allocation(
                config.network.physics.addresses,
                config.user.endpoints.physics.n_concurrent_rollouts,
                worker_id,
                num_workers,
            ),
            trafficsim=compute_instance_allocation(
                config.network.trafficsim.addresses,
                config.user.endpoints.trafficsim.n_concurrent_rollouts,
                worker_id,
                num_workers,
            ),
            controller=compute_instance_allocation(
                config.network.controller.addresses,
                config.user.endpoints.controller.n_concurrent_rollouts,
                worker_id,
                num_workers,
            ),
        )
        allocations.append(alloc)

    return allocations
