# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Unit tests for worker allocation logic.
"""

from alpasim_runtime.worker.allocation import compute_instance_allocation
from alpasim_runtime.worker.ipc import ServiceAllocations


class TestComputeInstanceAllocation:
    """Tests for compute_instance_allocation function."""

    def test_even_distribution_perfect_partitioning(self):
        """4 addresses × 4 concurrent = 16 instances across 4 workers = 4 each."""
        addresses = ["A", "B", "C", "D"]
        n_concurrent = 4
        num_workers = 4

        # Each worker should get exactly one address with all 4 slots
        assert compute_instance_allocation(addresses, n_concurrent, 0, num_workers) == {
            "A": 4
        }
        assert compute_instance_allocation(addresses, n_concurrent, 1, num_workers) == {
            "B": 4
        }
        assert compute_instance_allocation(addresses, n_concurrent, 2, num_workers) == {
            "C": 4
        }
        assert compute_instance_allocation(addresses, n_concurrent, 3, num_workers) == {
            "D": 4
        }

    def test_more_workers_than_addresses(self):
        """2 addresses × 4 concurrent = 8 instances across 4 workers = 2 each."""
        addresses = ["A", "B"]
        n_concurrent = 4
        num_workers = 4

        # Each worker gets 2 slots
        assert compute_instance_allocation(addresses, n_concurrent, 0, num_workers) == {
            "A": 2
        }
        assert compute_instance_allocation(addresses, n_concurrent, 1, num_workers) == {
            "A": 2
        }
        assert compute_instance_allocation(addresses, n_concurrent, 2, num_workers) == {
            "B": 2
        }
        assert compute_instance_allocation(addresses, n_concurrent, 3, num_workers) == {
            "B": 2
        }

    def test_uneven_split_spans_addresses(self):
        """2 addresses × 4 concurrent = 8 instances across 3 workers."""
        addresses = ["A", "B"]
        n_concurrent = 4
        num_workers = 3

        # Worker 0 gets 3 (ceil), Worker 1 gets 3 (ceil), Worker 2 gets 2 (floor)
        # 8 / 3 = 2.67, so 2 workers get 3, 1 worker gets 2
        result_0 = compute_instance_allocation(addresses, n_concurrent, 0, num_workers)
        result_1 = compute_instance_allocation(addresses, n_concurrent, 1, num_workers)
        result_2 = compute_instance_allocation(addresses, n_concurrent, 2, num_workers)

        # Worker 0: slots 0-2 = [A₀, A₁, A₂] = {A: 3}
        assert result_0 == {"A": 3}
        # Worker 1: slots 3-5 = [A₃, B₀, B₁] = {A: 1, B: 2}
        assert result_1 == {"A": 1, "B": 2}
        # Worker 2: slots 6-7 = [B₂, B₃] = {B: 2}
        assert result_2 == {"B": 2}

        # Verify total matches
        total = sum(result_0.values()) + sum(result_1.values()) + sum(result_2.values())
        assert total == 8

    def test_single_worker_gets_everything(self):
        """Single worker should get all instances."""
        addresses = ["A", "B", "C"]
        n_concurrent = 4
        num_workers = 1

        result = compute_instance_allocation(addresses, n_concurrent, 0, num_workers)
        assert result == {"A": 4, "B": 4, "C": 4}

    def test_empty_addresses(self):
        """Empty addresses should return empty allocation."""
        assert compute_instance_allocation([], 4, 0, 4) == {}

    def test_zero_concurrent(self):
        """Zero concurrent should return empty allocation."""
        assert compute_instance_allocation(["A", "B"], 0, 0, 4) == {}

    def test_zero_workers(self):
        """Zero workers should return empty allocation."""
        assert compute_instance_allocation(["A", "B"], 4, 0, 0) == {}

    def test_more_workers_than_instances(self):
        """Some workers may get zero instances."""
        addresses = ["A"]
        n_concurrent = 2
        num_workers = 4

        # Only 2 instances total, 4 workers
        assert compute_instance_allocation(addresses, n_concurrent, 0, num_workers) == {
            "A": 1
        }
        assert compute_instance_allocation(addresses, n_concurrent, 1, num_workers) == {
            "A": 1
        }
        assert (
            compute_instance_allocation(addresses, n_concurrent, 2, num_workers) == {}
        )
        assert (
            compute_instance_allocation(addresses, n_concurrent, 3, num_workers) == {}
        )

    def test_single_address_distributed(self):
        """Single address with 4 concurrent across 2 workers = 2 each."""
        addresses = ["A"]
        n_concurrent = 4
        num_workers = 2

        assert compute_instance_allocation(addresses, n_concurrent, 0, num_workers) == {
            "A": 2
        }
        assert compute_instance_allocation(addresses, n_concurrent, 1, num_workers) == {
            "A": 2
        }

    def test_preserves_total_capacity(self):
        """Sum of all worker allocations should equal total instances."""
        addresses = ["A", "B", "C", "D", "E"]
        n_concurrent = 3
        num_workers = 7

        total = sum(
            sum(
                compute_instance_allocation(
                    addresses, n_concurrent, worker_id, num_workers
                ).values()
            )
            for worker_id in range(num_workers)
        )
        assert total == len(addresses) * n_concurrent  # 5 * 3 = 15


class TestServiceAllocations:
    """Tests for ServiceAllocations dataclass."""

    def test_get_capacity_returns_minimum(self):
        """get_capacity should return minimum across all services."""
        alloc = ServiceAllocations(
            driver={"addr1": 4},
            sensorsim={"addr2": 3, "addr3": 2},
            physics={"addr4": 10},
            trafficsim={"addr5": 5},
            controller={"addr6": 2, "addr7": 2},
        )
        # driver=4, sensorsim=5, physics=10, trafficsim=5, controller=4
        assert alloc.get_capacity() == 4

    def test_get_capacity_empty(self):
        """Empty allocations should return 0."""
        alloc = ServiceAllocations()
        assert alloc.get_capacity() == 0

    def test_get_capacity_ignores_zero(self):
        """get_capacity should ignore services with zero allocation (skip mode)."""
        alloc = ServiceAllocations(
            driver={"addr1": 4},
            sensorsim={},  # skip mode
            physics={"addr4": 10},
            trafficsim={},  # skip mode
            controller={"addr6": 2},
        )
        # Only non-empty: driver=4, physics=10, controller=2
        assert alloc.get_capacity() == 2

    def test_defaults_to_empty_dicts(self):
        """Default allocations should be empty dicts."""
        alloc = ServiceAllocations()
        assert alloc.driver == {}
        assert alloc.sensorsim == {}
        assert alloc.physics == {}
        assert alloc.trafficsim == {}
        assert alloc.controller == {}
