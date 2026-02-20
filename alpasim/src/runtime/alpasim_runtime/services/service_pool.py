# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Service pool implementation for service architecture.
"""

from __future__ import annotations

from asyncio import Queue
from typing import Any, Generic, List, Type, TypeVar

from alpasim_runtime.config import ScenarioConfig
from alpasim_runtime.services.service_base import ServiceBase

ServiceType = TypeVar("ServiceType", bound=ServiceBase)

# Number of skip services to create in skip mode.
# This should be at least as large as nr_concurrent_rollouts * nr_replicas for
# all of the other services.
NR_SKIP_SERVICES = 100


class ServicePool(Generic[ServiceType]):
    """
    Generic service pool that works with any ServiceBase-derived service.
    """

    def __init__(self, services: List[ServiceType]):
        self.queue: Queue[ServiceType] = Queue()
        self.services = services

        # Add all services to the queue
        for service in services:
            self.queue.put_nowait(service)

    @classmethod
    async def create_from_allocation(
        cls,
        service_class: Type[ServiceType],
        allocation: dict[str, int],  # address -> concurrency
        skip: bool,
        connection_timeout_s: int,
        **service_kwargs: Any,
    ) -> ServicePool[ServiceType]:
        """Create a service pool from pre-computed allocation.

        Args:
            service_class: The service class to instantiate
            allocation: Dict mapping address -> number of concurrent slots
            skip: Whether this service is in skip mode
            connection_timeout_s: Timeout for establishing connections
            **service_kwargs: Additional kwargs passed to service constructor
        """
        services: List[ServiceType] = []

        if skip:
            # Skip mode: create skip services
            for i in range(NR_SKIP_SERVICES):
                service = service_class(
                    "skip",
                    skip=True,
                    connection_timeout_s=connection_timeout_s,
                    id=i,
                    **service_kwargs,
                )
                services.append(service)
        else:
            # Real mode: create services based on allocation
            service_id = 0
            for address, n_concurrent in allocation.items():
                for i in range(n_concurrent):
                    service = service_class(
                        address,
                        skip=False,
                        connection_timeout_s=connection_timeout_s,
                        id=service_id,
                        **service_kwargs,
                    )
                    services.append(service)
                    service_id += 1

        return cls(services)

    async def get(self) -> ServiceType:
        """Get a service instance from the pool."""
        return await self.queue.get()

    async def put_back(self, service: ServiceType) -> None:
        """Return a service instance to the pool."""
        await self.queue.put(service)

    def get_number_of_services(self) -> int:
        """Get the number of services in this pool."""
        return len(self.services)

    def get_number_of_available_services(self) -> int:
        """Get the number of services in the queue."""
        return self.queue.qsize()

    async def find_scenario_incompatibilities(
        self, scenario: ScenarioConfig
    ) -> List[str]:
        """Check if services in this pool can handle the scenario."""
        if self.services:
            return await self.services[0].find_scenario_incompatibilities(scenario)
        return []
