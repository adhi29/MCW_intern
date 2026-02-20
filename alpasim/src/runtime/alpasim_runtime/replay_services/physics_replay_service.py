# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Physics replay service implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from alpasim_grpc.v0 import physics_pb2_grpc
from alpasim_runtime.replay_services.asl_reader import ASLReader

import grpc

from .base_replay_servicer import BaseReplayServicer

logger = logging.getLogger(__name__)


class PhysicsReplayService(BaseReplayServicer, physics_pb2_grpc.PhysicsServiceServicer):
    """Replay service for the physics service"""

    def __init__(self, asl_reader: ASLReader):
        super().__init__(asl_reader, "physics")

    def ground_intersection(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Return recorded constrained poses"""
        return self.validate_request("ground_intersection", request, context)
