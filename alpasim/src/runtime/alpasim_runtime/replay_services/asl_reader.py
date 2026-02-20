# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
ASL (Alpasim Simulation Log) replay infrastructure for integration testing.

This module provides tools to record microservice interactions from ASL files
and replay them during testing to ensure refactoring preserves exact behavior.

:Warning: This module is meant only for simulations invovling a single instance
of each microservice, as its support for session management is not fully implemented.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Type, Union

from alpasim_grpc.v0 import common_pb2
from alpasim_grpc.v0.egodriver_pb2 import RolloutCameraImage
from alpasim_utils.logs import async_read_pb_log
from google.protobuf import json_format
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message

logger = logging.getLogger(__name__)


# List of fields that are expected to be different between runs
DYNAMIC_FIELDS = {
    "sessionUuid",
    "session_uuid",
    "randomSeed",
    "random_seed",
}

# Mapping of service names to their version field names in RolloutMetadata
SERVICE_VERSION_MAP = {
    "driver": "egodriver_version",
    "physics": "physics_version",
    "trafficsim": "traffic_version",
    "controller": "controller_version",
    "sensorsim": "sensorsim_version",
    "runtime": "runtime_version",
}


@dataclass
class ExchangeConfig:
    """Configuration for a request/response exchange."""

    method: str
    request_entry: str
    response: Union[
        str, Type, None
    ]  # Entry type name for paired response, or Type for direct response
    processor: Optional[Callable] = None  # Optional special processing

    @property
    def is_direct(self) -> bool:
        """Auto-detect if this is a direct exchange based on response type."""
        return not isinstance(self.response, str)


def _find_config_for_entry(
    entry_type: str,
) -> Tuple[Optional[str], Optional[ExchangeConfig]]:
    """Find the service and config for a given entry type."""
    for service, exchanges in SERVICE_EXCHANGES.items():
        for config in exchanges:
            # Check if it's a request entry
            if entry_type == config.request_entry:
                return service, config
            # Check if it's a response entry (for paired exchanges)
            if isinstance(config.response, str) and entry_type == config.response:
                return service, config
    return None, None


# Hierarchical service configuration
SERVICE_EXCHANGES = {
    "driver": [
        ExchangeConfig(
            method="drive", request_entry="driver_request", response="driver_return"
        ),
        ExchangeConfig(
            method="start_session",
            request_entry="driver_session_request",
            response=common_pb2.SessionRequestStatus,
            processor=lambda self, entry: self._process_driver_session(entry),
        ),
        ExchangeConfig(
            method="submit_image_observation",
            request_entry="driver_camera_image",
            response=common_pb2.Empty,
            processor=lambda self, entry: self.driver_images.append(entry),
        ),
        ExchangeConfig(
            method="submit_egomotion_observation",
            request_entry="driver_ego_trajectory",
            response=common_pb2.Empty,
        ),
        ExchangeConfig(
            method="submit_route",
            request_entry="route_request",
            response=common_pb2.Empty,
        ),
        ExchangeConfig(
            method="submit_recording_ground_truth",
            request_entry="ground_truth_request",
            response=common_pb2.Empty,
        ),
    ],
    "physics": [
        ExchangeConfig(
            method="ground_intersection",
            request_entry="physics_request",
            response="physics_return",
        ),
    ],
    "trafficsim": [
        ExchangeConfig(
            method="simulate",
            request_entry="traffic_request",
            response="traffic_return",
        ),
        ExchangeConfig(
            method="start_session",
            request_entry="traffic_session_request",
            response=common_pb2.SessionRequestStatus,
        ),
    ],
    "controller": [
        ExchangeConfig(
            method="run_controller_and_vehicle",
            request_entry="controller_request",
            response="controller_return",
        ),
    ],
    "sensorsim": [
        ExchangeConfig(
            method="render_rgb",
            request_entry="render_request",
            response=None,  # Special case - no response in ASL
        ),
        ExchangeConfig(
            method="get_available_cameras",
            request_entry="available_cameras_request",
            response="available_cameras_return",
        ),
    ],
}


class ASLReader:
    """Reader for ASL files with request/response pairing"""

    def __init__(
        self,
        asl_file_path: str,
    ) -> None:
        self.asl_file_path = asl_file_path
        # Store request/response pairs for each service.method combination
        self._exchanges: dict[str, list[tuple[Message, Optional[Message]]]] = {}
        # Track which exchanges have been consumed during replay
        self._consumed_indices: dict[str, set[int]] = {}
        # Store ASL metadata like configuration and rollout info
        self.asl_metadata: Dict[str, Any] = {}
        # Store camera images from driver for sensorsim correlation
        self.driver_images: List[RolloutCameraImage] = []

    def reset(self) -> None:
        """Reset the ASL reader for a fresh load"""
        self._exchanges.clear()
        self._consumed_indices.clear()
        self.asl_metadata.clear()
        self.driver_images.clear()

    async def load_exchanges(self) -> None:
        """Load and pair request/response exchanges from ASL file"""
        pending_requests: Dict[str, Deque[Any]] = defaultdict(deque)
        self.reset()

        logger.info(f"Loading exchanges from ASL file: {self.asl_file_path}")

        async for log_entry in async_read_pb_log(self.asl_file_path):
            entry_type = log_entry.WhichOneof("log_entry")

            # Handle special non-exchange entries
            if entry_type == "rollout_metadata":
                self.asl_metadata["rollout_metadata"] = log_entry.rollout_metadata
                continue
            elif entry_type in ["actor_poses", "egomotion_estimate_error"]:
                # Currently not used in replay
                continue

            # Find configuration for this entry type
            service, config = _find_config_for_entry(entry_type)
            if not config:
                raise AssertionError(f"Unknown entry type: {entry_type}")

            entry_data = getattr(log_entry, entry_type)

            # Handle based on whether it's a request or response
            if entry_type == config.request_entry:
                # This is a request
                if config.is_direct:
                    # Direct exchange - create immediately
                    response = config.response() if config.response else None
                    self._add_exchange(service, config.method, entry_data, response)
                else:
                    # Paired exchange - queue the request
                    queue_key = f"{service}.{config.method}"
                    pending_requests[queue_key].append(entry_data)
            else:
                # This is a response - pop matching request
                queue_key = f"{service}.{config.method}"
                request = pending_requests[queue_key].popleft()
                self._add_exchange(service, config.method, request, entry_data)

            # Run any special processing
            if config.processor:
                config.processor(self, entry_data)

        # Check for unmatched requests
        unmatched_requests = {
            key: len(queue) for key, queue in pending_requests.items() if queue
        }
        if unmatched_requests:
            for key, count in unmatched_requests.items():
                logger.warning(f"  {key}: {count} unmatched requests")
            raise AssertionError(f"Unmatched requests: {unmatched_requests}")

        # Log summary
        total_exchanges = sum(len(exchanges) for exchanges in self._exchanges.values())
        service_details = "\n  " + "\n  ".join(
            f"{service}: {len(exchanges)} exchanges"
            for service, exchanges in self._exchanges.items()
        )
        logger.info(
            "Loaded %d exchanges across %d services: %s",
            total_exchanges,
            len(self._exchanges),
            service_details,
        )
        logger.info(
            "Additional data: %d images",
            len(self.driver_images),
        )

    def _add_exchange(
        self, service: str, method: str, request: Any, response: Any
    ) -> None:
        """Add a request/response pair to the replay data"""
        key = f"{service}.{method}"
        if key not in self._exchanges:
            self._exchanges[key] = []
            self._consumed_indices[key] = set()

        self._exchanges[key].append((request, response))

    def _process_driver_session(
        self, session_request: common_pb2.SessionRequest
    ) -> None:
        """Special processor for driver session requests"""
        pass

    def get_exchange_summary(self) -> Dict[str, Any]:
        """Get summary of exchanges loaded and consumed"""
        summary = {}
        for key, exchanges in self._exchanges.items():
            consumed_set = self._consumed_indices.get(key, set())
            consumed_count = len(consumed_set)
            total = len(exchanges)

            result = {
                "total": total,
                "consumed": consumed_count,
                "remaining": total - consumed_count,
            }

            # Add unconsumed indices if any messages remain
            if consumed_count < total:
                all_indices = set(range(total))
                unconsumed_indices = sorted(all_indices - consumed_set)

                # Format unconsumed indices with ellipsis for long lists
                if len(unconsumed_indices) > 10:
                    displayed_indices = unconsumed_indices[:10]
                    result["unconsumed_indices"] = str(displayed_indices) + " ['...']"
                else:
                    result["unconsumed_indices"] = unconsumed_indices

                result["unconsumed_count"] = total - consumed_count

            summary[key] = result
        return summary

    def find_and_consume_matching_request(
        self, request: Any, service: str, method: str
    ) -> Optional[Tuple[int, Any]]:
        """Find a matching request and mark it as consumed.

        Args:
            request: The incoming request to match
            service: The service name
            method: The method name

        Returns:
            Tuple of (index, recorded_response) if found, None otherwise

        Raises:
            KeyError: If no exchanges are recorded for this service.method
        """
        key = f"{service}.{method}"
        exchanges = self._exchanges[key]

        # Find matching request
        match_result = self._find_matching_request(request, key, exchanges)

        if match_result:
            match_index, unused_recorded_response = match_result

            self._consumed_indices[key].add(match_index)

            return match_result

        return None

    def _find_matching_request(
        self, request: Any, key: str, exchanges: list, max_lookahead: int = 50
    ) -> Optional[Tuple[int, Any]]:
        """Find a matching request in the exchanges list, skipping consumed ones."""
        consumed = self._consumed_indices.get(key, set())

        # Get unconsumed indices within the lookahead window
        start_index = next(
            (i for i in range(len(exchanges)) if i not in consumed), None
        )
        if start_index is None:
            return None

        # Only check unconsumed messages within lookahead window
        for idx in range(start_index, min(start_index + max_lookahead, len(exchanges))):
            if idx in consumed:
                continue

            expected_request, recorded_response = exchanges[idx]
            if self.requests_match(request, expected_request):
                return (idx, recorded_response)

        return None

    def generate_no_match_error(
        self,
        request: Any,
        service: str,
        method: str,
    ) -> str:
        """Generate detailed error message when no matching request is found."""
        key = f"{service}.{method}"
        exchanges = self._exchanges.get(key, [])
        consumed = self._consumed_indices.get(key, set())
        unconsumed = [i for i in range(len(exchanges)) if i not in consumed]

        # Generate detailed error with unconsumed message info
        if unconsumed:
            # Show diff with first unconsumed message
            idx = unconsumed[0]
            expected_request, _ = exchanges[idx]
            diff = self.generate_diff(expected_request, request)[:1000]

            error_msg = (
                f"Request not found in {key}\n"
                f"Consumed indices: {sorted(consumed)}\n"
                f"Unconsumed indices: {unconsumed[:10]}...\n"  # Show first 10
                f"Diff with first unconsumed (index {idx}):\n{diff}"
            )
        else:
            error_msg = f"All {len(exchanges)} exchanges already consumed for {key}"

        return error_msg

    def get_driver_image_for_camera(
        self,
        camera_id: str,
        timestamp_us: int = 0,
    ) -> Optional[bytes]:
        """Get driver camera image data that corresponds to a render request.

        This is needed because we read in the camera images from the logged
        driver messages, not from the rendering returns (which aren't logged).

        Args:
            camera_id: The camera ID to match (logical name like "camera_front_wide_120fov")
            timestamp_us: The timestamp to match (frame_start_us from render request)

        Returns:
            The image bytes if found, None otherwise
        """

        # Find the image with matching camera index and closest timestamp
        best_match = None
        best_time_diff = float("inf")

        for image in self.driver_images:
            # Check if camera index matches
            if image.camera_image.logical_id == camera_id:
                # If timestamp is 0, return the first matching camera
                if timestamp_us == 0:
                    return image.camera_image.image_bytes
                # Otherwise, find the closest timestamp match
                time_diff = abs(image.camera_image.frame_start_us - timestamp_us)
                if time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_match = image

        if best_match is None:
            logger.error("No image found for camera %s", camera_id)
            return None

        assert best_time_diff == 0

        return best_match.camera_image.image_bytes

    def generate_diff(self, expected: Any, actual: Any) -> str:
        """Generate a detailed diff between expected and actual messages"""

        # Convert protobuf messages to dict for comparison

        expected_dict = MessageToDict(expected) if expected else {}
        actual_dict = MessageToDict(actual) if actual else {}

        expected_json = json.dumps(expected_dict, indent=2, sort_keys=True).splitlines()
        actual_json = json.dumps(actual_dict, indent=2, sort_keys=True).splitlines()

        diff = difflib.unified_diff(
            expected_json,
            actual_json,
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )

        return "\n".join(diff)

    def requests_match(self, actual: Any, expected: Any) -> bool:
        """Compare requests, ignoring dynamic fields that change between runs."""
        # If they're exactly equal, no need for complex comparison
        if actual == expected:
            return True

        # For protobuf messages, do field-by-field comparison
        if isinstance(actual, Message) and isinstance(expected, Message):
            # Convert to dicts for easier manipulation
            actual_dict = json_format.MessageToDict(actual)
            expected_dict = json_format.MessageToDict(expected)

            actual_normalized = _remove_dynamic_fields(actual_dict)
            expected_normalized = _remove_dynamic_fields(expected_dict)

            return actual_normalized == expected_normalized

        # For non-protobuf messages, use direct comparison
        return actual == expected

    def get_map_id(self) -> str:
        """Get the scene ID from the ASL metadata"""
        scene_id = self.asl_metadata["rollout_metadata"].session_metadata.scene_id
        pattern = r"^clipgt-([0-9a-fA-F-]{36})$"
        match = re.match(pattern, scene_id)
        if match is None:
            raise ValueError(
                f"Scene ID '{scene_id}' does not match expected pattern 'clipgt-<uuid>'"
            )
        return match.group(1)  # Map id (without clipgt prefix)

    def is_complete(self) -> bool:
        """Check if all messages consumed (ignoring skipped)"""
        for key, exchanges in self._exchanges.items():
            # Check if all exchanges for this key have been consumed
            consumed_set = self._consumed_indices.get(key, set())
            if len(consumed_set) < len(exchanges):
                return False
        return True

    def get_service_version(self, service_name: str) -> Optional[Message]:
        """Get the version information for a specific service from ASL metadata.

        Args:
            service_name: The service name (e.g., "driver", "physics", "trafficsim")

        Returns:
            The VersionId message for the service, or None if not found
        """
        return getattr(
            self.asl_metadata["rollout_metadata"].version_ids,
            SERVICE_VERSION_MAP[service_name],
        )


# Remove dynamic fields from both dicts
def _remove_dynamic_fields(d: Any) -> Any:
    if isinstance(d, dict):
        return {
            k: _remove_dynamic_fields(v)
            for k, v in d.items()
            if k not in DYNAMIC_FIELDS
        }
    elif isinstance(d, list):
        return [_remove_dynamic_fields(item) for item in d]
    return d
