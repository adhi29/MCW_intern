# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Unit tests for worker IPC types.
"""

import pickle
from multiprocessing import Queue
from unittest import mock

from alpasim_runtime.config import ScenarioConfig
from alpasim_runtime.worker.ipc import (
    JOB_POLL_TIMEOUT_S,
    RESULT_POLL_TIMEOUT_S,
    SHUTDOWN_SENTINEL,
    JobResult,
    RolloutJob,
    ServiceAllocations,
    _ShutdownSentinel,
    poll_job_queue,
    poll_result_queue,
)


class TestRolloutJob:
    """Tests for RolloutJob dataclass."""

    def test_creation(self):
        """Test basic creation."""
        scenario = ScenarioConfig(scene_id="test-scene", n_sim_steps=100, n_rollouts=5)
        job = RolloutJob(job_id="test-123", scenario=scenario, seed=42)

        assert job.job_id == "test-123"
        assert job.scenario.scene_id == "test-scene"
        assert job.seed == 42

    def test_pickling(self):
        """RolloutJob should be picklable for multiprocessing Queue."""
        scenario = ScenarioConfig(scene_id="test-scene", n_sim_steps=100, n_rollouts=5)
        job = RolloutJob(job_id="test-123", scenario=scenario, seed=42)

        # Pickle and unpickle
        pickled = pickle.dumps(job)
        unpickled = pickle.loads(pickled)

        assert unpickled.job_id == job.job_id
        assert unpickled.scenario.scene_id == job.scenario.scene_id
        assert unpickled.seed == job.seed


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = JobResult(
            job_id="test-123",
            success=True,
            error=None,
            error_traceback=None,
            rollout_uuid="uuid-456",
        )

        assert result.success is True
        assert result.error is None
        assert result.rollout_uuid == "uuid-456"

    def test_failure_result(self):
        """Test failure result."""
        result = JobResult(
            job_id="test-123",
            success=False,
            error="Something went wrong",
            error_traceback="Traceback...",
            rollout_uuid=None,
        )

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.error_traceback == "Traceback..."

    def test_pickling(self):
        """JobResult should be picklable for multiprocessing Queue."""
        result = JobResult(
            job_id="test-123",
            success=False,
            error="Error message",
            error_traceback="Full traceback",
            rollout_uuid="uuid-789",
        )

        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)

        assert unpickled.job_id == result.job_id
        assert unpickled.success == result.success
        assert unpickled.error == result.error


class TestShutdownSentinel:
    """Tests for shutdown sentinel."""

    def test_singleton_identity(self):
        """SHUTDOWN_SENTINEL should be a singleton-like object."""
        assert isinstance(SHUTDOWN_SENTINEL, _ShutdownSentinel)

    def test_distinct_from_none(self):
        """Sentinel should be distinct from None."""
        assert SHUTDOWN_SENTINEL is not None

    def test_pickling(self):
        """Sentinel should be picklable."""
        pickled = pickle.dumps(SHUTDOWN_SENTINEL)
        unpickled = pickle.loads(pickled)
        # After unpickling it's a new instance but same type
        assert isinstance(unpickled, _ShutdownSentinel)


class TestServiceAllocations:
    """Tests for ServiceAllocations dataclass."""

    def test_pickling(self):
        """ServiceAllocations should be picklable."""
        alloc = ServiceAllocations(
            driver={"addr1": 4},
            sensorsim={"addr2": 3},
            physics={"addr3": 2},
            trafficsim={"addr4": 1},
            controller={"addr5": 5},
        )

        pickled = pickle.dumps(alloc)
        unpickled = pickle.loads(pickled)

        assert unpickled.driver == alloc.driver
        assert unpickled.sensorsim == alloc.sensorsim


class TestQueueHelpers:
    """Tests for queue helper functions."""

    def test_poll_job_queue_returns_job(self):
        """Should return job when available."""
        queue: Queue = Queue()
        scenario = ScenarioConfig(scene_id="test", n_sim_steps=100, n_rollouts=1)
        job = RolloutJob(job_id="123", scenario=scenario, seed=1)
        queue.put(job)

        result = poll_job_queue(queue)
        assert result is not None
        assert result.job_id == "123"

    def test_poll_job_queue_returns_none_on_timeout(self):
        """Should return None on timeout."""
        queue: Queue = Queue()

        with mock.patch("alpasim_runtime.worker.ipc.JOB_POLL_TIMEOUT_S", 0.01):
            result = poll_job_queue(queue)
        assert result is None

    def test_poll_job_queue_returns_sentinel(self):
        """Should return sentinel when it's in the queue."""
        queue: Queue = Queue()
        queue.put(SHUTDOWN_SENTINEL)

        result = poll_job_queue(queue)
        assert isinstance(result, _ShutdownSentinel)

    def test_poll_result_queue_returns_result(self):
        """Should return result when available."""
        queue: Queue = Queue()
        result = JobResult(
            job_id="123",
            success=True,
            error=None,
            error_traceback=None,
            rollout_uuid="456",
        )
        queue.put(result)

        got = poll_result_queue(queue)
        assert got is not None
        assert got.job_id == "123"

    def test_poll_result_queue_returns_none_on_timeout(self):
        """Should return None on timeout."""
        queue: Queue = Queue()

        with mock.patch("alpasim_runtime.worker.ipc.RESULT_POLL_TIMEOUT_S", 0.01):
            result = poll_result_queue(queue)
        assert result is None


class TestTimeoutConstants:
    """Tests for timeout constants."""

    def test_job_poll_timeout(self):
        """Job poll timeout should be reasonable."""
        assert JOB_POLL_TIMEOUT_S > 0
        assert JOB_POLL_TIMEOUT_S == 10.0

    def test_result_poll_timeout(self):
        """Result poll timeout should be reasonable."""
        assert RESULT_POLL_TIMEOUT_S > 0
        assert RESULT_POLL_TIMEOUT_S == 30.0
