"""Geometry helpers and a `Polyline` class supporting 2D and 3D lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from alpasim_grpc.v0 import common_pb2 as grpc_types
from alpasim_grpc.v0 import egodriver_pb2 as ego_grpc
from alpasim_utils.qvec import QVec
from scipy.spatial.transform import Rotation as R


class ProjectionResult(NamedTuple):
    """Result of projecting a point onto a polyline."""

    point: np.ndarray
    segment_idx: int
    distance_along: float


@dataclass
class Polyline:
    """Represents a spatial polyline as an ordered set of waypoints."""

    points: np.ndarray

    def __post_init__(self) -> None:
        """Normalize point storage to a 2D float array and validate dimensionality."""
        arr = np.asarray(self.points)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError(
                "Polyline points must have shape (N, D) where D is the spatial dimension"
            )
        if arr.shape[1] not in (2, 3):
            raise ValueError("Polyline supports only 2D or 3D points")
        self.points = arr.astype(float, copy=False)

    def __len__(self) -> int:
        """Return the number of waypoints in the polyline."""
        return self.points.shape[0]

    def __repr__(self) -> str:
        """Summarise the polyline for debugging/logging purposes."""
        length = self.total_length
        return (
            f"Polyline(n_points={len(self)}, dimension={self.dimension}, "
            f"length={length:.2f}m)"
        )

    @property
    def dimension(self) -> int:
        """Spatial dimensionality of the polyline (2 or 3)."""
        return self.points.shape[1]

    @property
    def waypoints(self) -> np.ndarray:
        """Reference to the underlying waypoint array."""
        return self.points

    @property
    def is_empty(self) -> bool:
        """Whether the polyline contains zero waypoints."""
        return len(self) == 0

    @classmethod
    def create_empty(cls, dimension: int = 3) -> "Polyline":
        """Factory for an empty polyline in the requested dimension."""
        if dimension not in (2, 3):
            raise ValueError("Polyline.create_empty supports only 2D or 3D")
        return cls(points=np.zeros((0, dimension), dtype=float))

    @property
    def total_length(self) -> float:
        """Total arc length of the polyline."""
        arc_lengths = self.arc_lengths()
        return float(arc_lengths[-1]) if arc_lengths.size else 0.0

    @property
    def segment_lengths(self) -> np.ndarray:
        """Euclidean distance between consecutive waypoints."""
        if len(self) < 2:
            return np.array([], dtype=float)
        deltas = self.points[1:] - self.points[:-1]
        return np.linalg.norm(deltas, axis=1)

    def arc_lengths(self) -> np.ndarray:
        """Cumulative arc lengths along the polyline."""
        segment_lengths = self.segment_lengths
        if segment_lengths.size == 0:
            return np.zeros(len(self.points), dtype=float)
        return np.concatenate(([0.0], np.cumsum(segment_lengths)))

    def positions_at(self, distances: np.ndarray) -> np.ndarray:
        """Interpolate positions at specific distances along the polyline."""
        distances = np.asarray(distances, dtype=float)
        arc_lengths = self.arc_lengths()

        unique_lengths, unique_indices = np.unique(arc_lengths, return_index=True)
        unique_points = self.points[unique_indices]

        if unique_lengths.size == 0:
            raise ValueError("Cannot interpolate along an empty polyline")

        if distances.size:
            min_distance = float(np.min(distances))
            max_distance = float(np.max(distances))
            if min_distance < float(unique_lengths[0]) or max_distance > float(
                unique_lengths[-1]
            ):
                raise ValueError(
                    "Requested distances must lie within the polyline arc length range"
                )

        dims = unique_points.shape[1]
        components = [
            np.interp(distances, unique_lengths, unique_points[:, axis])
            for axis in range(dims)
        ]
        return np.stack(components, axis=-1)

    def transform(self, transform_qvec: QVec) -> "Polyline":
        """Apply a rigid transform to the waypoints (3D only)."""
        if self.dimension != 3:
            raise ValueError("transform is only defined for 3D polylines")
        if transform_qvec.batch_size != ():
            raise ValueError("transform expects an unbatched QVec pose")
        rotation = R.from_quat(transform_qvec.quat).as_matrix()
        transformed = (rotation @ self.points.T).T + transform_qvec.vec3
        return Polyline(points=transformed)

    def project_point(self, point: np.ndarray) -> ProjectionResult:
        """Orthogonally project ``point`` onto the polyline segments."""
        point_arr = np.asarray(point, dtype=float)
        if point_arr.shape[-1] != self.dimension:
            raise ValueError(
                f"Point dimension {point_arr.shape[-1]} does not match polyline dimension {self.dimension}"
            )
        if len(self) == 0:
            raise ValueError("Cannot project onto empty polyline")
        if len(self) == 1:
            return ProjectionResult(self.points[0].copy(), 0, 0.0)

        min_distance = float("inf")
        best_projection = self.points[0]
        best_index = 0
        best_distance_along = 0.0

        for i in range(len(self) - 1):
            projected, distance_along, distance = self._project_point_to_segment(
                point_arr, self.points[i], self.points[i + 1]
            )
            if distance < min_distance:
                min_distance = distance
                best_projection = projected
                best_index = i
                best_distance_along = distance_along

        return ProjectionResult(best_projection, best_index, best_distance_along)

    @staticmethod
    def _project_point_to_segment(
        point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """Project a point onto a line segment, returning the closest point and distances."""
        segment_vec = segment_end - segment_start
        segment_length = np.linalg.norm(segment_vec)

        if segment_length < 1e-6:
            projected = segment_start
            return projected, 0.0, np.linalg.norm(projected - point)

        segment_dir = segment_vec / segment_length
        t = np.dot(point - segment_start, segment_dir)
        t = np.clip(t, 0.0, segment_length)
        projected = segment_start + t * segment_dir
        distance = np.linalg.norm(projected - point)

        return projected, t, distance

    def remaining_from_point(
        self, point: np.ndarray
    ) -> tuple["Polyline", ProjectionResult]:
        """Return the polyline remainder after projecting ``point``.

        Projects ``point`` orthogonally onto the current polyline and returns a tuple
        ``(remaining_polyline, projection_result)`` where ``remaining_polyline`` starts at
        the projected point (included as the first waypoint) and continues to the end of
        the original polyline. The method handles edge cases by returning:
        * an empty polyline with a zero projection when the source polyline is empty,
        * a shallow copy of the original polyline when it contains a single waypoint,
        * either the terminal waypoint, the last segment, or an empty polyline when the
          projected point lies on or beyond the final segment.
        """
        point_arr = np.asarray(point, dtype=float)

        if point_arr.shape[-1] != self.dimension:
            raise ValueError(
                "Point dimension must match the polyline dimension for projection"
            )

        if len(self) == 0:
            empty = Polyline.create_empty(dimension=self.dimension)
            projection = ProjectionResult(np.zeros(self.dimension, dtype=float), 0, 0.0)
            return empty, projection

        projection = self.project_point(point_arr)

        if len(self) == 1:
            return Polyline(points=self.points.copy()), projection

        if projection.segment_idx == len(self) - 2:
            # Projection lies on the final segment, handle the boundary cases explicitly.
            last_waypoint = self.points[-1]
            segment_vec = last_waypoint - self.points[-2]
            segment_length = np.linalg.norm(segment_vec)

            if segment_length > 0:
                segment_dir = segment_vec / segment_length
                t_unclamped = np.dot(point_arr - self.points[-2], segment_dir)
                if t_unclamped > segment_length + 1e-6:
                    # Point projects past the final waypoint; nothing remains.
                    return Polyline.create_empty(self.dimension), projection

            if np.allclose(projection.point, last_waypoint):
                # Exact projection onto the last waypoint; preserve dimensionality.
                return (
                    Polyline(points=last_waypoint.reshape(1, self.dimension)),
                    projection,
                )

            # Projection lies on the final segment, but not on the final waypoint
            remaining_points = np.vstack([projection.point, last_waypoint])
            return Polyline(points=remaining_points), projection

        # General case: keep the projection point and all downstream waypoints.
        remaining_points = np.vstack(
            [projection.point, self.points[projection.segment_idx + 1 :]]
        )
        return Polyline(points=remaining_points), projection

    def get_cumulative_distances_from_point(
        self, point: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Cumulative distances along the remainder of the polyline from a projected point."""
        cumulative_distances = self.arc_lengths()
        if cumulative_distances.size == 0:
            return np.array([], dtype=float), 0.0

        remaining_polyline, projection = self.remaining_from_point(point)

        distance_to_projection = (
            cumulative_distances[projection.segment_idx] + projection.distance_along
        )

        if remaining_polyline.is_empty:
            return np.array([], dtype=float), distance_to_projection

        cumulative_from_projection = remaining_polyline.arc_lengths()
        return cumulative_from_projection, distance_to_projection

    def resample_from_point(
        self, start_point: np.ndarray, spacing: float, n_points: int
    ) -> "Polyline":
        """Uniformly resample the remainder of the polyline after ``start_point``.

        The method projects ``start_point`` onto the polyline (using
        ``remaining_from_point``) and then samples up to ``n_points`` waypoints
        separated by ``spacing`` metres along the residual arc length. Sampling stops
        when the end of the polyline is reached, so the returned polyline may contain
        fewer than ``n_points`` points. Returns an empty polyline when the projection
        lies beyond the end of the source or when no residual arc remains.
        """
        remaining_polyline, _ = self.remaining_from_point(start_point)

        if remaining_polyline.is_empty:
            return Polyline.create_empty(dimension=self.dimension)

        sample_distances = np.arange(n_points, dtype=float) * float(spacing)
        arc_lengths = remaining_polyline.arc_lengths()

        valid_mask = sample_distances <= arc_lengths[-1]
        sample_distances = sample_distances[valid_mask]

        resampled_waypoints = remaining_polyline.positions_at(sample_distances)
        return Polyline(points=resampled_waypoints)

    def clip(
        self, start_idx: int | None = None, end_idx: int | None = None
    ) -> "Polyline":
        """Return a shallow copy over the waypoint slice ``[start_idx:end_idx]``."""
        return Polyline(points=self.points[start_idx:end_idx].copy())

    def append(self, other: "Polyline") -> "Polyline":
        """Concatenate another polyline with matching dimensionality."""
        if self.dimension != other.dimension:
            raise ValueError("Cannot append polylines of different dimensions")

        if self.is_empty:
            return Polyline(points=other.points.copy())
        if other.is_empty:
            return Polyline(points=self.points.copy())

        combined = np.concatenate([self.points, other.points], axis=0)
        return Polyline(points=combined)

    def downsample_with_min_distance(self, min_distance: float) -> None:
        """
        Downsample the polyline ensuring a minimum distance between waypoints. No interpolation
        is performed; waypoints are simply skipped as needed.

        Args:
            min_distance: Minimum Euclidean distance between consecutive waypoints.
        Returns:
            None: polyline is modified in place.
        """
        if len(self.points) < 2:
            return

        keep = [0]
        last = self.points[0]

        for i in range(1, len(self.points)):
            if np.linalg.norm(self.points[i] - last) >= min_distance:
                keep.append(i)
                last = self.points[i]
        self.points = self.points[keep]

    @classmethod
    def from_grpc(cls, grpc_route: ego_grpc.Route) -> "Polyline":
        """Construct a polyline from a gRPC route message."""
        waypoints = [[wp.x, wp.y, wp.z] for wp in grpc_route.waypoints]

        if not waypoints:
            return cls.create_empty(dimension=3)

        return cls(points=np.array(waypoints, dtype=float))

    def to_grpc_route(self, timestamp_us: int) -> ego_grpc.Route:
        """Convert the polyline to a gRPC route message (3D only)."""
        if self.dimension != 3:
            raise ValueError("to_grpc_route is only defined for 3D polylines")
        route = ego_grpc.Route(timestamp_us=timestamp_us)
        for wp in self.points:
            route.waypoints.append(
                grpc_types.Vec3(x=float(wp[0]), y=float(wp[1]), z=float(wp[2]))
            )
        return route

    def zero_out_z(self) -> "Polyline":
        """Return a new polyline with the z coordinate set to zero (3D only)."""
        if self.dimension != 3:
            raise ValueError("zero_out_z is only defined for 3D polylines")
        new_waypoints = self.points.copy()
        new_waypoints[:, 2] = 0.0
        return Polyline(points=new_waypoints)
