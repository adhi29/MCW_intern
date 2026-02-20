import numpy as np
import pytest
from alpasim_utils.polyline import Polyline
from alpasim_utils.qvec import QVec
from alpasim_utils.trajectory import Trajectory
from numpy.testing import assert_almost_equal


@pytest.fixture
def simple_polyline() -> Polyline:
    """Create a simple square polyline for testing."""

    return Polyline(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )
    )


def test_compute_arc_lengths_basic():
    points = np.array([[0, 0, 0], [3, 4, 0], [3, 8, 0]], dtype=float)
    arc_lengths = Polyline(points).arc_lengths()

    assert arc_lengths.tolist() == [0.0, 5.0, 9.0]


def test_interpolate_along_path_handles_stationary_points():
    points = np.array([[0, 0, 0], [0, 0, 0], [10, 0, 0]], dtype=float)
    distances = np.array([0.0, 5.0, 10.0])

    interpolated = Polyline(points).positions_at(distances)

    assert np.allclose(interpolated[0], [0, 0, 0])
    assert np.allclose(interpolated[1], [5, 0, 0])
    assert np.allclose(interpolated[2], [10, 0, 0])


def test_project_point_to_polyline_mid_segment():
    points = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], dtype=float)
    projection = Polyline(points).project_point(np.array([4, 3, 0], dtype=float))

    assert projection.segment_idx == 0
    assert projection.distance_along == pytest.approx(4.0)
    assert np.allclose(projection.point, [4, 0, 0])


def test_remaining_polyline_from_point_at_route_end():
    points = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    remaining_polyline, projection = Polyline(points).remaining_from_point(
        np.array([20, 0, 0], dtype=float)
    )

    assert remaining_polyline.is_empty
    assert projection.segment_idx == 0


def test_interpolate_along_path_2d():
    points = np.array([[0, 0], [3, 4], [3, 8]], dtype=float)
    distances = np.array([0.0, 5.0, 9.0])

    interpolated = Polyline(points).positions_at(distances)

    assert interpolated.shape == (3, 2)
    assert np.allclose(interpolated[1], [3.0, 4.0])


def test_remaining_polyline_from_point_2d():
    points = np.array([[0, 0], [10, 0]], dtype=float)
    remaining_polyline, projection = Polyline(points).remaining_from_point(
        np.array([5, 5], dtype=float)
    )

    assert remaining_polyline.waypoints.shape == (2, 2)
    assert np.allclose(remaining_polyline.waypoints[0], [5.0, 0.0])
    assert projection.segment_idx == 0


def test_polyline_properties(simple_polyline: Polyline) -> None:
    assert len(simple_polyline) == 4
    assert simple_polyline.total_length == pytest.approx(30.0)
    assert len(simple_polyline.segment_lengths) == 3
    assert_almost_equal(simple_polyline.segment_lengths, [10.0, 10.0, 10.0])


def test_polyline_transform() -> None:
    polyline_obj = Polyline(points=np.array([[0, 0, 0], [1, 0, 0]]))
    transform = QVec(vec3=np.array([5, 5, 0]), quat=np.array([0, 0, 0, 1]))

    transformed = polyline_obj.transform(transform)
    assert_almost_equal(transformed.waypoints[0], [5, 5, 0])
    assert_almost_equal(transformed.waypoints[1], [6, 5, 0])


def test_polyline_trajectory_conversion() -> None:
    timestamps = np.array([0, 1_000_000, 2_000_000], dtype=np.uint64)
    poses = QVec(
        vec3=np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]]),
        quat=np.tile([0, 0, 0, 1], (3, 1)),
    )
    traj = Trajectory(timestamps_us=timestamps, poses=poses)

    polyline_obj = traj.to_polyline()
    assert len(polyline_obj) == 3
    assert_almost_equal(polyline_obj.waypoints, poses.vec3)


def test_polyline_create_empty() -> None:
    empty_3d = Polyline.create_empty()
    assert empty_3d.is_empty
    assert empty_3d.dimension == 3

    empty_2d = Polyline.create_empty(dimension=2)
    assert empty_2d.is_empty
    assert empty_2d.dimension == 2

    with pytest.raises(ValueError):
        Polyline.create_empty(dimension=4)


def test_polyline_grpc_conversion() -> None:
    points = np.array([[1, 2, 3], [4, 5, 6]])
    polyline_obj = Polyline(points=points)
    timestamp_us = 123_456_789

    grpc_route = polyline_obj.to_grpc_route(timestamp_us)
    assert len(grpc_route.waypoints) == 2
    assert grpc_route.timestamp_us == timestamp_us
    assert grpc_route.waypoints[0].x == 1.0
    assert grpc_route.waypoints[1].z == 6.0

    parsed = Polyline.from_grpc(grpc_route)
    assert_almost_equal(parsed.waypoints, points)


def test_polyline_project_point() -> None:
    polyline_obj = Polyline(points=np.array([[0, 0, 0], [10, 0, 0]]))

    result = polyline_obj.project_point(np.array([5, 2, 0]))
    assert np.allclose(result.point, [5, 0, 0])
    assert result.segment_idx == 0
    assert result.distance_along == pytest.approx(5.0)

    result = polyline_obj.project_point(np.array([15, 0, 0]))
    assert np.allclose(result.point, [10, 0, 0])
    assert result.distance_along == pytest.approx(10.0)

    result = polyline_obj.project_point(np.array([-5, 0, 0]))
    assert np.allclose(result.point, [0, 0, 0])
    assert result.distance_along == pytest.approx(0.0)


def test_polyline_resample_from_point_cases() -> None:
    polyline_obj = Polyline(points=np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]]))

    at_end = polyline_obj.resample_from_point(
        start_point=np.array([20, 0, 0]), spacing=5.0, n_points=5
    )
    assert len(at_end) == 1
    assert np.allclose(at_end.waypoints[0], [20, 0, 0])

    past_end = polyline_obj.resample_from_point(
        start_point=np.array([25, 0, 0]), spacing=5.0, n_points=5
    )
    assert len(past_end) == 0

    extended_polyline = Polyline(
        points=np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
    )

    between = extended_polyline.resample_from_point(
        start_point=np.array([5, 0, 0]), spacing=10.0, n_points=3
    )
    assert len(between) == 3
    assert np.allclose(between.waypoints[0], [5, 0, 0])
    assert np.allclose(between.waypoints[1], [15, 0, 0])
    assert np.allclose(between.waypoints[2], [25, 0, 0])

    off_path = extended_polyline.resample_from_point(
        start_point=np.array([5, 5, 0]), spacing=5.0, n_points=4
    )
    assert len(off_path) == 4
    assert np.allclose(off_path.waypoints[0], [5, 0, 0])
    assert np.allclose(off_path.waypoints[-1], [20, 0, 0])


def test_polyline_get_cumulative_distances_from_point() -> None:
    polyline_obj = Polyline(
        points=np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
    )

    cumulative, distance_to_projection = (
        polyline_obj.get_cumulative_distances_from_point(np.array([5, 0, 0]))
    )

    assert distance_to_projection == pytest.approx(5.0)
    assert len(cumulative) == 4
    assert cumulative[0] == 0.0
    assert cumulative[1] == pytest.approx(5.0)
    assert cumulative[2] == pytest.approx(15.0)
    assert cumulative[3] == pytest.approx(25.0)


def test_polyline_append() -> None:
    polyline1 = Polyline(points=np.array([[0, 0, 0], [10, 0, 0]]))
    polyline2 = Polyline(points=np.array([[10, 0, 0], [20, 0, 0]]))

    combined = polyline1.append(polyline2)
    assert len(combined) == 4
    assert np.allclose(combined.waypoints[0], [0, 0, 0])
    assert np.allclose(combined.waypoints[-1], [20, 0, 0])


def test_polyline_append_dimension_mismatch() -> None:
    polyline1 = Polyline(points=np.array([[0, 0]]))
    polyline2 = Polyline(points=np.array([[0, 0, 0]]))

    with pytest.raises(ValueError):
        polyline1.append(polyline2)


def test_polyline_append_empty() -> None:
    polyline_obj = Polyline(points=np.array([[0, 0, 0], [10, 0, 0]]))
    empty = Polyline.create_empty()

    result = polyline_obj.append(empty)
    assert len(result) == 2
    assert np.allclose(result.waypoints, polyline_obj.waypoints)

    result = empty.append(polyline_obj)
    assert len(result) == 2
    assert np.allclose(result.waypoints, polyline_obj.waypoints)


def test_polyline_clip() -> None:
    polyline_obj = Polyline(
        points=np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
    )

    clipped = polyline_obj.clip(1, 3)
    assert len(clipped) == 2
    assert np.allclose(clipped.waypoints[0], [10, 0, 0])
    assert np.allclose(clipped.waypoints[1], [20, 0, 0])

    clipped = polyline_obj.clip(end_idx=2)
    assert len(clipped) == 2
    assert np.allclose(clipped.waypoints[0], [0, 0, 0])
    assert np.allclose(clipped.waypoints[1], [10, 0, 0])

    clipped = polyline_obj.clip(start_idx=2)
    assert len(clipped) == 2
    assert np.allclose(clipped.waypoints[0], [20, 0, 0])
    assert np.allclose(clipped.waypoints[1], [30, 0, 0])


def test_polyline_zero_out_z() -> None:
    polyline_obj = Polyline(points=np.array([[0, 0, 1], [10, 0, 2], [20, 0, 3]]))

    zeroed = polyline_obj.zero_out_z()
    assert np.allclose(zeroed.waypoints[:, :2], polyline_obj.waypoints[:, :2])
    assert np.allclose(zeroed.waypoints[:, 2], 0.0)
    assert np.allclose(polyline_obj.waypoints[:, 2], [1, 2, 3])


def test_polyline_single_waypoint() -> None:
    polyline_obj = Polyline(points=np.array([[5, 5, 0]]))

    assert len(polyline_obj) == 1
    assert polyline_obj.total_length == 0.0
    assert len(polyline_obj.segment_lengths) == 0

    result = polyline_obj.project_point(np.array([0, 0, 0]))
    assert np.allclose(result.point, [5, 5, 0])
    assert result.segment_idx == 0
    assert result.distance_along == 0.0

    resampled = polyline_obj.resample_from_point(
        np.array([0, 0, 0]), spacing=1.0, n_points=5
    )
    assert len(resampled) == 1
    assert np.allclose(resampled.waypoints[0], [5, 5, 0])


def test_polyline_degenerate_segments() -> None:
    polyline_obj = Polyline(
        points=np.array(
            [
                [0, 0, 0],
                [10, 0, 0],
                [10, 0, 0],
                [20, 0, 0],
            ]
        )
    )

    result = polyline_obj.project_point(np.array([10, 5, 0]))
    assert np.allclose(result.point, [10, 0, 0])
    assert polyline_obj.total_length == pytest.approx(20.0)


def test_polyline_init_from_1d() -> None:
    polyline_obj = Polyline(points=np.array([1.0, 2.0, 3.0]))
    assert polyline_obj.waypoints.shape == (1, 3)
    assert np.allclose(polyline_obj.waypoints[0], [1.0, 2.0, 3.0])


def test_polyline_remaining_from_empty() -> None:
    empty = Polyline.create_empty()
    remaining, projection = empty.remaining_from_point(np.zeros(3))

    assert remaining.is_empty
    assert np.allclose(projection.point, np.zeros(3))
    assert projection.segment_idx == 0
    assert projection.distance_along == 0.0


def test_polyline_transform_with_rotation() -> None:
    polyline_obj = Polyline(points=np.array([[1, 0, 0], [2, 0, 0]]))
    quat_z_90 = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    transform = QVec(vec3=np.array([0, 0, 0]), quat=quat_z_90)

    transformed = polyline_obj.transform(transform)
    assert_almost_equal(transformed.waypoints[0], [0, 1, 0], decimal=5)
    assert_almost_equal(transformed.waypoints[1], [0, 2, 0], decimal=5)


def test_polyline_get_cumulative_distances_from_point_off_path() -> None:
    polyline_obj = Polyline(points=np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]]))

    cumulative, distance_to_projection = (
        polyline_obj.get_cumulative_distances_from_point(np.array([5, 5, 0]))
    )

    assert distance_to_projection == pytest.approx(5.0)
    assert len(cumulative) == 3
    assert cumulative[0] == 0.0
    assert cumulative[1] == pytest.approx(5.0)
    assert cumulative[2] == pytest.approx(15.0)


def test_polyline_project_point_near_segment_midpoint() -> None:
    polyline_obj = Polyline(points=np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]]))
    query = np.array([15, 1, 0])

    result = polyline_obj.project_point(query)
    assert result.segment_idx == 1
    assert np.allclose(result.point, [15, 0, 0])
    assert np.linalg.norm(result.point - query) == pytest.approx(1.0)


def test_polyline_projection_perpendicular() -> None:
    polyline_obj = Polyline(points=np.array([[0, 0, 0], [10, 0, 0]]))

    result = polyline_obj.project_point(np.array([5, 3, 0]))
    assert np.allclose(result.point, [5, 0, 0])
    assert result.segment_idx == 0
    assert result.distance_along == pytest.approx(5.0)


def test_polyline_2d_support_and_guards() -> None:
    polyline_2d = Polyline(points=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]]))
    assert polyline_2d.dimension == 2
    assert polyline_2d.total_length == pytest.approx(np.linalg.norm([1, 1]) + 1.0)

    distances = np.linspace(0.0, polyline_2d.total_length, 5)
    sampled = polyline_2d.positions_at(distances)
    assert sampled.shape == (5, 2)

    with pytest.raises(ValueError):
        polyline_2d.transform(QVec(vec3=np.zeros(3), quat=np.array([0, 0, 0, 1])))
    with pytest.raises(ValueError):
        polyline_2d.to_grpc_route(timestamp_us=0)
    with pytest.raises(ValueError):
        polyline_2d.zero_out_z()


def test_polyline_requires_matching_point_dimension_on_projection() -> None:
    polyline_obj = Polyline(points=np.array([[0.0, 0.0], [1.0, 0.0]]))

    with pytest.raises(ValueError):
        polyline_obj.project_point(np.array([0.0, 0.0, 0.0]))


def test_polyline_downsample_with_min_distance_empty() -> None:
    """Test downsampling an empty polyline."""
    polyline = Polyline.create_empty()
    polyline.downsample_with_min_distance(min_distance=5.0)
    assert len(polyline) == 0


def test_polyline_downsample_with_min_distance_basic() -> None:
    """Test basic downsampling with minimum distance."""
    points = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0]]
    )
    polyline = Polyline(points=points.copy())

    # All big enough, should keep all points
    polyline.downsample_with_min_distance(min_distance=3.0)
    assert len(polyline) == 4
    for i in range(4):
        assert np.allclose(polyline.waypoints[i], points[i])

    # Diffs too small, drop a few
    polyline.downsample_with_min_distance(min_distance=6.0)

    assert len(polyline) == 2
    assert np.allclose(polyline.waypoints[0], [0.0, 0.0, 0.0])
    assert np.allclose(polyline.waypoints[1], [10.0, 0.0, 0.0])


def test_polyline_downsample_with_min_distance_irregular_spacing() -> None:
    """Test downsampling with irregularly spaced points."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Duplicate point
            [10.0, 0.0, 0.0],
            [10.4, 0.0, 0.0],  # Not far enough
            [20.0, 0.0, 0.0],
        ]
    )
    polyline = Polyline(points=points.copy())
    polyline.downsample_with_min_distance(min_distance=0.5)

    assert (
        len(polyline) == 4
    ), f"Expected 4 points after downsampling, got {polyline.points}"
    assert np.allclose(polyline.waypoints[0], points[0])
    assert np.allclose(polyline.waypoints[1], points[1])
    assert np.allclose(polyline.waypoints[2], points[3])
    assert np.allclose(polyline.waypoints[3], points[5])
