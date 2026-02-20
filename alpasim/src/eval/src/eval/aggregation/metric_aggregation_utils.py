# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl  # type: ignore
import polars.selectors as ps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METADATA_COLUMNS_TO_DROP_FROM_AVMF = [
    "metadata.source.type",
    "metadata.source.local_files.local_logfile_filepath",
    "metadata.source.local_files.test_name",
    "metadata.source.local_files.rov_mapfile_path",
    "metadata.diff_source.type",
    "metadata.job_start_time",
    "metadata.requirement_id",
    "completion_info.failure_type",
    "metadata.version",
]


def _repeat_metadata_and_concat(
    metadata_df: pl.DataFrame, df: pl.DataFrame
) -> pl.DataFrame:
    return pl.concat(
        [pl.concat([metadata_df] * len(df), how="vertical"), df], how="horizontal"
    )


def _rename_avmf_metadata(df: pl.DataFrame) -> pl.DataFrame:

    return (
        df.with_columns(
            pl.col("metadata.source.local_files.segment_info.start_time")
            .cast(pl.Int64)
            .alias("metadata.segment_start_time"),
            pl.col("metadata.source.local_files.segment_info.end_time")
            .cast(pl.Int64)
            .alias("metadata.segment_end_time"),
            pl.col("metadata.version").cast(pl.Float32),
        )
        .with_columns(
            (
                pl.col("metadata.segment_end_time")
                - pl.col("metadata.segment_start_time")
            ).alias("segment_duration_us"),
        )
        .drop(
            [
                "metadata.source.local_files.segment_info.start_time",
                "metadata.source.local_files.segment_info.end_time",
            ]
        )
    )


def _get_trajectory_ids_from_json_row(json_row: dict) -> tuple[str, str, int]:
    path = Path(
        json_row[0]["metadata"]["source"]["local_files"]["local_logfile_filepath"]
    )
    clipgt_id, batch_id = str(path.parent).split("/")[-2:]
    rollout_id = int(path.stem)
    return clipgt_id, batch_id, rollout_id


def _get_metadata_df(row: dict[str, Any]) -> pl.DataFrame:
    json_row = json.loads(row["analyzer_output"])
    flat_df = pl.json_normalize(json_row)
    clipgt_id, batch_id, rollout_id = _get_trajectory_ids_from_json_row(json_row)
    assert clipgt_id == row["clipgt_id"]
    assert batch_id == row["batch_id"]
    metadata = {
        "run_name": row["run_name"],
        "run_uuid": row["run_uuid"],
        "clipgt_id": row["clipgt_id"],
        "batch_id": row["batch_id"],
        "rollout_id": rollout_id,
    }
    metadata_df = flat_df.select(
        ps.matches("^metadata.*$"),
        ps.matches("^completion_info.*$"),
    )
    metadata_df = _rename_avmf_metadata(metadata_df)
    metadata_df = metadata_df.with_columns(
        [pl.lit(metadata[key]).alias(key) for key in metadata.keys()]
    )
    return metadata_df


def _get_analyzer_df(row: dict[str, Any]) -> pl.DataFrame:
    json_row = json.loads(row["analyzer_output"])
    flat_df = pl.json_normalize(json_row)
    analyzer_df = flat_df.select(ps.matches("^analyzer_output.custom.*$"))
    return analyzer_df


def _parse_replay_collision_df(
    analyzer_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    parsed_dfs: dict[str, list[pl.DataFrame]],
) -> dict[str, list[pl.DataFrame]]:
    df = analyzer_df.rename(
        {
            col: col.replace("analyzer_output.custom.replay_collision.", "")
            for col in analyzer_df.columns
        }
    )
    df = df.unpivot(
        index=["timestamp"],
        on=[
            "has_collision",
            "impact_speed_kph",
            "min_distance",
            "impact_speed_relative_kph",
        ],
    ).with_columns(
        pl.col("timestamp"),
    )
    parsed_dfs["per_scene"].append(_repeat_metadata_and_concat(metadata_df, df))
    return parsed_dfs


def _parse_kinematics_df(
    analyzer_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    parsed_dfs: dict[str, list[pl.DataFrame]],
) -> dict[str, list[pl.DataFrame]]:
    df = analyzer_df["analyzer_output.custom.kinematics.timeseries_data"][0]
    df = pl.DataFrame(df).unnest("")
    df = df.unpivot(
        index=["timestamp"],
        on=[
            "jerk_x",
            "jerk_y",
            "speed",
            "velocity_x",
            "velocity_y",
            "accel_x",
            "accel_y",
        ],
    ).with_columns(
        pl.col("timestamp").cast(pl.Int64),
    )
    parsed_dfs["per_timestep"].append(_repeat_metadata_and_concat(metadata_df, df))
    return parsed_dfs


def _parse_lateral_positioning_df(
    analyzer_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    parsed_dfs: dict[str, list[pl.DataFrame]],
) -> dict[str, list[pl.DataFrame]]:
    df = analyzer_df["analyzer_output.custom.lateral_positioning.lane_positioning"][0]
    if len(df) > 0:
        df = pl.DataFrame(df).unnest("")
        df = df.rename({"time": "timestamp"}).with_columns(
            pl.col("timestamp").cast(pl.Int64),
        )
        df = df.unpivot(
            index="timestamp",
            on=["distance_center_m", "distance_left_m", "distance_right_m"],
        )
        parsed_dfs["per_timestep"].append(_repeat_metadata_and_concat(metadata_df, df))
    else:
        print("No lane positioning data")

    df = analyzer_df["analyzer_output.custom.lateral_positioning.lane_hugging_events"][
        0
    ]
    if len(df) > 0:
        df = pl.DataFrame(df).unnest("").unnest("interval")
        df = df.with_columns(
            pl.col("start").cast(pl.Int64).cast(pl.Datetime(time_unit="us")),
            pl.col("end").cast(pl.Int64).cast(pl.Datetime(time_unit="us")),
        )
        df = df.unpivot(
            index=["start", "end"],
            on=["dtles_min"],
        )
        parsed_dfs["events"].append(_repeat_metadata_and_concat(metadata_df, df))
    else:
        print("No lane hugging events")
    # TODO: add right and left lane excursions
    # df = analyzer_df["analyzer_output.custom.lateral_positioning.right_lane_excursions"][0]
    # df = analyzer_df["analyzer_output.custom.lateral_positioning.left_lane_excursions"][0]
    return parsed_dfs


def parse_avmf_json(
    avmf_metrics_table: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:

    parsed_dfs: dict[str, list[pl.DataFrame]] = {
        # Have "timestamp", "variable", "value"
        "per_scene": [],
        "per_timestep": [],
        # Have "start", "end", "variable", "value"
        "events": [],
    }

    # analyzer_specific_dfs = defaultdict(list)
    # analyzer_specific_json_example = {}  # For debugging
    for row in avmf_metrics_table.iter_rows(named=True):
        json_row = json.loads(row["analyzer_output"])
        flat_df = pl.json_normalize(json_row)
        metadata_df = _get_metadata_df(row)
        analyzer_df = _get_analyzer_df(row)

        if not metadata_df["completion_info.completed"][0]:
            print("\033[91mNot completed:\033[0m", end=" ")
            print(flat_df["completion_info.analyzer_failure_info.exception_message"][0])
            continue

        if row["analyzer"] == "REPLAY_COLLISION":
            _parse_replay_collision_df(analyzer_df, metadata_df, parsed_dfs)
        if row["analyzer"] == "KINEMATICS":
            _parse_kinematics_df(analyzer_df, metadata_df, parsed_dfs)
        if row["analyzer"] == "LATERAL_POSITIONING":
            _parse_lateral_positioning_df(analyzer_df, metadata_df, parsed_dfs)

    events_df = (
        pl.concat(parsed_dfs["events"], how="vertical").drop(
            METADATA_COLUMNS_TO_DROP_FROM_AVMF
        )
        if len(parsed_dfs["events"]) > 0
        else None
    )
    per_timestep_df = pl.concat(parsed_dfs["per_timestep"], how="vertical").drop(
        METADATA_COLUMNS_TO_DROP_FROM_AVMF
    )
    per_scene_df = pl.concat(parsed_dfs["per_scene"], how="vertical").drop(
        METADATA_COLUMNS_TO_DROP_FROM_AVMF
    )
    return events_df, per_timestep_df, per_scene_df


def pick_relevant_avmf_columns(
    per_timestep_df: pl.DataFrame,
) -> pl.DataFrame:

    return per_timestep_df.select(
        pl.col("run_name"),
        pl.col("run_uuid"),
        pl.col("clipgt_id"),
        pl.col("batch_id"),
        pl.col("rollout_id").cast(pl.Int64),
        pl.col("variable"),
        pl.col("value"),
        (pl.col("timestamp") - pl.col("metadata.segment_start_time")).alias(
            "rel_timestamp"
        ),
    ).sort("rel_timestamp")


def add_rollout_and_trajectory_uids(
    combined_per_timestep_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add rollout and trajectory uids to the per_timestep_df.

    The rollout_uid is a unique id _accross_ scenes. It's like the `rollout_id`,
    but due to alpasim internals, the rollout_id is not unique within a scene if
    the total number of rollouts was split across multiple "batches".
    So if `n_rollouts` was greater than one, we can use this
    to compute e.g. the std by treating each rollout as a separate simulation.

    The trajectory_uid is a unique id across scenes and even runs. It's used for
    easier grouped computations in the internal processing of this script, but
    not needed by the end user.

    Returns:
        combined_per_timestep_df: The input dataframe with trajectory uids.
        trajectory_uid_df: A dataframe with the trajectory uids, rollout uids,
        and other metadata. Used to later join in the the rollout_uid and run
        name, after processing based on the trajectory_uid.
    """
    combined_per_timestep_df = combined_per_timestep_df.with_columns(
        # Unique for each scene
        pl.concat_str(["batch_id", "rollout_id"], separator="_").alias("rollout_uid"),
        # Unique even across scenes
        pl.concat_str(
            ["run_uuid", "clipgt_id", "batch_id", "rollout_id"], separator="_"
        ).alias("trajectory_uid"),
    ).with_columns(
        # pl.col("rollout_uid").rank("dense").alias("rollout_uid"),
        pl.col("trajectory_uid").rank("dense").alias("trajectory_uid"),
        pl.col("rollout_uid").rank("dense").over("clipgt_id").alias("rollout_uid"),
    )

    trajectory_uid_df = combined_per_timestep_df.select(
        pl.col("trajectory_uid"),
        pl.col("rollout_uid"),
        pl.col("run_name"),
        pl.col("run_uuid"),
        pl.col("clipgt_id"),
        pl.col("batch_id"),
        pl.col("rollout_id"),
    ).unique()

    combined_per_timestep_df = combined_per_timestep_df.drop(
        ["run_name", "run_uuid", "clipgt_id", "batch_id", "rollout_id", "rollout_uid"]
    )

    return combined_per_timestep_df, trajectory_uid_df


def filter_collision_type(
    df_wide: pl.DataFrame,
    collisions_to_filter: list[str],
) -> pl.DataFrame:
    return df_wide.filter(
        ~pl.col("kpi_collision_type").is_in(collisions_to_filter + [None])
        | pl.col("kpi_collision_type").is_null()
    )


def filter_route_deviation(
    df_wide: pl.DataFrame,
    route_deviation_th: float,
) -> pl.DataFrame:
    # Filter by route deviation: Remove all timesteps _after_ we violated the route deviation threshold.
    len_df = len(df_wide)
    df_wide = df_wide.filter(
        pl.col("rel_timestamp")
        < (
            pl.col("rel_timestamp")
            .filter(pl.col("kpi_route_deviation") > route_deviation_th)
            .first()  # TODO: Can I use min here as well?
            .fill_null(pl.col("rel_timestamp").max() + 1)  # If filter returns empty.
        ).over("trajectory_uid")
    )

    if set(df_wide["kpi_route_deviation"].unique()) <= {None, -1.0}:
        print("\033[91mNo route deviation data\033[0m")
    print(f"Selected {len(df_wide)}/{len_df} rows for {route_deviation_th=}")
    return df_wide


def add_aggregate_metrics(
    df_wide: pl.DataFrame,
) -> pl.DataFrame:
    """Add aggregate metrics to the dataframe.

    This includes:
    - offroad_or_wrong_lane: Whether the vehicle was offroad or in the wrong lane.
    - comfort: Whether the vehicle was comfortable.
    """
    # Combine offroad and wrong lane
    df_wide = df_wide.with_columns(
        (
            pl.col("kpi_offroad").cast(pl.Boolean)
            & pl.col("kpi_wrong_lane").cast(pl.Boolean)
        ).alias("kpi_offroad_or_wrong_lane"),
    )

    # Combine comfort metrics
    comfort_columns = [
        "kpi_comfort_lon_accel",
        "kpi_comfort_lat_accel",
        "kpi_comfort_lon_jerk",
        "kpi_comfort_jerk",
        "kpi_comfort_yaw_rate",
        "kpi_comfort_yaw_accel",
    ]
    df_wide = df_wide.with_columns(
        pl.all_horizontal(
            pl.col(col).cast(pl.Boolean) for col in comfort_columns
        ).alias("kpi_comfort")
    )

    return df_wide


def average_metrics_across_timesteps(
    df_wide: pl.DataFrame,
    metric_averaging_across_timesteps: dict[str, list[str]],
) -> pl.DataFrame:
    """Average metrics across timesteps.

    This function takes a dataframe with metrics cols and averages them across
    timesteps. How metrics are aggregated is specified in the
    `metric_averaging_across_timesteps` dictionary.

    Args:
        df_wide: The dataframe to average.
        metric_averaging_across_timesteps: A dictionary with the metrics to average.
    """

    non_regex_metrics = {
        key: [m for m in metrics if not (m.startswith("^") and m.endswith("$"))]
        for key, metrics in metric_averaging_across_timesteps.items()
    }

    regex_metrics = {
        key: [m for m in metrics if m.startswith("^") and m.endswith("$")]
        for key, metrics in metric_averaging_across_timesteps.items()
    }

    print("Non regex metrics: ", non_regex_metrics)
    print("Regex metrics: ", regex_metrics)

    df_wide_avg_individual_metrics = df_wide.group_by(["trajectory_uid"]).agg(
        *[pl.col(metric).max() for metric in non_regex_metrics["max"]],
        *[pl.col(metric).min() for metric in non_regex_metrics["min"]],
        *[pl.col(metric).mean() for metric in non_regex_metrics["mean"]],
    )

    df_wide_without_individual_metrics = df_wide.drop(
        non_regex_metrics["max"] + non_regex_metrics["min"] + non_regex_metrics["mean"]
    )

    df_wide_avg_regex_metrics = df_wide_without_individual_metrics.group_by(
        ["trajectory_uid"]
    ).agg(
        *[pl.col(metric).max() for metric in regex_metrics["max"]],
        *[pl.col(metric).min() for metric in regex_metrics["min"]],
        *[pl.col(metric).mean() for metric in regex_metrics["mean"]],
    )
    df_wide_avg = df_wide_avg_individual_metrics.join(
        df_wide_avg_regex_metrics, on="trajectory_uid", how="full"
    )

    # Check that all columns from df_wide (except timestep) are in df_wide_avg
    missing_cols = set(df_wide.columns) - {"rel_timestamp"} - set(df_wide_avg.columns)
    if missing_cols:
        print(
            "\033[91mSome columns were not captured in `metric_averaging_across_timesteps` and are dropped:\033[0m"
        )
        print(f"{missing_cols}")

    print("Length before aggregating: ", len(df_wide))
    print("Length after aggregating across timesteps: ", len(df_wide_avg))
    return df_wide_avg


def rename_legend_handles(ax, trajectory_uid_df: pl.DataFrame) -> None:
    """Rename the legend handles to the run names.

    Motivation: Run names aren't guaranteed to be unique. By doing the renaming
    only in the legend, and using run_uuids until then, we guarantee that
    nothing is meshed together by mistake.

    Args:
        ax: The axis to rename the legend handles for.
        trajectory_uid_df: The dataframe with the trajectory uids and run names.
    """
    run_uuid_to_name = (
        trajectory_uid_df.select(pl.col("run_uuid"), pl.col("run_name"))
        .unique()
        .to_dicts()
    )
    run_uuid_to_name = {d["run_uuid"]: d["run_name"] for d in run_uuid_to_name}
    run_uuid_to_name

    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Apply mapping to create new labels
    new_labels = [run_uuid_to_name.get(label, label) for label in labels]

    # Create new legend
    ax.legend(handles=handles, labels=new_labels)


def save_metrics_results_txt(
    df_wide_avg: pl.DataFrame,
    trajectory_uid_df: pl.DataFrame,
    metric_averaging_across_timesteps: dict[str, list[str]],
    output_path: str,
) -> None:
    # Compute number of rollouts per run for the next aggregation step
    df_fully_aggregated = (
        df_wide_avg.drop("rollout_uid")
        .group_by("run_uuid")
        .agg(
            pl.col("*").mean(),
            pl.len().alias("n_rollouts"),
        )
    )
    # Join back the run_name for easier interpretation of results
    df_fully_aggregated = df_fully_aggregated.join(
        trajectory_uid_df.select(pl.col("run_uuid"), pl.col("run_name")).unique(),
        on="run_uuid",
        how="left",
    ).drop("run_uuid")
    # Move run_name to be first column
    df_fully_aggregated = df_fully_aggregated.select(
        pl.col("run_name"), pl.all().exclude("run_name")
    )

    # Save results to txt file
    with open(output_path, "w") as f:
        for row in df_fully_aggregated.iter_rows(named=True):
            f.write(
                f"Run: {row['run_name']} (num_clips: {row['num_clips']}, n_rollouts per clip: {row['n_rollouts']})\n"
            )
            f.write("Metrics Results Summary:\n")
            f.write("-" * 40 + "\n")

            # Write all metrics except run_name, num_clips, and n_rollouts
            for col, val in row.items():
                if col not in ["run_name", "n_rollouts", "num_clips"]:
                    # Format floats to 4 decimal places
                    if isinstance(val, float):
                        f.write(f"{col}: {val:.4f}\n")
                    else:
                        f.write(f"{col}: {val}\n")

            f.write(
                f"n_clips: {row['num_clips']}, n_rollouts/clip: {row['n_rollouts']}\n"
            )
            f.write("-" * 80 + "\n\n")

    print(f"Results saved to {output_path}")
