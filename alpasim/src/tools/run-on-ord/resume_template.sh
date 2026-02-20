#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

# Usage: Just execute this script.

# Find the current directory where this script resides (ARRAY_JOB_DIR)
SCRIPT_PATH="$(readlink -f "$0")"
ARRAY_JOB_DIR="$(dirname "$SCRIPT_PATH")"
echo "Resuming job from directory: $ARRAY_JOB_DIR"
TIMESTAMP=$(date +%Y_%m_%d__%H_%M_%S)

OLD_AGGREGATE_DIR="$ARRAY_JOB_DIR/aggregate_old_${TIMESTAMP}"
#
# Rename aggregate folder if it exists
if [ -d "$ARRAY_JOB_DIR/aggregate" ]; then
    echo "Renaming existing aggregate folder to ${OLD_AGGREGATE_DIR}"
    mv "$ARRAY_JOB_DIR/aggregate" "$OLD_AGGREGATE_DIR"
else
    # Create OLD_AGGREGATE_DIR if it doesn't exist
    mkdir -p "$OLD_AGGREGATE_DIR"
fi


# Perform cleanup work before resuming
echo "Performing cleanup work..."
# Move any temporary files or lock files to OLD_AGGREGATE_DIR if they exist
if [ -f "$ARRAY_JOB_DIR/post_eval_aggregation.json" ]; then
    mv "$ARRAY_JOB_DIR/post_eval_aggregation.json" "$OLD_AGGREGATE_DIR/" 2>/dev/null
fi
if [ -f "$ARRAY_JOB_DIR/post_eval_aggregation.lock" ]; then
    mv "$ARRAY_JOB_DIR/post_eval_aggregation.lock" "$OLD_AGGREGATE_DIR/" 2>/dev/null
fi


echo "Cleanup complete, ready to resume job"

# The actual resume command will be appended by submit.sh
