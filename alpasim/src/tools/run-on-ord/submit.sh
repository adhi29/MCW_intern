#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

#SBATCH --account av_alpamayo_sim
#SBATCH --partition polar,polar3,polar4,grizzly
#SBATCH --time 03:59:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --job-name alpasim
#SBATCH --output=./runs/slurm_output/%j.log

# Submit via: sbatch -A <allocation> [--array=0-<n>] submit.sh [--oss] [<hydra_overrides>]
#
# Where <allocation> is your project allocation and <n> is the number of jobs to
# run. The scenarios will automatically be distributed across the jobs.

# Uncomment to log every line
# set -x

# ------------------------------- SHOULD NOT REQUIRE CHANGES -------------------------------
# Detect if running on slurm node
if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script should run on SLURM. Start with 'sbatch submit.sh'. Exiting."
    exit 1
fi

# Provide a default description if not set
if [[ -z "$SUBMITTER" ]]; then
    SUBMITTER="$(whoami)"
fi

if [[ -z "$DESCRIPTION" ]]; then
    DESCRIPTION="unspecified"
fi

# Parse script args
DEPLOY_TARGET="ord"
HYDRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --oss)
            DEPLOY_TARGET="ord_oss"
            shift
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                HYDRA_ARGS+=("$1")
                shift
            done
            break
            ;;
        *)
            HYDRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Find parent directory, no matter where to script is called from

# Note (Qi): On SLURM, SLURM_JOB_ID is uniquely defined for every job, including jobs in an array.
# However, for array jobs, it is possible for SLURM_JOB_ID to equal SLURM_ARRAY_JOB_ID for one of the
# jobs. This can cause issues with scontrol because scontrol may return multiple job entries corresponding
# to the entire array. To resolve this issue, array launches and regular launches can be distinguished
# by checking SLURM_ARRAY_JOB_ID. For array jobs, use SLURM_ARRAY_JOB_ID_SLURM_ARRAY_TASK_ID instead of
# SLURM_JOB_ID when interacting with scontrol.
if [[ -z $SLURM_ARRAY_JOB_ID ]]; then
    UNIQUE_JOB_ID="${SLURM_JOB_ID}"
else
    UNIQUE_JOB_ID="${SLURM_ARRAY_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
fi
SCRIPT_PATH=$(scontrol show job "${UNIQUE_JOB_ID}" | awk -F= '/Command=/{print $2}')

SCRIPT_DIR=$(readlink -f "$(dirname $SCRIPT_PATH)")

# If LOGDIR is not specified, we generate a logdir in the folder where this script lives. If
# a relative LOGDIR is specified, we assume the user wants to set the LOGDIR relative to where
# the script is submitted.
if [[ -z "$LOGDIR" ]]; then
    if [ -z "${SLURM_ARRAY_JOB_ID}" ]; then
        # Non array job
        LOGDIR=$SCRIPT_DIR/runs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}_$(date +%Y_%m_%d__%H_%M_%S)
        ARRAY_JOB_DIR=$LOGDIR
    else
        # Array job, we use a hierarchical logdir to group all array jobs
        ARRAY_JOB_DIR=$SCRIPT_DIR/runs/${SLURM_ARRAY_JOB_ID}_${SLURM_JOB_NAME}
        LOGDIR=$ARRAY_JOB_DIR/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$(date +%Y_%m_%d__%H_%M_%S)
    fi
else
    if [[ "$LOGDIR" != /* ]]; then
        LOGDIR=$(readlink -f "$SLURM_SUBMIT_DIR/$LOGDIR")
    fi
fi

# Create txt-logs directory if it doesn't exist
mkdir -p ${LOGDIR}/txt-logs

# Want the Slurm logs in LOGDIR, but don't know LOGDIR until after job started.
# Copy any early output to our new log file.
cat ./runs/slurm_output/${SLURM_JOB_ID}.log > ${LOGDIR}/txt-logs/slurm.log 2>/dev/null

# Redirect all future output to both terminal and the log file
exec > >(tee -a "${LOGDIR}/txt-logs/slurm.log") 2>&1

# Create resume.sh script
# We build a resume command that can be executed from any directory by using absolute paths.
#
# Step 1: Get the original sbatch command used to submit this job.
#   Example output: "sbatch -A av_alpamayo_sim /path/to/run-on-ord/submit.sh --oss"
ORIG_SUBMIT_CMD=$(sacct -j ${SLURM_JOB_ID} -o submitline -P | head -n 2 | sed '1d')
#
# Step 2: Extract just the sbatch options (account, partition, etc.) by:
#   - First sed: Remove everything from "submit.sh" onwards
#     "sbatch -A av_alpamayo_sim /path/to/run-on-ord/submit.sh --oss" -> "sbatch -A av_alpamayo_sim /path/to/run-on-ord/"
#   - Second sed: Remove any path prefix up to and including "run-on-ord/"
#     "sbatch -A av_alpamayo_sim /path/to/run-on-ord/" -> "sbatch -A av_alpamayo_sim "
SBATCH_OPTIONS=$(echo "${ORIG_SUBMIT_CMD}" | sed 's/submit\.sh.*//' | sed 's|.*/run-on-ord/||')
#
# Step 3: Build the resume command using the absolute path to resume_slurm_job.sh via SCRIPT_DIR.
# This ensures resume.sh can be executed from any directory.
#   Example result: "sbatch -A av_alpamayo_sim /path/to/run-on-ord/resume_slurm_job.sh /path/to/job"
RESUME_CMD="${SBATCH_OPTIONS}${SCRIPT_DIR}/resume_slurm_job.sh ${ARRAY_JOB_DIR}"

# Copy resume template to LOGDIR and append the actual resume command if it doesn't exist yet
if [ ! -f "${ARRAY_JOB_DIR}/resume.sh" ] && [[ -z "${SLURM_ARRAY_TASK_ID}" || "${SLURM_ARRAY_TASK_ID}" == "${SLURM_ARRAY_TASK_MIN}" ]]; then
    cp ${SCRIPT_DIR}/resume_template.sh ${ARRAY_JOB_DIR}/resume.sh
    echo "${RESUME_CMD} \"\$@\"" >> ${ARRAY_JOB_DIR}/resume.sh
    chmod +x ${ARRAY_JOB_DIR}/resume.sh
fi

# Install new dependencies if required.
uv tool upgrade alpasim_wizard
alpasim_wizard \
    +deploy=${DEPLOY_TARGET} \
    wizard.log_dir=$LOGDIR \
    wizard.array_job_dir=$ARRAY_JOB_DIR \
    wizard.latest_symlink=true \
    wizard.submitter="$SUBMITTER" \
    wizard.description="$DESCRIPTION" \
    "${HYDRA_ARGS[@]}" # pass hydra overrides
