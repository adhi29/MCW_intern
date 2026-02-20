#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

apt-get update
apt-get install -y net-tools

find_free_port() {
    local port=$1
    local max_port=$2
    while [ $port -le $max_port ]; do
        if netstat -an | grep -q ":$port "; then
            ((port++))
        else
            echo $port
            return 0
        fi
    done

    echo "No free port found in the specified range"
    return 1
}

ROOT=$PWD
PORT=$(find_free_port 7000 8000)

pushd /mnt/vscode
echo bash code_server_simple.sh alpasim${SLURM_JOB_ID} ${PORT}
     bash code_server_simple.sh alpasim${SLURM_JOB_ID} ${PORT}
popd
