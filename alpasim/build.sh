#!/bin/bash

# For manual building of the VAM driver docker image
#
# Usage:
#
# DOCKER_IMAGE_TAG=nvcr.io/nvidian/alpamayo/alpasim-vam-driver:1.0.0_dev ./build.sh

# Go to src
echo "Building in $(pwd)"

# Print commands before executing
set -x

# Check if DOCKER_IMAGE_TAG is set
if [ -z "$DOCKER_IMAGE_TAG" ]; then
  echo "Error: DOCKER_IMAGE_TAG is not set."
  exit 1
fi
# Extract just the image name without the registry prefix
IMAGE_NAME=$(echo $DOCKER_IMAGE_TAG | rev | cut -d'/' -f1 | rev)
# Replace all : in IMAGE_NAME with _
IMAGE_NAME=${IMAGE_NAME//:/_}
IMAGE_NAME=${IMAGE_NAME//-/_}

echo "Building image: $IMAGE_NAME"

# Build the Docker image
docker build \
  --network=host \
  --secret id=netrc,src=$HOME/.netrc \
  -t $DOCKER_IMAGE_TAG .

rm -f $IMAGE_NAME.sqsh
enroot import -o $IMAGE_NAME.sqsh dockerd://$DOCKER_IMAGE_TAG

rsync -avz --progress "$IMAGE_NAME.sqsh" "nvpark:/lustre/fs12/portfolios/av/projects/av_alpamayo_sim/.cache/sqsh/dev/${IMAGE_NAME}.sqsh"

# docker push $DOCKER_IMAGE_TAG
