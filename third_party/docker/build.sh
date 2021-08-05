#!/bin/bash
IMAGE_NAME=clink-base
IMAGE_VERSION=1.0.0
IMAGE=${IMAGE_NAME}:${IMAGE_VERSION}
docker build --network=host --no-cache . -f Dockerfile -t ${IMAGE}
exit 0
