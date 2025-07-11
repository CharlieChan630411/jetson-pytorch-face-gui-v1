#!/bin/bash

IMAGE_NAME="jetson-cuda-devel-jp6"
CONTAINER_NAME="cuda-devel-jp6"

echo "🚀 啟動 Docker container: $CONTAINER_NAME"
docker run -it --rm \
  --runtime nvidia \
  --network host \
  --name ${CONTAINER_NAME} \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $(pwd):/workspace \
  ${IMAGE_NAME}

