#!/bin/bash

IMAGE_NAME="jetson-cuda-devel-jp6"
DOCKERFILE="Dockerfile.cuda-devel-jp6"

echo "🛠️ 建立 Docker image: $IMAGE_NAME"
docker build -t ${IMAGE_NAME} -f ${DOCKERFILE} .

