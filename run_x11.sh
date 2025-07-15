#!/bin/bash

IMAGE_NAME="jetson-cuda-devel-jp6"
CONTAINER_NAME="cuda-devel-jp6"

echo "🚀 啟動 Docker container: $CONTAINER_NAME"

# 自動刪除舊容器（如已存在）
docker rm -f ${CONTAINER_NAME} 2>/dev/null

# 判斷是否為互動終端
if [ -t 1 ]; then
    TTY_ARGS="-it"
else
    TTY_ARGS="-i"
fi

docker run $TTY_ARGS \
  --runtime nvidia \
  --network host \
  --name ${CONTAINER_NAME} \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -e DISPLAY=:0 \
  -e XAUTHORITY=/home/user/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/user/.Xauthority:/home/user/.Xauthority \
  -v $(pwd):/workspace \
  ${IMAGE_NAME}

