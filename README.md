# Jetson CUDA Development Docker Environment (jetson-cuda-devel-jp6)

🇹🇼 中文說明請往下滑 Scroll down for Traditional Chinese.

---

## Overview

This is a custom Jetson Docker container based on JetPack 6.0 (L4T R36.2), CUDA 12.2, with support for:

- CUDA development (`nvcc`, `make`, `gcc`, `cmake`)
- PyTorch CUDA extension building
- Testing `.cu` files via Makefile
- Verified PyTorch + C++/CUDA custom op compilation

### Environment Summary

| Component     | Version                |
|---------------|------------------------|
| JetPack       | 6.0 (L4T R36.2.0)      |
| CUDA          | 12.2                   |
| PyTorch       | Prebuilt Jetson Wheel  |
| Container Tag | `jetson-cuda-devel-jp6`|

---

## Quickstart

```bash
# Build image
./build.sh

# Run container
./run.sh

