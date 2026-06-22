# Base: the PyTorch image built locally with jetson-containers (JetPack 7.2 / r39 / CUDA 13)
ARG PYTORCH_IMAGE=pytorch:r39.2.tegra-aarch64-cu132-24.04-pytorch
FROM ${PYTORCH_IMAGE}

# System deps for OpenCV runtime + git for cloning yolov5
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone YOLOv5 locally so torch.hub needs no network at runtime
RUN git clone https://github.com/ultralytics/yolov5.git /app/yolov5

# App Python deps.
# Do NOT install yolov5/requirements.txt - its pins (NumPy 1.x, old opencv) conflict
# with the base image's torch 2.12 / NumPy 2.x. Install only what inference needs and
# let the base torch/numpy stand. --extra-index-url falls back to PyPI for anything the
# jetson index lacks (same gap we patched during the base build).
RUN pip3 install --no-cache-dir --extra-index-url https://pypi.org/simple \
    websockets \
    ultralytics \
    opencv-python-headless \
    pyyaml \
    requests \
    pandas \
    seaborn \
    tqdm \
    matplotlib \
    psutil

# ultralytics pulls nvidia-cudnn-cu13 9.20 which shadows the system cuDNN 9.21
# torch 2.12 was built against. Remove it so torch loads the system lib.
RUN pip3 uninstall -y nvidia-cudnn-cu13

# TensorRT Python bindings (needed for the --backend engine path).
# The wheel is staged in the base image; install the cp312 one into the venv.
RUN pip3 install --no-cache-dir /usr/local/lib/python3.12/dist-packages/tensorrt-10.16.1.11-cp312-none-linux_aarch64.whl

# TensorRT + CUDA libs live under the cuda targets dir, off the default loader path.
# Required so the tensorrt binding finds libnvinfer.so at runtime.
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}

WORKDIR /app

# App code
COPY run_lprtos_jetson.py /app/run_lprtos_jetson.py



EXPOSE 8765

# Default backend is pt; override with --backend engine or -e BACKEND=engine
CMD ["python3", "run_lprtos_jetson.py"]
