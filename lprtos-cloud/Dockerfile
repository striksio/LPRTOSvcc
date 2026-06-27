01-py3 ships TensorRT 10.x + trtexec (so --fp16 works) and supports
# Turing/Ampere/Ada (T4/L4/A10/A100). It does NOT support Volta (V100/P100),
# which TensorRT 10.x dropped as an engine-build target.
FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive

# System libs OpenCV needs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps. DO NOT install torch or tensorrt: both are already in the base
# image and reinstalling them would break the validated CUDA build.
# Pin numpy<2: the container's torch is built against NumPy 1.x, so let pip keep
# numpy 1.x rather than letting ultralytics/onnx pull NumPy 2.x (which crashes torch).
RUN pip install --no-cache-dir \
        "numpy<2" \
        ultralytics \
        websockets \
        opencv-python-headless \
        pyyaml \
        requests \
        pandas \
        seaborn \
        tqdm \
        matplotlib \
        psutil \
        scipy \
        pycocotools \
        onnx \
        onnxscript

# yolov5 repo for the torch.hub 'custom' loader (matches how the .pt was trained).
RUN git clone --depth 1 https://github.com/ultralytics/yolov5.git /app/yolov5

WORKDIR /app

# Server + benchmark code. Models/data are mounted at runtime.
COPY run_lprtos.py /app/run_lprtos.py
COPY benchmark_lprtos.py /app/benchmark_lprtos.py

EXPOSE 8765

CMD ["python", "run_lprtos.py"]
