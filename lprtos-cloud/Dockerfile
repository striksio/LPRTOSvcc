# Preinstalled Docker with Cuda >= 12.1
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Extra python packages
RUN pip install --no-cache-dir \
        ultralytics \
        websockets \
        opencv-python-headless \
        numpy

# yolov5 repo
RUN git clone --depth 1 https://github.com/ultralytics/yolov5.git /app/yolov5 \
    && pip install --no-cache-dir -r /app/yolov5/requirements.txt

WORKDIR /app

# Main pipeline
COPY run_lprtos.py /app/run_lprtos.py

EXPOSE 8765

CMD ["python", "run_lprtos.py"]
