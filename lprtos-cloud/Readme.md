# Running Laser-Prompted Real-Time Object Segmentation on Smartphones via Cloud Computing

### GPU requirements

- The PT backend runs on any CUDA GPU (T4, L4, V100, A10, A100, ...)
- The TensorRT engine backend needs a Turing-or-newer GPU (T4, L4, A10, A100)
  TensorRT 10.x dropped Volta support, so V100 / P100 cannot build engines, need additional work on that
- Recommended L4

### Build the image

```bash
docker build -t lprtos-server .
```

### Generate models

##### Create folder model and download models

```bash
mkdir models && cd models
wget https://huggingface.co/striksio/Yolov5n-LPD/resolve/main/YOLOv5n-LPD.pt
```

#### Download FastSAM-s from official Repo https://github.com/CASIA-IVA-Lab/FastSAM and place it in models folder

#### Generate onnx format from .pt

YOLOv5n-LPD.pt

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/models:/app/models \
  lprtos-server \
  python3 /app/yolov5/export.py \
    --weights /app/models/YOLOv5n-LPD.pt \
    --include onnx \
    --imgsz 640 640 \
    --opset 12
```

FastSAM-s.pt

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/models:/app/models \
  lprtos-server \
  python3 -c "from ultralytics import FastSAM; FastSAM('/app/models/FastSAM-s.pt').export(format='onnx', imgsz=640, opset=12)"
```

---

#### Now, generate the engine models from onnx

YOLOv5n-LPD.onnx

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/models:/app/models \
  lprtos-server \
  trtexec --onnx=/app/models/YOLOv5n-LPD.onnx \
          --saveEngine=/app/models/YOLOv5n-LPD.engine \
          --fp16
```

FastSAM-s.onnx

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/models:/app/models \
  lprtos-server \
  trtexec --onnx=/app/models/FastSAM-s.onnx \
          --saveEngine=/app/models/FastSAM-s.engine \
          --fp16
```

---

#### Running with the TensorRT

```bash
docker run --gpus all -it --rm --network=host \
  -v $(pwd)/models:/app/models \
  lprtos-server \
  python3 run_lprtos.py --backend engine
```

#### Running with the original PT

```bash
docker run --gpus all -it --rm --network=host \
  -v $(pwd)/models:/app/models \
  lprtos-server \
  python3 run_lprtos.py --backend pt
```


### For the dataset evaluation

####  Download Green Laser dataset from the Hugging Face (https://huggingface.co/datasets/striksio/Green-Laser-Guided-Segmentation-Dataset), and place it into folder data. Have models saved as before. 

```bash
mkdir data && cd data
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/striksio/Green-Laser-Guided-Segmentation-Dataset
cd Green-Laser-Guided-Segmentation-Dataset
git lfs pull

mv GLGSD-Dataset-COCO GLGSD-Dataset-COCO.json ../ && cd ../

```

---

#### Running with the TensorRT

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  lprtos-server \
  python3 benchmark_lprtos.py --backend engine
```

#### Running with the original PT

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  lprtos-server \
  python3 benchmark_lprtos.py --backend pt
```
