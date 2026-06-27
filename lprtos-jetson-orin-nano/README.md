# Running Laser-Prompted Real-Time Object Segmentation on Smartphones via Nvidia Jetson Orin Nano


### Build the image on Nvidia Jetson Orin Nano
```bash
docker build -t lprtos-jetson-orin-nano .
```

### Generate models

#### Create folder model and download models

```bash
mkdir models && cd models
wget https://huggingface.co/striksio/Yolov5n-LPD/resolve/main/YOLOv5n-LPD.pt
```

#### Download FastSAM-s from official Repo https://github.com/CASIA-IVA-Lab/FastSAM and place it in models folder

#### Generate onnx format from .pt
YOLOv5n-LPD.pt

```bash
docker run --runtime nvidia -it --rm \
  -v ~/lprtos-jetson-orin-nano/models:/app/models \
  lprtos-jetson-orin-nano \
  python3 /app/yolov5/export.py \
    --weights /app/models/YOLOv5n-LPD.pt \
    --include onnx \
    --imgsz 640 640 \
    --opset 12
```

FastSAM-s.pt
```bash
docker run --runtime nvidia -it --rm \
  -v ~/lprtos-jetson-orin-nano/models:/app/models \
  lprtos-jetson-orin-nano \
  python3 -c "from ultralytics import FastSAM; FastSAM('/app/models/FastSAM-s.pt').export(format='onnx', imgsz=640, opset=12)"
```

---

#### Now, generate the engine models from onnx

YOLOv5n-LPD.pt

```bash
docker run --runtime nvidia -it --rm \
  -v ~/lprtos-jetson-orin-nano/models:/app/models \
  lprtos-jetson-orin-nano \
  trtexec --onnx=/app/models/YOLOv5n-LPD.onnx \
          --saveEngine=/app/models/YOLOv5n-LPD.engine \
          --fp16
```

FastSAM-s.pt
```bash
docker run --runtime nvidia -it --rm \
  -v ~/lprtos-jetson-orin-nano/models:/app/models \
  lprtos-jetson-orin-nano \
  trtexec --onnx=/app/models/FastSAM-s.onnx \
          --saveEngine=/app/models/FastSAM-s.engine \
          --fp16
```


#### Running with the TensorRT (preferable)

```bash
docker run --runtime nvidia -it --rm --network=host \
  -v ~/lprtos-jetson-orin-nano/models:/app/models \
  lprtos-jetson-orin-nano \
  python3 run_lprtos_jetson.py --backend engine
```

#### Running with the original PT

```bash
docker run --runtime nvidia -it --rm --network=host \
  -v ~/lprtos-jetson-orin-nano/models:/app/models \
  lprtos-jetson-orin-nano \
  python3 run_lprtos_jetson.py --backend pt
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
docker run --runtime nvidia -it --rm \
  -v ~/lprtos-jetson-orin-nano/models:/app/models \
  -v ~/lprtos-jetson-orin-nano/data:/app/data \
  -v ~/lprtos-jetson-orin-nano/benchmark_lprtos_jetson.py:/app/benchmark_lprtos_jetson.py \
  lprtos-jetson-orin-nano \
  python3 benchmark_lprtos_jetson.py --backend engine
```

#### Running with the original PT

```bash
docker run --runtime nvidia -it --rm \
  -v ~/lprtos-jetson-orin-nano/models:/app/models \
  -v ~/lprtos-jetson-orin-nano/data:/app/data \
  -v ~/lprtos-jetson-orin-nano/benchmark_lprtos_jetson.py:/app/benchmark_lprtos_jetson.py \
  lprtos-jetson-orin-nano \
  python3 benchmark_lprtos_jetson.py --backend pt
```



