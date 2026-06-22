# Running Laser-Prompted Real-Time Object Segmentation on Smartphones via Cloud Computing

### Create folder model and download models

```bash
mkdir models && cd models
wget https://huggingface.co/striksio/Yolov5n-LPD/resolve/main/YOLOv5n-LPD.pt
```

###  Download FastSAM-s from official Repo https://github.com/CASIA-IVA-Lab/FastSAM and place it in models folder

### Build Image
```bash
docker build -t lprtos-server .
```

### Run lprtos-server
```bash
docker run --gpus all -it --rm --network=host -v $(pwd)/models:/app/models lprtos-server
```
