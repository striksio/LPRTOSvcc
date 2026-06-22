# Running Laser-Prompted Real-Time Object Segmentation on Smartphones via Nvidia Jetson Orin Nano


### Build the image on Nvidia Jetson Orin Nano
```bash
docker build -t lprtos-jetson-orin-nano .
```

### Generate models

More to be added soon.

### Running with the TensorRT (prefferable, works well with ~30 FPS)

```bash
docker run --runtime nvidia -it --rm --network=host \
  -v ~/lprtos-jetson-orin-nano/models:/app/models \
  lprtos-jetson-orin-nano \
  python3 run_lprtos_jetson.py --backend engine
```

### Running with the original PT

```bash
docker run --runtime nvidia -it --rm --network=host \
  -v ~/lprtos-jetson-orin-nano/models:/app/models \
  lprtos-jetson-orin-nano \
  python3 run_lprtos_jetson.py --backend pt
```

