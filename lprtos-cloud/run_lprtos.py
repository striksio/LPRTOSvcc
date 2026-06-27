import asyncio
import argparse
import websockets
import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import torch
from ultralytics import FastSAM

# ======================================================================================
# CONFIG
# ======================================================================================
YOLO_REPO  = os.environ.get("YOLO_REPO", "/app/yolov5")
YOLO_PATH  = os.environ.get("YOLO_PATH", "/app/models/YOLOv5n-LPD.pt")
SAM_PATH   = os.environ.get("SAM_PATH",  "/app/models/FastSAM-s.pt")
YOLO_ENGINE = os.environ.get("YOLO_ENGINE", "/app/models/YOLOv5n-LPD.engine")
SAM_ENGINE  = os.environ.get("SAM_ENGINE",  "/app/models/FastSAM-s.engine")
CONFIDENCE_THRESHOLD = 0.25
HOST_IP = '0.0.0.0'
PORT = 8765
IMGSZ = 640

# ======================================================================================
# MASK -> CONTOUR (static, contour-only protocol)
# ======================================================================================
def encode_mask_contour(mask):
    # Largest external contour as packed little-endian int16 (x,y) pairs
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return b'', 0
    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2)
    return points.flatten().astype(np.int16).tobytes(), len(points)


# ======================================================================================
# LETTERBOX (engine backend works in 640x640 space)
# ======================================================================================
def letterbox(img, new=IMGSZ, color=114):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new, new, 3), color, np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def unletterbox_mask(mask640, r, l, t, oh, ow):
    nh, nw = int(round(oh * r)), int(round(ow * r))
    crop = mask640[t:t + nh, l:l + nw].astype(np.uint8)
    return cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)


# ======================================================================================
# DETECTION (the only part that differs between backends)
# pt    : detect on the raw frame, return point + box_avg in ORIGINAL coords
# engine: detect on the 640 letterbox, return point + box_avg in LETTERBOX coords
# ======================================================================================
def detect_pt(frame, yolo):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo(rgb)
    det = results.xyxy[0]
    if det is None or not len(det):
        return None
    d = det[det[:, 4] > CONFIDENCE_THRESHOLD]
    if not len(d):
        return None
    top = d[d[:, 4].argmax()]
    x1, y1, x2, y2 = float(top[0]), float(top[1]), float(top[2]), float(top[3])
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    box_avg = ((x2 - x1) + (y2 - y1)) / 2
    return cx, cy, box_avg


def detect_engine(work_img, yolo_trt):
    # work_img is the 640 letterboxed BGR frame
    rgb = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)
    blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = trt_infer(yolo_trt, blob)[0]
    obj = out[:, 4]
    conf = obj * out[:, 5:].max(axis=1) if out.shape[1] > 5 else obj
    if not len(conf) or float(conf.max()) <= CONFIDENCE_THRESHOLD:
        return None
    i = int(conf.argmax())
    cx, cy, ww, hh = out[i, 0], out[i, 1], out[i, 2], out[i, 3]
    return int(cx), int(cy), (ww + hh) / 2


# ======================================================================================
# WEBSOCKET HANDLER  (shared flow: detect -> inpaint -> FastSAM -> contour)
# ======================================================================================
async def handler(websocket, backend, yolo, sam):
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for frame_data in websocket:
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            h, w = frame.shape[:2]

            binary_payload = b''
            num_points = 0

            # ---- Choose the working image + detect (differs by backend) ----
            if backend == "pt":
                work_img = frame                       # original space
                det = detect_pt(frame, yolo)
            else:
                work_img, r, l, t = letterbox(frame, IMGSZ)  # 640 space
                det = detect_engine(work_img, yolo)

            laser_found = det is not None

            if laser_found:
                cx, cy, box_avg = det
                wh, ww_ = work_img.shape[:2]

                # ---- Inpainting (on the working image) ----
                radius = max(4, int(box_avg * 0.9))
                im = np.zeros((wh, ww_), np.uint8)
                cv2.circle(im, (cx, cy), radius, 255, -1)
                rgb_in = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)
                rgb_in = cv2.inpaint(rgb_in, im, 3, cv2.INPAINT_TELEA)

                # ---- FastSAM at the laser point ----
                res = sam(rgb_in, points=[cx, cy], labels=[1], verbose=False)
                if res and res[0].masks is not None and len(res[0].masks.data):
                    mask = res[0].masks.data[0].cpu().numpy().astype(bool)
                    if backend == "pt":
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        mask = unletterbox_mask(mask, r, l, t, h, w)
                    binary_payload, num_points = encode_mask_contour(mask)

            print(f"Laser: {'yes' if laser_found else 'no '} | "
                  f"points: {num_points} | payload: {len(binary_payload)/1024:.1f} KB")

            try:
                await websocket.send(binary_payload)
            except websockets.exceptions.ConnectionClosed:
                print("Client disconnected mid-send.")
                break
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print(f"Connection closed: {websocket.remote_address}")


# ======================================================================================
# TENSORRT YOLO RUNNER (torch tensors for GPU memory, no pycuda)
# ======================================================================================
def load_engine_yolo(engine_path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    in_name, out_names = None, []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            in_name = n
        else:
            out_names.append(n)
    return {"ctx": ctx, "in_name": in_name, "out_names": out_names,
            "stream": torch.cuda.Stream()}


def trt_infer(yolo_trt, blob):
    ctx = yolo_trt["ctx"]
    d_in = torch.from_numpy(np.ascontiguousarray(blob)).cuda()
    ctx.set_input_shape(yolo_trt["in_name"], tuple(blob.shape))
    ctx.set_tensor_address(yolo_trt["in_name"], int(d_in.data_ptr()))
    outs = {}
    for n in yolo_trt["out_names"]:
        shape = tuple(ctx.get_tensor_shape(n))
        out_t = torch.empty(shape, dtype=torch.float32, device="cuda")
        outs[n] = out_t
        ctx.set_tensor_address(n, int(out_t.data_ptr()))
    ctx.execute_async_v3(yolo_trt["stream"].cuda_stream)
    yolo_trt["stream"].synchronize()
    return outs[yolo_trt["out_names"][0]].cpu().numpy()


# ======================================================================================
# MAIN
# ======================================================================================
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("BACKEND", "pt"),
                    choices=["pt", "engine"],
                    help="pt = torch.hub YOLO + FastSAM .pt; engine = TensorRT")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Backend: {args.backend}")

    if args.backend == "pt":
        print("Loading detector (pt) ...")
        yolo = torch.hub.load(YOLO_REPO, "custom", path=YOLO_PATH, source="local")
        yolo.to(device).eval()
        yolo.conf = CONFIDENCE_THRESHOLD
        print(f"  {YOLO_PATH}")
        print("Loading FastSAM (pt) ...")
        sam = FastSAM(SAM_PATH)
        print(f"  {SAM_PATH}")
    else:
        print("Loading detector (engine) ...")
        yolo = load_engine_yolo(YOLO_ENGINE)
        print(f"  {YOLO_ENGINE}")
        print("Loading FastSAM (engine) ...")
        sam = FastSAM(SAM_ENGINE)
        _ = sam(np.zeros((IMGSZ, IMGSZ, 3), np.uint8),
                points=[IMGSZ // 2, IMGSZ // 2], labels=[1], verbose=False)  # warmup
        print(f"  {SAM_ENGINE}")

    print(f"\nLPRTOS server on ws://{HOST_IP}:{PORT}  (contour protocol, {args.backend})")
    print("Waiting for clients. Ctrl-C to stop.")

    bound = lambda ws: handler(ws, args.backend, yolo, sam)
    async with websockets.serve(bound, HOST_IP, PORT, max_size=10_000_000):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
