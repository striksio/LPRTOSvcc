import asyncio
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
CONFIDENCE_THRESHOLD = 0.25
HOST_IP = '0.0.0.0'
PORT = 8765

# ======================================================================================
# MASK -> CONTOUR (static, contour-only protocol, matches the Jetson server)
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
# WEBSOCKET HANDLER
# ======================================================================================
async def handler(websocket, yolo, sam, device):
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for frame_data in websocket:
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            binary_payload = b''
            num_points = 0

            # ---- Laser detection  ----
            results = yolo(rgb)
            det = results.xyxy[0]
            laser_found = False
            if det is not None and len(det):
                d = det[det[:, 4] > CONFIDENCE_THRESHOLD]
                if len(d):
                    top = d[d[:, 4].argmax()]
                    x1, y1, x2, y2 = float(top[0]), float(top[1]), float(top[2]), float(top[3])
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    box_avg = ((x2 - x1) + (y2 - y1)) / 2
                    laser_found = True

            if laser_found:
                # ---- Inpainting function ----
                radius = max(4, int(box_avg * 0.9))
                im = np.zeros((h, w), np.uint8)
                cv2.circle(im, (cx, cy), radius, 255, -1)
                rgb_in = cv2.inpaint(rgb, im, 3, cv2.INPAINT_TELEA)

                # ---- FastSAM at the laser point ----
                res = sam(rgb_in, points=[cx, cy], labels=[1], verbose=False)
                if res and res[0].masks is not None and len(res[0].masks.data):
                    mask = res[0].masks.data[0].cpu().numpy().astype(bool)
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
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
# MAIN
# ======================================================================================
async def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading detector ...")
    yolo = torch.hub.load(YOLO_REPO, "custom", path=YOLO_PATH, source="local")
    yolo.to(device).eval()
    yolo.conf = CONFIDENCE_THRESHOLD
    print(f"  {YOLO_PATH}")

    print("Loading FastSAM ...")
    sam = FastSAM(SAM_PATH)
    print(f"  {SAM_PATH}")

    print(f"\nLPRTOS server on ws://{HOST_IP}:{PORT}  (contour protocol)")
    print("Waiting for clients. Ctrl-C to stop.")

    bound = lambda ws: handler(ws, yolo, sam, device)
    async with websockets.serve(bound, HOST_IP, PORT, max_size=10_000_000):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
