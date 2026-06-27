import argparse
import os
import time
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch
from ultralytics import FastSAM
import tensorrt as trt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

# ======================================================================================
# CONFIG
# ======================================================================================
YOLO_REPO  = os.environ.get("YOLO_REPO", "/app/yolov5")
YOLO_PATH  = os.environ.get("YOLO_PATH", "/app/models/YOLOv5n-LPD.pt")
SAM_PATH   = os.environ.get("SAM_PATH",  "/app/models/FastSAM-s.pt")
YOLO_ENGINE = os.environ.get("YOLO_ENGINE", "/app/models/YOLOv5n-LPD.engine")
SAM_ENGINE  = os.environ.get("SAM_ENGINE",  "/app/models/FastSAM-s.engine")

COCO_PATH  = os.environ.get("COCO_PATH", "/app/data/GLGSD-Dataset-COCO.json")
IMG_ROOT   = os.environ.get("IMG_ROOT",  "/app/data")

CONFIDENCE_THRESHOLD = 0.25
IMGSZ = 640
LASER_CAT = 1
OBJECT_CAT = 2
WARMUP = 15  # frames discarded from timing


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
# DETECTION (same as the server; pt -> original coords, engine -> 640 coords)
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
# TENSORRT YOLO RUNNER (torch tensors for GPU memory, no pycuda)
# ======================================================================================
def load_engine_yolo(engine_path):
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
# HELPERS for scoring
# ======================================================================================
def point_in_box(px, py, b):
    # b is COCO bbox [x, y, w, h]
    return (b[0] <= px <= b[0] + b[2]) and (b[1] <= py <= b[1] + b[3])


def iou_binary(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def dice_binary(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s > 0 else 0.0


# ======================================================================================
# MAIN
# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("BACKEND", "pt"),
                    choices=["pt", "engine"],
                    help="pt = torch.hub YOLO + FastSAM .pt; engine = TensorRT")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load models (same style as the server) ----
    if args.backend == "pt":
        yolo = torch.hub.load(YOLO_REPO, "custom", path=YOLO_PATH, source="local")
        yolo.to(device).eval()
        yolo.conf = CONFIDENCE_THRESHOLD
        sam = FastSAM(SAM_PATH)
    else:
        yolo = load_engine_yolo(YOLO_ENGINE)
        sam = FastSAM(SAM_ENGINE)
        _ = sam(np.zeros((IMGSZ, IMGSZ, 3), np.uint8),
                points=[IMGSZ // 2, IMGSZ // 2], labels=[1], verbose=False)  # warmup

    coco = COCO(COCO_PATH)
    img_ids = coco.getImgIds()

    # ---- Accumulators ----
    det_tp = det_fp = det_fn = 0
    center_dists = []
    frames_with_gt = 0

    ious, dices = [], []
    seg_results = []

    t_detect, t_inpaint, t_fastsam, t_total = [], [], [], []

    for fi, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(IMG_ROOT, info["file_name"])
        frame = cv2.imread(path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        gt_laser = [a["bbox"] for a in anns if a["category_id"] == LASER_CAT]
        gt_object = [a for a in anns if a["category_id"] == OBJECT_CAT]
        if gt_laser:
            frames_with_gt += 1

        t0 = time.perf_counter()

        # ---- DETECT (same working image logic as server) ----
        if device == "cuda":
            torch.cuda.synchronize()
        td = time.perf_counter()
        if args.backend == "pt":
            work_img = frame
            det = detect_pt(frame, yolo)
        else:
            work_img, r, l, t = letterbox(frame, IMGSZ)
            det = detect_engine(work_img, yolo)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_detect = time.perf_counter() - td

        # predicted center in ORIGINAL coords (engine maps back from 640)
        pred_center = None
        if det is not None:
            cx, cy, box_avg = det
            if args.backend == "pt":
                ocx, ocy = cx, cy
            else:
                ocx, ocy = (cx - l) / r, (cy - t) / r
            pred_center = (ocx, ocy)

        # ---- DETECTION SCORING (top-conf center inside GT laser box) ----
        if pred_center is not None:
            ocx, ocy = pred_center
            hit = None
            for gb in gt_laser:
                if point_in_box(ocx, ocy, gb):
                    hit = gb
                    break
            if hit is not None:
                det_tp += 1
                gcx, gcy = hit[0] + hit[2] / 2.0, hit[1] + hit[3] / 2.0
                center_dists.append(float(np.hypot(ocx - gcx, ocy - gcy)))
            else:
                det_fp += 1
                if gt_laser:
                    det_fn += 1
        else:
            if gt_laser:
                det_fn += 1

        # ---- PROMPT for segmentation: detector center, else GT-center fallback ----
        # work in the same coord space as the working image of the backend
        prompt = None
        if det is not None:
            prompt = (int(cx), int(cy))           # working-image coords
            seg_box_avg = box_avg
        elif gt_laser:
            bx = gt_laser[0]
            gcx, gcy = bx[0] + bx[2] / 2.0, bx[1] + bx[3] / 2.0
            if args.backend == "pt":
                prompt = (int(gcx), int(gcy))
                seg_box_avg = (bx[2] + bx[3]) / 2.0 if bx[2] > 0 else 20.0
            else:
                prompt = (int(gcx * r + l), int(gcy * r + t))
                seg_box_avg = ((bx[2] + bx[3]) / 2.0) * r if bx[2] > 0 else 20.0

        # ---- INPAINT (always, on the working image) ----
        ti = time.perf_counter()
        if prompt is not None:
            wh, ww_ = work_img.shape[:2]
            radius = max(4, int(seg_box_avg * 0.9))
            im = np.zeros((wh, ww_), np.uint8)
            cv2.circle(im, prompt, radius, 255, -1)
            rgb_in = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)
            rgb_in = cv2.inpaint(rgb_in, im, 3, cv2.INPAINT_TELEA)
        else:
            rgb_in = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)
        dt_inpaint = time.perf_counter() - ti

        # ---- FASTSAM ----
        if device == "cuda":
            torch.cuda.synchronize()
        ts = time.perf_counter()
        mask = None
        if prompt is not None:
            res = sam(rgb_in, points=[prompt[0], prompt[1]], labels=[1], verbose=False)
            if res and res[0].masks is not None and len(res[0].masks.data):
                m = res[0].masks.data[0].cpu().numpy().astype(bool)
                if args.backend == "pt":
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                    mask = m
                else:
                    mask = unletterbox_mask(m, r, l, t, h, w)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_fastsam = time.perf_counter() - ts

        dt_total = time.perf_counter() - t0

        if fi >= WARMUP:
            t_detect.append(dt_detect)
            t_inpaint.append(dt_inpaint)
            t_fastsam.append(dt_fastsam)
            t_total.append(dt_total)

        # ---- SEGMENTATION SCORING (all frames with a GT object) ----
        if gt_object:
            gt_m = coco.annToMask(gt_object[0]).astype(bool)
            pred_m = mask if mask is not None else np.zeros((h, w), bool)
            ious.append(iou_binary(pred_m, gt_m))
            dices.append(dice_binary(pred_m, gt_m))
            if mask is not None:
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                seg_results.append({"image_id": img_id, "category_id": OBJECT_CAT,
                                    "segmentation": rle, "score": 1.0})

        if (fi + 1) % 200 == 0:
            print(f"  {fi+1}/{len(img_ids)} processed", flush=True)

    # ======================================================================================
    # METRICS
    # ======================================================================================
    precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) else 0.0
    recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) else 0.0
    center_med = float(np.median(center_dists)) if center_dists else 0.0

    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_dice = float(np.mean(dices)) if dices else 0.0

    ap50 = ap5095 = 0.0
    if seg_results:
        import json
        with open("/tmp/seg.json", "w") as f:
            json.dump(seg_results, f)
        cs = coco.loadRes("/tmp/seg.json")
        se = COCOeval(coco, cs, "segm")
        se.params.catIds = [OBJECT_CAT]
        se.evaluate(); se.accumulate(); se.summarize()
        ap5095 = float(se.stats[0])
        ap50 = float(se.stats[1])

    md = lambda xs: float(np.median(xs)) * 1000.0 if xs else 0.0
    det_ms = md(t_detect)
    seg_ms = md(t_fastsam)
    inp_ms = md(t_inpaint)
    tot_ms = md(t_total)

    yolo_name = "YOLOv5n-LPD.pt" if args.backend == "pt" else "YOLOv5n-LPD.engine"
    sam_name = "FastSAM-s.pt" if args.backend == "pt" else "FastSAM-s.engine"

    # ======================================================================================
    # PRINT  (fixed-width columns so it stays aligned regardless of content width)
    # ======================================================================================
    def rule(width=70):
        print("=" * width)

    print()
    rule()
    print(f"Device:               {device.upper()}")
    print(f"Backend:              {args.backend}")
    print(f"num_images:           {len(img_ids)}")
    print(f"frames_with_gt_laser: {frames_with_gt}")
    rule()
    print("Laser detection: top-confidence predicted center inside GT laser box")
    print("(center-based, not box overlap)")
    print()

    # ---- Laser Detection table ----
    print("LASER DETECTION")
    hdr = f"{'Model':<20}{'P':>7}{'R':>7}{'TP':>7}{'FP':>7}{'FN':>7}{'CenterMed':>12}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{yolo_name:<20}{precision:>7.3f}{recall:>7.3f}{det_tp:>7}{det_fp:>7}{det_fn:>7}"
          f"{center_med:>10.2f}px")
    print()

    # ---- Object Segmentation table ----
    print("OBJECT SEGMENTATION")
    hdr = f"{'Model':<20}{'IoU':>8}{'Dice':>8}{'AP50':>8}{'AP50:95':>10}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{sam_name:<20}{mean_iou:>8.3f}{mean_dice:>8.3f}{ap50:>8.3f}{ap5095:>10.3f}")
    print()

    # ---- Timing table ----
    print("TOTAL INFERENCE TIME (median per frame)")
    hdr = f"{'Detection':>16}{'Segmentation':>16}{'Inpainting':>14}{'Total':>10}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{yolo_name:>16}{sam_name:>16}{'Telea':>14}{'-':>10}")
    print(f"{det_ms:>14.0f}ms{seg_ms:>14.0f}ms{inp_ms:>12.0f}ms{tot_ms:>8.0f}ms")
    rule()


if __name__ == "__main__":
    main()
