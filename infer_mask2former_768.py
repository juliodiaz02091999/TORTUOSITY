#!/usr/bin/env python3
"""
Mask2Former @ 768×768 — crop bbox del párpado (sin ennegrecer), padding, CLAHE OpenCV LAB.
Compatible con infer_server (model_type=mask2former_768).
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from transformers import (
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
)
from skimage.color import label2rgb

MODEL_ID = "facebook/mask2former-swin-small-cityscapes-instance"

ID2LABEL = {0: "background", 1: "gland"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

NUM_QUERIES = 150
OUTPUT_SIZE = (768, 768)
MIN_INSTANCE_AREA = 30
INSTANCE_SCORE_THRESHOLD = 0.20
INSTANCE_MASK_THRESHOLD = 0.40
CROP_MARGIN = 20

IMAGE_PATH = "meibo4.jpg"
CONTOUR_PATH = None
CHECKPOINT_PATH = "best_model_mask2former (5).pth"
DEFAULT_MODEL_WEIGHTS = CHECKPOINT_PATH
OUTPUT_DIR = "inference_outputs_mask2former_768"


def resize_with_padding_image(img_np, output_size=OUTPUT_SIZE):
    target_w, target_h = output_size
    h, w = img_np.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Imagen inválida con shape: {img_np.shape}")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_padded = np.zeros((target_h, target_w, 3), dtype=img_np.dtype)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    img_padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = img_resized

    meta = {
        "original_h": h,
        "original_w": w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "scale": scale,
    }
    return img_padded, meta


def crop_by_eyelid_bbox(img_np, contour_path, margin=CROP_MARGIN):
    if not contour_path or not os.path.exists(contour_path):
        if contour_path:
            print(f"[ADVERTENCIA] No existe contorno: {contour_path}")
        return img_np, None

    eyelid = Image.open(contour_path).convert("L")
    eyelid_np = (np.array(eyelid) > 0).astype(np.uint8)
    if eyelid_np.shape[:2] != img_np.shape[:2]:
        eyelid_np = cv2.resize(
            eyelid_np,
            (img_np.shape[1], img_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    ys, xs = np.where(eyelid_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        print("[ADVERTENCIA] Contorno vacío. Imagen completa.")
        return img_np, None

    y1 = max(0, int(ys.min()) - margin)
    y2 = min(img_np.shape[0], int(ys.max()) + margin + 1)
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(img_np.shape[1], int(xs.max()) + margin + 1)
    crop_meta = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return img_np[y1:y2, x1:x2], crop_meta


def apply_clahe_rgb(img_np):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


def preprocess_for_inference(image_path, contour_path=None, crop_margin=CROP_MARGIN):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    full_h, full_w = img_np.shape[:2]

    img_crop, crop_meta = crop_by_eyelid_bbox(img_np, contour_path, margin=crop_margin)
    img_resized, resize_meta = resize_with_padding_image(img_crop, OUTPUT_SIZE)
    img_processed = apply_clahe_rgb(img_resized)

    return img_np, img_processed, crop_meta, resize_meta, full_h, full_w


def pred_padded_to_original_canvas(pred_model, crop_meta, resize_meta, full_h, full_w):
    pad_y = int(resize_meta["pad_y"])
    pad_x = int(resize_meta["pad_x"])
    new_h = int(resize_meta["new_h"])
    new_w = int(resize_meta["new_w"])
    ow = int(resize_meta["original_w"])
    oh = int(resize_meta["original_h"])

    sl = pred_model[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
    pred_crop = cv2.resize(
        sl.astype(np.float32),
        (ow, oh),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)

    if crop_meta is None:
        return pred_crop

    x1, y1, x2, y2 = crop_meta["x1"], crop_meta["y1"], crop_meta["x2"], crop_meta["y2"]
    pred_full = np.zeros((full_h, full_w), dtype=np.int32)
    pred_full[y1:y2, x1:x2] = pred_crop
    return pred_full


_LOCAL_CFG = "/app/mask2former_config"

def build_model():
    src = _LOCAL_CFG if os.path.isdir(_LOCAL_CFG) else MODEL_ID
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        src,
        num_labels=2,
        ignore_mismatched_sizes=True,
        local_files_only=os.path.isdir(_LOCAL_CFG),
    )
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.config.num_labels = 2
    model.config.num_queries = NUM_QUERIES
    model.config.is_thing_map = {0: False, 1: True}
    return model


def load_checkpoint(model, checkpoint_path, device):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean, strict=False)
    return model


@torch.no_grad()
def predict_instances(model, processor, img_processed, device):
    pixel_values = transforms.ToTensor()(
        Image.fromarray(img_processed.astype(np.uint8))
    ).unsqueeze(0).to(device)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        outputs = model(pixel_values=pixel_values)

    H, W = img_processed.shape[:2]
    pred_inst_dict = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[(H, W)],
        threshold=INSTANCE_SCORE_THRESHOLD,
        mask_threshold=INSTANCE_MASK_THRESHOLD,
    )[0]

    seg_map = pred_inst_dict["segmentation"].detach().cpu().numpy()
    instance_map = np.zeros((H, W), dtype=np.int32)
    k = 1
    segments = []
    num_raw = 0

    for seg_info in pred_inst_dict.get("segments_info", []):
        if int(seg_info["label_id"]) != 1:
            continue
        num_raw += 1
        mask = seg_map == int(seg_info["id"])
        area = int(mask.sum())
        if area < MIN_INSTANCE_AREA:
            continue
        instance_map[mask] = k
        segments.append({
            "id": k,
            "score": float(seg_info.get("score", 0.0)),
            "area": area,
        })
        k += 1

    return instance_map, (instance_map > 0).astype(np.uint8), segments, num_raw


def write_result_files(img_orig, img_processed, instance_map, binary_mask, overlay, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(img_orig).save(os.path.join(output_dir, "00_original.png"))
    Image.fromarray(img_processed).save(os.path.join(output_dir, "01_processed_768.png"))
    Image.fromarray((binary_mask * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "02_binary_mask.png")
    )
    if instance_map.max() > 0:
        colored = label2rgb(instance_map, image=None, bg_label=0)
        rgb = (np.clip(colored, 0, 1) * 255).astype(np.uint8)
    else:
        rgb = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(os.path.join(output_dir, "03_instances_colored.png"))
    Image.fromarray(overlay).save(os.path.join(output_dir, "04_overlay_full_res.png"))


@torch.no_grad()
def run_inference(
    image_path,
    model_path,
    contour_path=None,
    output_dir=OUTPUT_DIR,
    crop_margin=CROP_MARGIN,
    save_outputs=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mask2former_768] Device: {device}")

    img_orig, img_processed, crop_meta, resize_meta, full_h, full_w = preprocess_for_inference(
        image_path,
        contour_path=contour_path,
        crop_margin=crop_margin,
    )

    model = build_model()
    model = load_checkpoint(model, model_path, device)
    model.to(device).eval()

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    instance_map, binary_model, segments, num_raw = predict_instances(
        model, processor, img_processed, device
    )

    pred_count_model = int(instance_map.max())
    pred_full = pred_padded_to_original_canvas(
        instance_map, crop_meta, resize_meta, full_h, full_w
    )
    pred_count_full = int(pred_full.max())

    overlay_rgb = label2rgb(
        pred_full,
        image=img_orig.astype(np.float64) / 255.0,
        bg_label=0,
        alpha=0.45,
    )
    overlay_u8 = (np.clip(overlay_rgb, 0, 1) * 255).astype(np.uint8)

    result = {
        "img_original": img_orig.astype(np.uint8),
        "img_model_input": img_processed.astype(np.uint8),
        "pred_binary": (pred_full > 0).astype(np.uint8),
        "pred_instance": pred_full,
        "overlay_rgb": overlay_u8,
        "num_raw": int(num_raw),
        "pred_count_512": pred_count_model,
        "pred_count_original": pred_count_full,
        "device": str(device),
        "image_path": image_path,
        "model_path": model_path,
        "segments": segments,
    }

    if save_outputs:
        write_result_files(
            img_orig.astype(np.uint8),
            img_processed,
            instance_map,
            binary_model,
            overlay_u8,
            output_dir,
        )

    return result


def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"No existe IMAGE_PATH: {IMAGE_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"No existe CHECKPOINT_PATH: {CHECKPOINT_PATH}")

    res = run_inference(
        image_path=IMAGE_PATH,
        model_path=CHECKPOINT_PATH,
        contour_path=CONTOUR_PATH,
        output_dir=OUTPUT_DIR,
        save_outputs=True,
    )
    print(f"[RESULTADO] {res['pred_count_original']} glándulas detectadas")


if __name__ == "__main__":
    main()


# ============================================================
# Server helpers (called once at startup by main.py)
# ============================================================

def load_model_server(checkpoint_path, device=None):
    """Pre-load model + processor for server use."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model = load_checkpoint(model, checkpoint_path, device)
    model.to(device).eval()
    _proc_src = _LOCAL_CFG if os.path.isdir(_LOCAL_CFG) else MODEL_ID
    processor = AutoImageProcessor.from_pretrained(
        _proc_src, use_fast=False, local_files_only=os.path.isdir(_LOCAL_CFG)
    )
    return model, processor


def infer_with_model(image_path, model, processor, device=None, contour_path=None):
    """Run inference and return instance map in original image coordinates."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, img_processed, crop_meta, resize_meta, full_h, full_w = preprocess_for_inference(
        image_path, contour_path=contour_path, crop_margin=CROP_MARGIN
    )
    instance_map, _, _, _ = predict_instances(model, processor, img_processed, device)
    return pred_padded_to_original_canvas(instance_map, crop_meta, resize_meta, full_h, full_w)
