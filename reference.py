#!/usr/bin/env python3
# infer_v3_single_image.py

import os
import json
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image

from skimage.color import label2rgb
from torchvision import transforms
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerConfig,
    AutoImageProcessor,
)

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "facebook/mask2former-swin-small-cityscapes-instance"

# Versión v3: background + gland
ID2LABEL = {0: "background", 1: "gland"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

NUM_QUERIES = 20
OUTPUT_SIZE = (512, 512)  # (W, H)

INSTANCE_SCORE_THRESHOLD = 0.45
INSTANCE_MASK_THRESHOLD = 0.40
MIN_INSTANCE_AREA = 20
CROP_MARGIN = 20

DEFAULT_CKPT = "best_model.pth"

# ============================================================
# MODELO
# ============================================================

def build_model():
    config = Mask2FormerConfig.from_pretrained(MODEL_ID)
    config.num_labels = 2
    config.num_queries = NUM_QUERIES
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    config.is_thing_map = {0: False, 1: True}

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.config.is_thing_map = {0: False, 1: True}
    model.config.num_queries = NUM_QUERIES

    return model


def load_checkpoint(model, ckpt_path, device):
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state, strict=True)
    return model


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def preprocess_image(image_path, contour_path=None):
    """
    Replica la lógica de entrenamiento:
    - RGB
    - opcionalmente aplicar máscara de contorno
    - crop por bounding box del eyelid
    - resize a OUTPUT_SIZE
    - normalize ImageNet
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    if contour_path is not None and os.path.exists(contour_path):
        contour = Image.open(contour_path).convert("L")
        contour_np = (np.array(contour) > 0).astype(np.uint8)

        # Aplica máscara del párpado
        img_np = img_np * contour_np[..., None]

        ys, xs = np.where(contour_np > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1 = max(0, xs.min() - CROP_MARGIN)
            x2 = min(img_np.shape[1], xs.max() + 1 + CROP_MARGIN)
            y1 = max(0, ys.min() - CROP_MARGIN)
            y2 = min(img_np.shape[0], ys.max() + 1 + CROP_MARGIN)
            img_np = img_np[y1:y2, x1:x2]

    # Resize al tamaño de entrenamiento
    img_resized = cv2.resize(
        img_np,
        OUTPUT_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    img_tensor = transform(Image.fromarray(img_resized.astype(np.uint8)))

    return img_tensor, img_resized


# ============================================================
# POSTPROCESO
# ============================================================

def parse_instance_output(inst, H, W, score_threshold=0.45, min_area=20):
    """
    Convierte output de post_process_instance_segmentation a mapa HxW.
    Devuelve:
    - pred_instance: mapa de instancias
    - num_pred_masks_raw: número bruto de máscaras
    - kept_segments: info de segmentos retenidos
    """
    pred_instance = np.zeros((H, W), dtype=np.int32)
    num_pred_masks_raw = 0
    kept_segments = []

    seg = inst["segmentation"]
    seg_np = seg.cpu().numpy()

    scores = inst.get("scores", None)
    if scores is not None:
        scores = scores.cpu().numpy()

    labels = inst.get("labels", None)
    if labels is not None:
        labels = labels.cpu().numpy()

    segments_info = inst.get("segments_info", None)

    # Caso 1: mapa 2D de ids
    if seg_np.ndim == 2:
        unique_ids = np.unique(seg_np)
        unique_ids = unique_ids[unique_ids != 0]
        num_pred_masks_raw = len(unique_ids)

        k = 1

        if segments_info is not None:
            for seginfo in segments_info:
                sid = seginfo["id"]
                lab = seginfo.get("label_id", 1)
                score = seginfo.get("score", 1.0)

                # En v3 tenemos background=0 y gland=1
                if lab == 0:
                    continue

                if score < score_threshold:
                    continue

                region = (seg_np == sid)
                area = int(region.sum())

                if area < min_area:
                    continue

                pred_instance[region] = k
                kept_segments.append({
                    "instance_id": k,
                    "raw_segment_id": int(sid),
                    "label_id": int(lab),
                    "score": float(score),
                    "area": area
                })
                k += 1

        else:
            for sid in unique_ids:
                region = (seg_np == sid)
                area = int(region.sum())

                if area < min_area:
                    continue

                pred_instance[region] = k
                kept_segments.append({
                    "instance_id": k,
                    "raw_segment_id": int(sid),
                    "label_id": 1,
                    "score": 1.0,
                    "area": area
                })
                k += 1

    # Caso 2: stack de máscaras
    elif seg_np.ndim == 3:
        num_pred_masks_raw = seg_np.shape[0]
        k = 1

        for i, m in enumerate(seg_np):
            score = scores[i] if scores is not None and i < len(scores) else 1.0
            lab = labels[i] if labels is not None and i < len(labels) else 1

            if lab == 0:
                continue

            if score < score_threshold:
                continue

            region = m.astype(bool)
            area = int(region.sum())

            if area < min_area:
                continue

            pred_instance[region] = k
            kept_segments.append({
                "instance_id": k,
                "raw_segment_id": int(i),
                "label_id": int(lab),
                "score": float(score),
                "area": area
            })
            k += 1

    else:
        raise ValueError(f"Formato inesperado de segmentation: {seg_np.shape}")

    return pred_instance, num_pred_masks_raw, kept_segments


# ============================================================
# INFERENCIA
# ============================================================

@torch.no_grad()
def predict_single_image(
    image_path,
    contour_path=None,
    ckpt_path=DEFAULT_CKPT,
    output_dir="inference_outputs_v3",
    score_threshold=INSTANCE_SCORE_THRESHOLD,
    mask_threshold=INSTANCE_MASK_THRESHOLD,
):
    os.makedirs(output_dir, exist_ok=True)
    instances_dir = os.path.join(output_dir, "instances")
    os.makedirs(instances_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar modelo
    model = build_model()
    model = load_checkpoint(model, ckpt_path, device)
    model.to(device).eval()

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)

    # Preprocesar
    img_tensor, img_vis = preprocess_image(image_path, contour_path)
    H, W = img_tensor.shape[1], img_tensor.shape[2]

    # Inferencia
    pixel_values = img_tensor.unsqueeze(0).to(device)

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        outputs = model(pixel_values=pixel_values)

    # Post-process de instance segmentation
    inst = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[(H, W)],
        threshold=score_threshold,
        mask_threshold=mask_threshold,
    )[0]

    pred_instance, raw_count, kept_segments = parse_instance_output(
        inst,
        H,
        W,
        score_threshold=score_threshold,
        min_area=MIN_INSTANCE_AREA
    )

    pred_count = int(pred_instance.max())
    binary_mask = (pred_instance > 0).astype(np.uint8)

    # Guardar mapa de instancias
    instance_map_path = os.path.join(output_dir, "pred_instance_map.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(pred_instance, cmap="nipy_spectral")
    plt.axis("off")
    plt.title(f"Predicted instances: {pred_count}")
    plt.tight_layout()
    plt.savefig(instance_map_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Guardar máscara binaria
    binary_mask_path = os.path.join(output_dir, "pred_binary_mask.png")
    cv2.imwrite(binary_mask_path, binary_mask * 255)

    # Overlay sobre la imagen
    img_float = img_vis.astype(np.float32) / 255.0
    overlay = label2rgb(pred_instance, image=img_float, bg_label=0, alpha=0.35)

    overlay_path = os.path.join(output_dir, "pred_overlay.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Instances={pred_count} | raw_masks={raw_count}")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Guardar máscaras individuales
    for k in range(1, pred_count + 1):
        inst_mask = (pred_instance == k).astype(np.uint8) * 255
        inst_path = os.path.join(instances_dir, f"instance_{k:03d}.png")
        cv2.imwrite(inst_path, inst_mask)

    # Resumen JSON
    summary = {
        "image_path": image_path,
        "contour_path": contour_path,
        "checkpoint": ckpt_path,
        "image_size": [int(H), int(W)],
        "pred_count": pred_count,
        "raw_pred_masks": int(raw_count),
        "score_threshold": float(score_threshold),
        "mask_threshold": float(mask_threshold),
        "min_instance_area": int(MIN_INSTANCE_AREA),
        "segments": kept_segments,
    }

    summary_path = os.path.join(output_dir, "prediction_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== INFERENCE DONE ===")
    print(f"Imagen: {image_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Instancias predichas: {pred_count}")
    print(f"Masks crudas: {raw_count}")
    print(f"Overlay guardado en: {overlay_path}")
    print(f"Instance map guardado en: {instance_map_path}")
    print(f"Binary mask guardada en: {binary_mask_path}")
    print(f"Resumen JSON guardado en: {summary_path}")

    if len(kept_segments) > 0:
        print("\nTop segmentos:")
        for s in kept_segments[:10]:
            print(
                f"  inst={s['instance_id']} | score={s['score']:.3f} | area={s['area']}"
            )

    return pred_instance, summary


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia para una nueva imagen usando ddp_segmentation_v3.py"
    )
    parser.add_argument("--image_path", type=str, required=True, help="Ruta a la imagen RGB")
    parser.add_argument(
        "--contour_path",
        type=str,
        default=None,
        help="Ruta opcional a la máscara de contorno del párpado"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CKPT,
        help="Ruta al checkpoint best_model.pth"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_outputs_v3",
        help="Directorio de salida"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=INSTANCE_SCORE_THRESHOLD,
        help="Threshold de score para filtrar máscaras"
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=INSTANCE_MASK_THRESHOLD,
        help="Threshold de máscara para postprocesado"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"No existe la imagen: {args.image_path}")

    if args.contour_path is not None and not os.path.exists(args.contour_path):
        raise FileNotFoundError(f"No existe contour_path: {args.contour_path}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"No existe checkpoint: {args.checkpoint}")

    predict_single_image(
        image_path=args.image_path,
        contour_path=args.contour_path,
        ckpt_path=args.checkpoint,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
    )


if __name__ == "__main__":
    IMAGE_PATH = "meibomio4.jpg"
    CONTOUR_PATH = "meibomio4_mask.png"
    CKPT_PATH = "best_model (13).pth"
    OUTPUT_DIR = "inference_outputs4"

    predict_single_image(
        image_path=IMAGE_PATH,
        contour_path=CONTOUR_PATH,
        ckpt_path=CKPT_PATH,
        output_dir=OUTPUT_DIR
    )