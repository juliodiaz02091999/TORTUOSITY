#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from skimage.color import label2rgb


# ======================================================
# CONFIGURACIÓN
# ======================================================

MIN_INSTANCE_AREA = 20

SCORE_THRESHOLD = 0.50
MASK_THRESHOLD = 0.50

APPLY_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILES = 8

CROP_MARGIN = 20

# Default checkpoint usado en __main__; en infer_server (maskcrnn_py2) también por defecto (20).pth
DEFAULT_MODEL_WEIGHTS = "best_model (18).pth"


# ======================================================
# MODELO
# ======================================================


def build_model(pretrained=False):
    """
    Reconstruye la misma arquitectura usada en entrenamiento:
    Mask R-CNN ResNet50 FPN con 2 clases:
    - clase 0: fondo
    - clase 1: glándula
    """

    if pretrained:
        try:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights="DEFAULT"
            )
        except Exception:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=True
            )
    else:
        try:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None
            )
        except Exception:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=False
            )

    num_classes = 2

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


# ======================================================
# PREPROCESAMIENTO
# ======================================================

def apply_clahe_rgb(img_rgb_np, roi_mask=None, clip=2.0, tiles=(8, 8)):
    """
    Aplica CLAHE en el canal L del espacio LAB.
    Si roi_mask existe, solo reemplaza dentro del ROI.
    """

    img_bgr = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=tiles
    )

    L_eq = clahe.apply(L)

    if roi_mask is not None:
        m = roi_mask.astype(bool)
        L_out = L.copy()
        L_out[m] = L_eq[m]
    else:
        L_out = L_eq

    lab_eq = cv2.merge([L_out, A, B])
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    rgb_eq = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2RGB)

    return rgb_eq


def crop_by_eyelid(img_np, eyelid_mask_np, crop_margin=20):
    """
    Recorta la imagen usando la máscara del párpado.
    Devuelve:
    - imagen recortada
    - máscara de párpado recortada
    - coordenadas del recorte: x1, y1, x2, y2
    """

    ys, xs = np.where(eyelid_mask_np > 0)

    if len(xs) == 0 or len(ys) == 0:
        h, w = img_np.shape[:2]
        return img_np, eyelid_mask_np, (0, 0, w, h)

    x1 = max(0, xs.min() - crop_margin)
    x2 = min(img_np.shape[1], xs.max() + 1 + crop_margin)

    y1 = max(0, ys.min() - crop_margin)
    y2 = min(img_np.shape[0], ys.max() + 1 + crop_margin)

    img_crop = img_np[y1:y2, x1:x2]
    eyelid_crop = eyelid_mask_np[y1:y2, x1:x2]

    return img_crop, eyelid_crop, (x1, y1, x2, y2)


def resize_prediction_to_original(pred_crop, crop_info):
    x1, x2 = crop_info["x1"], crop_info["x2"]
    y1, y2 = crop_info["y1"], crop_info["y2"]
    orig_h, orig_w = crop_info["orig_h"], crop_info["orig_w"]
    crop_h, crop_w = y2 - y1, x2 - x1

    if pred_crop.shape[:2] != (crop_h, crop_w):
        pred_crop = cv2.resize(
            pred_crop.astype(np.int32),
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST,
        )

    pred_original = np.zeros((orig_h, orig_w), dtype=np.int32)
    pred_original[y1:y2, x1:x2] = pred_crop
    return pred_original


def resize_prediction_full_image(pred_crop, orig_h, orig_w):
    return cv2.resize(
        pred_crop.astype(np.int32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )


def _normalize_state_dict(state):
    if not isinstance(state, dict):
        return state
    if any(k.startswith("module.") for k in state):
        return {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
    return state


def preprocess_image(
    image_path,
    contour_path=None,
    crop_margin=None,
):
    """
    Devuelve:
    - img_np_original: RGB completa (EXIF)
    - img_np_crop: región para el modelo (recorte + CLAHE)
    - img_tensor: tensor listo para Mask R-CNN
    - crop_info: dict para re-mapear predicciones a la imagen original
    """
    margin = CROP_MARGIN if crop_margin is None else int(crop_margin)
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    img_np_original = np.array(img, dtype=np.uint8)

    if contour_path is not None and os.path.exists(contour_path):
        eyelid_mask = Image.open(contour_path).convert("L")
        eyelid_mask_np = (np.array(eyelid_mask) > 0).astype(np.uint8)

        if eyelid_mask_np.shape[:2] != img_np_original.shape[:2]:
            eyelid_mask_np = cv2.resize(
                eyelid_mask_np,
                (img_np_original.shape[1], img_np_original.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        img_masked = img_np_original * eyelid_mask_np[..., None]

        img_crop, _eyelid_crop, crop_tuple = crop_by_eyelid(
            img_masked,
            eyelid_mask_np,
            crop_margin=margin,
        )
        x1, y1, x2, y2 = crop_tuple
        crop_info = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "orig_h": img_np_original.shape[0],
            "orig_w": img_np_original.shape[1],
        }
        img_np_for_model = img_crop
    else:
        h, w = img_np_original.shape[:2]
        img_np_for_model = img_np_original.copy()
        crop_info = {
            "x1": 0,
            "y1": 0,
            "x2": w,
            "y2": h,
            "orig_h": h,
            "orig_w": w,
        }

    if APPLY_CLAHE:
        roi_mask = img_np_for_model.sum(axis=2) > 0
        img_np_for_model = apply_clahe_rgb(
            img_np_for_model,
            roi_mask=roi_mask,
            clip=CLAHE_CLIP,
            tiles=(CLAHE_TILES, CLAHE_TILES),
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    img_tensor = transform(Image.fromarray(img_np_for_model.astype(np.uint8)))
    return img_np_original, img_np_for_model, img_tensor, crop_info


def unnormalize_image(img_tensor):
    """
    Desnormaliza tensor con mean=std=0.5.
    """

    img = img_tensor.detach().cpu().mul(0.5).add(0.5)
    img = img.permute(1, 2, 0).numpy()

    return np.clip(img, 0, 1)


# ======================================================
# INFERENCIA
# ======================================================


@torch.no_grad()
def run_inference(
    image_path,
    model_path,
    contour_path=None,
    output_dir="inference_outputs_maskcrnn_py",
    crop_margin=CROP_MARGIN,
    score_threshold=SCORE_THRESHOLD,
    mask_threshold=MASK_THRESHOLD,
    min_instance_area=MIN_INSTANCE_AREA,
    save_outputs=False,
):
    """
    Misma forma de retorno que maskcrnn3.run_inference (API / infer_server).
    """
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[maskcrnn_2.py] dispositivo: {device}")

    model = build_model(pretrained=False)
    try:
        state = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(model_path, map_location=device)

    state = _normalize_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    img_np_original, img_np_crop, img_tensor, crop_info = preprocess_image(
        image_path=image_path,
        contour_path=contour_path,
        crop_margin=crop_margin,
    )

    prediction = model([img_tensor.to(device)])[0]

    masks = prediction["masks"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()

    keep = (scores >= score_threshold) & (labels == 1)
    num_raw = int(keep.sum())

    selected_masks = masks[keep, 0]

    H, W = img_tensor.shape[1], img_tensor.shape[2]
    pred_instance_crop = np.zeros((H, W), dtype=np.int32)

    k = 1
    for m in selected_masks:
        binary_mask = (m > mask_threshold).astype(np.uint8)
        if binary_mask.sum() < min_instance_area:
            continue
        pred_instance_crop[binary_mask > 0] = k
        k += 1

    pred_instance_original = resize_prediction_to_original(
        pred_instance_crop,
        crop_info,
    )

    pred_binary_original = (pred_instance_original > 0).astype(np.uint8)
    pred_count = int(pred_instance_original.max())

    overlay_rgb = label2rgb(
        pred_instance_original,
        image=img_np_original.astype(np.float64) / 255.0,
        bg_label=0,
        alpha=0.45,
    )
    overlay_u8 = (np.clip(overlay_rgb, 0, 1) * 255).astype(np.uint8)

    result = {
        "img_original": img_np_original,
        "img_model_input": img_np_crop,
        "pred_binary": pred_binary_original,
        "pred_instance": pred_instance_original,
        "overlay_rgb": overlay_u8,
        "num_raw": num_raw,
        "pred_count_512": pred_count,
        "pred_count_original": pred_count,
        "device": str(device),
        "image_path": image_path,
        "model_path": model_path,
    }

    if not save_outputs:
        return result

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    binary_path = os.path.join(output_dir, f"{base_name}_binary_mask.png")
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(binary_path, (pred_binary_original * 255).astype(np.uint8))
    plt.imsave(overlay_path, overlay_rgb)
    result["saved_paths"] = {"binary": binary_path, "overlay": overlay_path}
    print(f"[maskcrnn_2.py] Guardado: {binary_path}, {overlay_path}")
    return result


@torch.no_grad()
def predict_single_image(
    image_path,
    model_path,
    contour_path=None,
    output_dir="single_inference_output",
    score_threshold=SCORE_THRESHOLD,
    mask_threshold=MASK_THRESHOLD,
    min_instance_area=MIN_INSTANCE_AREA
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Usando dispositivo: {device}")

    model = build_model(pretrained=False)

    try:
        state = torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        state = torch.load(
            model_path,
            map_location=device
        )

    state = _normalize_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    img_np_original, img_np_processed, img_tensor, crop_info = preprocess_image(
        image_path=image_path,
        contour_path=contour_path,
    )

    image_gpu = img_tensor.to(device)

    prediction = model([image_gpu])[0]

    masks = prediction["masks"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()

    keep = (scores >= score_threshold) & (labels == 1)

    selected_masks = masks[keep, 0]
    selected_scores = scores[keep]
    selected_boxes = boxes[keep]

    H, W = img_tensor.shape[1], img_tensor.shape[2]

    pred_instance = np.zeros((H, W), dtype=np.int32)

    instance_id = 1

    valid_scores = []
    valid_boxes = []

    for m, score, box in zip(selected_masks, selected_scores, selected_boxes):
        binary_mask = (m > mask_threshold).astype(np.uint8)

        if binary_mask.sum() < min_instance_area:
            continue

        pred_instance[binary_mask > 0] = instance_id

        valid_scores.append(float(score))
        valid_boxes.append(box.tolist())

        instance_id += 1

    pred_count = int(pred_instance.max())

    pred_binary = (pred_instance > 0).astype(np.uint8)

    # Guardar máscaras
    binary_path = os.path.join(output_dir, "pred_binary_mask.png")
    instance_path = os.path.join(output_dir, "pred_instance_mask.npy")
    overlay_path = os.path.join(output_dir, "prediction_overlay.png")
    figure_path = os.path.join(output_dir, "prediction_visualization.png")

    cv2.imwrite(binary_path, pred_binary * 255)
    np.save(instance_path, pred_instance)

    # Overlay sobre imagen procesada
    instance_rgb = label2rgb(pred_instance, image=img_np_processed / 255.0, bg_label=0)

    plt.imsave(overlay_path, instance_rgb)

    # Figura resumen
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    axes[0].imshow(img_np_processed)
    axes[0].set_title("Imagen preprocesada")
    axes[0].axis("off")

    axes[1].imshow(pred_binary, cmap="gray")
    axes[1].set_title("Predicción binaria")
    axes[1].axis("off")

    axes[2].imshow(label2rgb(pred_instance, bg_label=0))
    axes[2].set_title(f"Instancias predichas: {pred_count}")
    axes[2].axis("off")

    axes[3].imshow(instance_rgb)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n==============================")
    print("RESULTADO DE INFERENCIA")
    print("==============================")
    print(f"Imagen: {image_path}")
    print(f"Modelo: {model_path}")
    crop_box = (
        crop_info["x1"],
        crop_info["y1"],
        crop_info["x2"],
        crop_info["y2"],
    )
    print(f"Contorno usado: {contour_path if contour_path else 'No'}")
    print(f"Crop box: {crop_box}")
    print(f"Número de instancias predichas: {pred_count}")
    print(f"Scores válidos: {[round(s, 4) for s in valid_scores]}")
    print("\nArchivos guardados:")
    print(f"- Máscara binaria: {binary_path}")
    print(f"- Máscara de instancias .npy: {instance_path}")
    print(f"- Overlay: {overlay_path}")
    print(f"- Visualización: {figure_path}")

    return {
        "pred_count": pred_count,
        "scores": valid_scores,
        "boxes": valid_boxes,
        "binary_mask_path": binary_path,
        "instance_mask_path": instance_path,
        "overlay_path": overlay_path,
        "figure_path": figure_path,
    }


# ======================================================
# SERVER INFERENCE (pre-loaded model)
# ======================================================

@torch.no_grad()
def infer_with_model(
    image_path,
    model,
    device,
    contour_path=None,
    score_threshold=SCORE_THRESHOLD,
    mask_threshold=MASK_THRESHOLD,
    min_instance_area=MIN_INSTANCE_AREA,
    crop_margin=CROP_MARGIN,
):
    """Pre-loaded model inference — used by the FastAPI server."""
    _, _, img_tensor, crop_info = preprocess_image(
        image_path=image_path,
        contour_path=contour_path,
        crop_margin=crop_margin,
    )

    prediction = model([img_tensor.to(device)])[0]

    masks  = prediction["masks"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()

    keep = (scores >= score_threshold) & (labels == 1)
    selected_masks = masks[keep, 0]

    H, W = img_tensor.shape[1], img_tensor.shape[2]
    pred_instance_crop = np.zeros((H, W), dtype=np.int32)

    k = 1
    for m in selected_masks:
        binary = (m > mask_threshold).astype(np.uint8)
        if binary.sum() < min_instance_area:
            continue
        pred_instance_crop[binary > 0] = k
        k += 1

    return resize_prediction_to_original(pred_instance_crop, crop_info)


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    # ==========================
    # RUTAS
    # ==========================

    image_path = "meibomio1.jpg"

    model_path = DEFAULT_MODEL_WEIGHTS

    # Opcional.
    # Si tienes máscara/contorno de párpado, coloca la ruta.
    # Si no tienes, deja contour_path = None.
    contour_path = None
    # contour_path = "/kaggle/input/datasets/sneh1619/mgd1k-dataset/Expore MGD1k Dataset/Eyelid Lebels/imagen.png"

    output_dir = "inferencia_nueva_imagen"

    # ==========================
    # PARÁMETROS DE INFERENCIA
    # ==========================

    score_threshold = 0.50
    mask_threshold = 0.50
    min_instance_area = 20

    # ==========================
    # EJECUCIÓN
    # ==========================

    predict_single_image(
        image_path=image_path,
        model_path=model_path,
        contour_path=contour_path,
        output_dir=output_dir,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        min_instance_area=min_instance_area
    )