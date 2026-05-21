#!/usr/bin/env python3

import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from PIL import Image
from skimage.color import label2rgb
from torchvision import transforms
from transformers import (
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
)

# ============================================================
# RUTAS
# ============================================================

IMAGE_PATH = "meibomio4.jpg"

EYELID_PATH = None

CHECKPOINT_PATH = "best_model_mask2former_1024.pth"
DEFAULT_MODEL_WEIGHTS = CHECKPOINT_PATH

OUTPUT_DIR = "inference_outputs8"

USE_EYELID_CROP = True


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_ID = "facebook/mask2former-swin-small-cityscapes-instance"

ID2LABEL = {0: "background", 1: "gland"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

OUTPUT_SIZE = (1024, 1024)

NUM_QUERIES = 150

# Si no detecta nada, baja score a 0.20 y min_area a 30.
MIN_INSTANCE_AREA = 30
INSTANCE_SCORE_THRESHOLD = 0.20
INSTANCE_MASK_THRESHOLD = 0.40

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _get_device()


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def resize_with_padding_image_and_mask(img_np, mask_np=None, output_size=(1024, 1024)):
    target_w, target_h = output_size
    h, w = img_np.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError(f"Imagen inválida con shape: {img_np.shape}")

    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = cv2.resize(
        img_np,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    img_padded = np.zeros((target_h, target_w, 3), dtype=img_np.dtype)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    img_padded[
        pad_y:pad_y + new_h,
        pad_x:pad_x + new_w,
    ] = img_resized

    info = {
        "scale": scale,
        "new_w": new_w,
        "new_h": new_h,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "orig_w": w,
        "orig_h": h,
    }

    if mask_np is None:
        return img_padded, None, info

    mask_resized = cv2.resize(
        mask_np,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )

    mask_padded = np.zeros((target_h, target_w), dtype=mask_np.dtype)

    mask_padded[
        pad_y:pad_y + new_h,
        pad_x:pad_x + new_w,
    ] = mask_resized

    return img_padded, mask_padded, info


def apply_clahe_rgb(img_np):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    out = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return out


def preprocess_image(
    image_path,
    eyelid_path=None,
    crop_to_eyelid=True,
    crop_margin=20,
):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    eyelid_mask_np = None

    if eyelid_path is not None and os.path.exists(eyelid_path):
        eyelid = Image.open(eyelid_path).convert("L")
        eyelid_mask_np = (np.array(eyelid) > 0).astype(np.uint8)

        if eyelid_mask_np.shape[:2] != img_np.shape[:2]:
            eyelid_mask_np = cv2.resize(
                eyelid_mask_np,
                (img_np.shape[1], img_np.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
    else:
        print("Advertencia: no se encontró máscara de párpado. Se usará imagen completa.")
        eyelid_mask_np = None

    crop_info = None

    if crop_to_eyelid and eyelid_mask_np is not None:
        img_np = img_np * eyelid_mask_np[..., None]

        ys, xs = np.where(eyelid_mask_np > 0)

        if len(xs) > 0 and len(ys) > 0:
            m = int(crop_margin)
            y1 = max(0, int(ys.min()) - m)
            y2 = min(img_np.shape[0], int(ys.max()) + 1 + m)
            x1 = max(0, int(xs.min()) - m)
            x2 = min(img_np.shape[1], int(xs.max()) + 1 + m)

            crop_info = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "orig_h": img_np.shape[0],
                "orig_w": img_np.shape[1],
            }

            img_np = img_np[y1:y2, x1:x2]

    img_padded, _, pad_info = resize_with_padding_image_and_mask(
        img_np,
        mask_np=None,
        output_size=OUTPUT_SIZE,
    )

    img_padded = apply_clahe_rgb(img_padded)

    img_tensor = transforms.ToTensor()(
        Image.fromarray(img_padded.astype(np.uint8))
    )

    return img_tensor, img_padded, crop_info, pad_info


# ============================================================
# MODELO
# ============================================================

_LOCAL_CFG = "/app/mask2former_config"

def build_model():
    import os
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


def _normalize_state_dict(raw):
    if isinstance(raw, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                raw = raw[k]
                break
    if not isinstance(raw, dict):
        return raw
    out = {}
    for k, v in raw.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def load_trained_model(checkpoint_path, device=None):
    if device is None:
        device = _get_device()

    model = build_model()

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = _normalize_state_dict(checkpoint)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Pesos cargados desde: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()

    return model


# ============================================================
# INFERENCIA
# ============================================================

def predict_instances(model, processor, img_tensor, device=None):
    if device is None:
        device = _get_device()

    pixel_values = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast(
        "cuda",
        enabled=device.type == "cuda",
    ):
        outputs = model(pixel_values=pixel_values)

    H, W = img_tensor.shape[1:]

    pred = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[(H, W)],
        threshold=INSTANCE_SCORE_THRESHOLD,
        mask_threshold=INSTANCE_MASK_THRESHOLD,
    )[0]

    seg_map = pred["segmentation"].detach().cpu().numpy()

    pred_instance = np.zeros((H, W), dtype=np.int32)

    instance_id = 1
    kept_segments = []

    print("\nSegmentos crudos devueltos por Mask2Former:")
    print(f"Total segments_info: {len(pred.get('segments_info', []))}")

    for seg_info in pred.get("segments_info", []):
        label_id = int(seg_info["label_id"])
        raw_score = float(seg_info.get("score", 0.0))
        seg_id = int(seg_info["id"])

        mask = seg_map == seg_id
        area = int(mask.sum())

        print(
            f"  raw segment_id={seg_id} | "
            f"label_id={label_id} | "
            f"score={raw_score:.4f} | "
            f"area={area}"
        )

        if label_id != 1:
            continue

        if area < MIN_INSTANCE_AREA:
            continue

        pred_instance[mask] = instance_id

        kept_segments.append({
            "instance_id": instance_id,
            "segment_id": seg_id,
            "label_id": label_id,
            "score": raw_score,
            "area": area,
        })

        instance_id += 1

    pred_binary = (pred_instance > 0).astype(np.uint8)

    num_raw = sum(
        1
        for seg_info in pred.get("segments_info", [])
        if int(seg_info.get("label_id", -1)) == 1
    )

    print("\nResumen de predicción filtrada:")
    print(f"  pred_instance dtype: {pred_instance.dtype}")
    print(f"  pred_instance min: {pred_instance.min()}")
    print(f"  pred_instance max: {pred_instance.max()}")
    print(f"  unique values: {np.unique(pred_instance)[:100]}")
    print(f"  instancias conservadas: {int(pred_instance.max())}")

    return pred_instance, pred_binary, kept_segments, num_raw


def _pred_padded_to_original_canvas(pred_1024, crop_info, pad_info):
    """Quita padding 1024, escala al recorte, y pega en el lienzo original completo."""
    pad_y = int(pad_info["pad_y"])
    pad_x = int(pad_info["pad_x"])
    new_h = int(pad_info["new_h"])
    new_w = int(pad_info["new_w"])
    ow = int(pad_info["orig_w"])
    oh = int(pad_info["orig_h"])

    sl = pred_1024[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
    pred_crop = cv2.resize(
        sl.astype(np.float32),
        (ow, oh),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)

    if crop_info is None:
        return pred_crop

    full_h = int(crop_info["orig_h"])
    full_w = int(crop_info["orig_w"])
    x1 = int(crop_info["x1"])
    y1 = int(crop_info["y1"])
    x2 = int(crop_info["x2"])
    y2 = int(crop_info["y2"])

    pred_full = np.zeros((full_h, full_w), dtype=np.int32)
    pred_full[y1:y2, x1:x2] = pred_crop
    return pred_full


@torch.no_grad()
def run_inference(
    image_path,
    model_path,
    contour_path=None,
    output_dir="inference_outputs_mask2former_1024",
    crop_margin=20,
    use_eyelid_crop=True,
    save_outputs=False,
):
    """
    Misma forma de retorno que panoptic6 / infer_server (overlay sobre imagen original).
    `contour_path` = máscara de párpado (opcional).
    """
    device = _get_device()

    img_tensor, img_padded, crop_info, pad_info = preprocess_image(
        image_path=image_path,
        eyelid_path=contour_path,
        crop_to_eyelid=use_eyelid_crop,
        crop_margin=crop_margin,
    )

    model = load_trained_model(model_path, device=device)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    pred_1024, pred_binary_1024, _kept, num_raw = predict_instances(
        model,
        processor,
        img_tensor,
        device=device,
    )

    pred_instance_original = _pred_padded_to_original_canvas(
        pred_1024,
        crop_info,
        pad_info,
    )
    pred_binary_original = (pred_instance_original > 0).astype(np.uint8)

    img_orig = np.array(Image.open(image_path).convert("RGB")).astype(np.uint8)

    overlay_rgb = label2rgb(
        pred_instance_original,
        image=img_orig.astype(np.float64) / 255.0,
        bg_label=0,
        alpha=0.45,
    )
    overlay_u8 = (np.clip(overlay_rgb, 0, 1) * 255).astype(np.uint8)

    pred_count_512 = int(pred_1024.max())
    pred_count_original = int(pred_instance_original.max())

    result = {
        "img_original": img_orig,
        "img_model_input": img_padded.astype(np.uint8),
        "pred_binary": pred_binary_original,
        "pred_instance": pred_instance_original,
        "overlay_rgb": overlay_u8,
        "num_raw": int(num_raw),
        "pred_count_512": pred_count_512,
        "pred_count_original": pred_count_original,
        "device": str(device),
        "image_path": image_path,
        "model_path": model_path,
    }

    if not save_outputs:
        return result

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    save_instance_mask(
        pred_1024,
        os.path.join(output_dir, f"{base}_instances_uint16.png"),
    )
    save_binary_mask(
        pred_binary_original,
        os.path.join(output_dir, f"{base}_binary.png"),
    )
    save_overlay(
        img_padded,
        pred_1024,
        os.path.join(output_dir, f"{base}_overlay_1024.png"),
    )
    Image.fromarray(overlay_u8).save(
        os.path.join(output_dir, f"{base}_overlay_full.png")
    )
    return result


# ============================================================
# GUARDADO Y VISUALIZACIÓN
# ============================================================

def save_instance_mask(pred_instance, output_path):
    """
    Guarda la máscara real de instancias en uint16.
    Esta imagen sirve para análisis, no necesariamente para verla directamente.
    Los IDs son:
      0 = fondo
      1 = instancia 1
      2 = instancia 2
      ...
    """
    pred_uint16 = pred_instance.astype(np.uint16)
    Image.fromarray(pred_uint16).save(output_path)

    print(f"\nMáscara técnica de instancias guardada en: {output_path}")
    print(f"  dtype guardado: uint16")
    print(f"  valores únicos: {np.unique(pred_instance)[:100]}")
    print(f"  número de instancias: {int(pred_instance.max())}")


def save_instance_mask_visual(pred_instance, output_path):
    """
    Guarda una versión visible de la máscara de instancias.
    Escala los IDs a 0-255 para que no se vea negra.
    """
    max_id = int(pred_instance.max())

    if max_id == 0:
        visual = np.zeros_like(pred_instance, dtype=np.uint8)
    else:
        visual = (
            pred_instance.astype(np.float32) / max_id * 255.0
        ).astype(np.uint8)

    Image.fromarray(visual).save(output_path)

    print(f"Máscara visual escalada guardada en: {output_path}")


def save_colored_instance_mask(pred_instance, output_path):
    """
    Guarda una imagen RGB coloreada por instancia.
    Es la mejor para inspección visual humana.
    """
    colored = label2rgb(
        pred_instance,
        bg_label=0,
    )

    colored_u8 = np.clip(colored * 255, 0, 255).astype(np.uint8)

    Image.fromarray(colored_u8).save(output_path)

    print(f"Máscara coloreada de instancias guardada en: {output_path}")


def save_binary_mask(pred_binary, output_path):
    pred_u8 = (pred_binary * 255).astype(np.uint8)
    Image.fromarray(pred_u8).save(output_path)

    print(f"Máscara binaria guardada en: {output_path}")


def save_overlay(img_np, pred_instance, output_path):
    overlay = label2rgb(
        pred_instance,
        image=img_np / 255.0,
        bg_label=0,
        alpha=0.45,
    )

    overlay_u8 = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    Image.fromarray(overlay_u8).save(output_path)

    print(f"Overlay guardado en: {output_path}")


def save_visualization(img_np, pred_instance, pred_binary, kept_segments, output_path):
    colored_pred = label2rgb(
        pred_instance,
        image=None,
        bg_label=0,
    )

    overlay = label2rgb(
        pred_instance,
        image=img_np / 255.0,
        bg_label=0,
        alpha=0.45,
    )

    count = int(pred_instance.max())

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(img_np)
    axes[0].set_title("Imagen preprocesada 1024x1024")
    axes[0].axis("off")

    axes[1].imshow(pred_binary, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Predicción binaria")
    axes[1].axis("off")

    axes[2].imshow(colored_pred)
    axes[2].set_title(f"Instancias coloreadas: {count}")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualización comparativa guardada en: {output_path}")

    if kept_segments:
        print("\nSegmentos conservados:")
        for s in kept_segments:
            print(
                f"  instancia={s['instance_id']:02d} | "
                f"score={s['score']:.4f} | "
                f"area={s['area']}"
            )
    else:
        print("\nNo se conservaron segmentos después del filtrado.")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("============================================================")
    print("INFERENCIA MASK2FORMER - GLÁNDULAS DE MEIBOMIO")
    print("============================================================")
    print(f"Dispositivo: {DEVICE}")
    print(f"Imagen: {IMAGE_PATH}")
    print(f"Máscara párpado: {EYELID_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Salida: {OUTPUT_DIR}")
    print(f"USE_EYELID_CROP: {USE_EYELID_CROP}")
    print(f"OUTPUT_SIZE: {OUTPUT_SIZE}")
    print(f"INSTANCE_SCORE_THRESHOLD: {INSTANCE_SCORE_THRESHOLD}")
    print(f"INSTANCE_MASK_THRESHOLD: {INSTANCE_MASK_THRESHOLD}")
    print(f"MIN_INSTANCE_AREA: {MIN_INSTANCE_AREA}")
    print("============================================================")

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"No existe IMAGE_PATH: {IMAGE_PATH}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"No existe CHECKPOINT_PATH: {CHECKPOINT_PATH}")

    img_tensor, img_np, crop_info, pad_info = preprocess_image(
        image_path=IMAGE_PATH,
        eyelid_path=EYELID_PATH,
        crop_to_eyelid=USE_EYELID_CROP,
        crop_margin=20,
    )

    print("\nInformación de preprocesamiento:")
    print(f"  img_tensor shape: {tuple(img_tensor.shape)}")
    print(f"  img_np shape: {img_np.shape}")
    print(f"  crop_info: {crop_info}")
    print(f"  pad_info: {pad_info}")

    model = load_trained_model(CHECKPOINT_PATH)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    pred_instance, pred_binary, kept_segments, _num_raw = predict_instances(
        model,
        processor,
        img_tensor,
    )

    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    technical_inst_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_instances_uint16.png",
    )

    visual_inst_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_instances_visual.png",
    )

    colored_inst_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_instances_colored.png",
    )

    binary_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_binary.png",
    )

    overlay_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_overlay.png",
    )

    visualization_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_visualization.png",
    )

    save_instance_mask(pred_instance, technical_inst_path)
    save_instance_mask_visual(pred_instance, visual_inst_path)
    save_colored_instance_mask(pred_instance, colored_inst_path)
    save_binary_mask(pred_binary, binary_path)
    save_overlay(img_np, pred_instance, overlay_path)
    save_visualization(
        img_np,
        pred_instance,
        pred_binary,
        kept_segments,
        visualization_path,
    )

    print("\n============================================================")
    print("ARCHIVOS GENERADOS")
    print("============================================================")
    print(f"Máscara técnica uint16: {technical_inst_path}")
    print(f"Máscara visual 0-255:   {visual_inst_path}")
    print(f"Máscara coloreada:      {colored_inst_path}")
    print(f"Máscara binaria:        {binary_path}")
    print(f"Overlay:                {overlay_path}")
    print(f"Visualización general:  {visualization_path}")
    print("============================================================")


def load_model_server(checkpoint_path, device=None):
    """Pre-load model + processor for server use (call once at startup)."""
    if device is None:
        device = _get_device()
    import os
    model = load_trained_model(checkpoint_path, device=device)
    _proc_src = _LOCAL_CFG if os.path.isdir(_LOCAL_CFG) else MODEL_ID
    processor = AutoImageProcessor.from_pretrained(
        _proc_src, use_fast=False, local_files_only=os.path.isdir(_LOCAL_CFG)
    )
    return model, processor


def infer_with_model(image_path, model, processor, device=None, contour_path=None):
    """Run inference with pre-loaded model; returns pred_instance at original image coords."""
    if device is None:
        device = _get_device()

    img_tensor, _, crop_info, pad_info = preprocess_image(
        image_path=image_path,
        eyelid_path=contour_path,
        crop_to_eyelid=contour_path is not None,
        crop_margin=20,
    )

    pred_1024, _, _, _ = predict_instances(model, processor, img_tensor, device=device)
    return _pred_padded_to_original_canvas(pred_1024, crop_info, pad_info)


if __name__ == "__main__":
    main()