#!/usr/bin/env python3
# inference.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
from transformers import (
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
)
from skimage.color import label2rgb

# ============================================================
# CONFIGURACIÓN (Debe coincidir con el entrenamiento)
# ============================================================
MODEL_ID = "facebook/mask2former-swin-small-cityscapes-instance"
DEFAULT_MODEL_WEIGHTS = "best_model (32).pth"

ID2LABEL = {0: "background", 1: "gland"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

NUM_QUERIES = 100
OUTPUT_SIZE = (512, 512)

INSTANCE_SCORE_THRESHOLD = 0.75
INSTANCE_MASK_THRESHOLD = 0.60
MIN_INSTANCE_AREA = 50
CROP_MARGIN = 20

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def load_model(model_path, device):
    print("Cargando modelo y procesador...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MODEL_ID,
        num_labels=2,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.config.is_thing_map = [0, 1]
    model.config.num_queries = NUM_QUERIES
    
    # Cargar los pesos entrenados
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {model_path}")
        
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
        
    # Eliminar el prefijo 'module.' si el modelo fue guardado con DDP
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    
    return model, processor

def preprocess_image(image_path, contour_path=None, crop_margin=CROP_MARGIN):
    """
    Carga la imagen, aplica el recorte por el párpado (si se provee),
    y la redimensiona/normaliza.
    Devuelve:
      - img_tensor (para el modelo)
      - img_cropped (resolución original del recorte)
      - crop_info dict (para rearmar al tamaño original)
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    orig_h, orig_w = img_np.shape[:2]
    crop_info = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "x1": 0,
        "y1": 0,
        "x2": orig_w,
        "y2": orig_h,
    }
    
    # Aplicar recorte del párpado (altamente recomendado para la precisión)
    if contour_path and os.path.exists(contour_path):
        eyelid_mask = Image.open(contour_path).convert("L")
        eyelid_mask_np = (np.array(eyelid_mask) > 0).astype(np.uint8)
        
        # Enmascarar el fondo
        img_np = img_np * eyelid_mask_np[..., None]
        
        # Recortar (Crop)
        ys, xs = np.where(eyelid_mask_np > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1 = max(0, int(xs.min()) - int(crop_margin))
            x2 = min(img_np.shape[1], int(xs.max()) + 1 + int(crop_margin))
            y1 = max(0, int(ys.min()) - int(crop_margin))
            y2 = min(img_np.shape[0], int(ys.max()) + 1 + int(crop_margin))
            img_np = img_np[y1:y2, x1:x2]
            crop_info.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    # Guardar la imagen recortada a tamaño real (para el overlay final)
    img_cropped = img_np.copy()

    # Redimensionar al tamaño esperado por el modelo
    img_resized = cv2.resize(img_np, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    img_pil_resized = Image.fromarray(img_resized)
    
    # Transformaciones de Torchvision
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    img_tensor = transform(img_pil_resized).unsqueeze(0)
    
    return img_tensor, img_cropped, crop_info

def parse_instance_output(inst, H, W, score_threshold):
    """Convierte la salida del modelo en un mapa de instancias 2D"""
    pred_instance = np.zeros((H, W), dtype=np.int32)
    
    seg = inst["segmentation"].cpu().numpy()
    scores = inst.get("scores", None)
    labels = inst.get("labels", None)
    segments_info = inst.get("segments_info", None)

    if scores is not None: scores = scores.cpu().numpy()
    if labels is not None: labels = labels.cpu().numpy()

    if seg.ndim == 2:
        unique_ids = np.unique(seg)
        unique_ids = unique_ids[unique_ids != 0]
        k = 1

        if segments_info is not None:
            for seginfo in segments_info:
                sid = seginfo["id"]
                lab = seginfo.get("label_id", 1)
                score = seginfo.get("score", 1.0)

                if lab == 0 or score < score_threshold:
                    continue

                region = (seg == sid)
                if region.sum() < MIN_INSTANCE_AREA:
                    continue

                pred_instance[region] = k
                k += 1
    return pred_instance

# ============================================================
# FUNCIÓN PRINCIPAL DE INFERENCIA
# ============================================================

def _instance_pred_to_original(pred_instance_512, img_cropped, crop_info):
    """Resize 512→crop and paste back into original canvas."""
    crop_h, crop_w = img_cropped.shape[:2]
    pred_crop = cv2.resize(
        pred_instance_512.astype(np.float32),
        (crop_w, crop_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)

    orig_h = int(crop_info["orig_h"])
    orig_w = int(crop_info["orig_w"])
    x1, y1, x2, y2 = int(crop_info["x1"]), int(crop_info["y1"]), int(crop_info["x2"]), int(crop_info["y2"])

    pred_orig = np.zeros((orig_h, orig_w), dtype=np.int32)
    pred_orig[y1:y2, x1:x2] = pred_crop
    return pred_crop, pred_orig


@torch.no_grad()
def run_inference(
    image_path,
    model_path,
    contour_path=None,
    output_dir="inference_outputs_panoptic6",
    crop_margin=CROP_MARGIN,
    save_outputs=False,
):
    """
    Inferencia compatible con infer_server:
      - Devuelve pred_binary, pred_instance, overlay_rgb, conteos y device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(model_path=model_path, device=device)

    img_tensor, img_cropped, crop_info = preprocess_image(
        image_path=image_path,
        contour_path=contour_path,
        crop_margin=crop_margin,
    )
    img_tensor = img_tensor.to(device)

    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        outputs = model(pixel_values=img_tensor)

    H, W = OUTPUT_SIZE

    try:
        inst = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(H, W)],
            threshold=INSTANCE_SCORE_THRESHOLD,
            mask_threshold=INSTANCE_MASK_THRESHOLD,
        )[0]
        pred_instance_512 = parse_instance_output(inst, H, W, INSTANCE_SCORE_THRESHOLD)
    except Exception as e:
        print(f"Error procesando instancias: {e}")
        pred_instance_512 = np.zeros((H, W), dtype=np.int32)

    pred_crop, pred_instance_original = _instance_pred_to_original(
        pred_instance_512=pred_instance_512,
        img_cropped=img_cropped,
        crop_info=crop_info,
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

    result = {
        "img_original": img_orig,
        "img_model_input": img_cropped,
        "pred_binary": pred_binary_original,
        "pred_instance": pred_instance_original,
        "overlay_rgb": overlay_u8,
        "num_raw": int(pred_instance_512.max()),
        "pred_count_512": int(pred_instance_512.max()),
        "pred_count_original": int(pred_instance_original.max()),
        "device": str(device),
        "image_path": image_path,
        "model_path": model_path,
        "crop_info": crop_info,
    }

    if not save_outputs:
        return result

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base}_pred_instance.png"), pred_instance_original.astype(np.uint16))
    cv2.imwrite(os.path.join(output_dir, f"{base}_pred_binary.png"), (pred_binary_original * 255).astype(np.uint8))
    Image.fromarray(overlay_u8).save(os.path.join(output_dir, f"{base}_overlay.png"))
    return result

# ============================================================
# SERVER INFERENCE (pre-loaded model)
# ============================================================

_LOCAL_CONFIG = "/app/mask2former_config"

def load_model_server(model_path, device):
    """
    Loads model + processor for the FastAPI server.
    Uses local config dir if available (Cloud Run), otherwise downloads from HF.
    Handles DDP prefix and weights_only fallback.
    """
    from transformers import Mask2FormerConfig
    _local = os.path.isdir(_LOCAL_CONFIG)
    _src = _LOCAL_CONFIG if _local else MODEL_ID

    processor = AutoImageProcessor.from_pretrained(_src, local_files_only=_local)

    config = Mask2FormerConfig.from_pretrained(_src, local_files_only=_local)
    config.num_labels = 2
    config.num_queries = NUM_QUERIES
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    config.is_thing_map = {0: False, 1: True}

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        _src,
        config=config,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
        local_files_only=_local,
    )
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.config.is_thing_map = {0: False, 1: True}
    model.config.num_queries = NUM_QUERIES

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        state_dict = torch.load(model_path, map_location=device)

    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    return model, processor


@torch.no_grad()
def infer_with_model(image_path, model, processor, device, contour_path=None):
    """
    Pre-loaded model inference for the FastAPI server.
    Returns pred_instance (HxW int32) in original image coordinates.
    Uses processor bilinear resize at crop dimensions (avoids INTER_NEAREST artifacts).
    """
    img_tensor, img_cropped, crop_info = preprocess_image(
        image_path=image_path,
        contour_path=contour_path,
    )
    img_tensor = img_tensor.to(device)

    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        outputs = model(pixel_values=img_tensor)

    crop_h, crop_w = img_cropped.shape[:2]

    try:
        inst = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(crop_h, crop_w)],
            threshold=INSTANCE_SCORE_THRESHOLD,
            mask_threshold=INSTANCE_MASK_THRESHOLD,
        )[0]
        pred_instance_crop = parse_instance_output(inst, crop_h, crop_w, INSTANCE_SCORE_THRESHOLD)
    except Exception as e:
        print(f"[panoptic6] Error post-processing: {e}")
        pred_instance_crop = np.zeros((crop_h, crop_w), dtype=np.int32)

    orig_h = int(crop_info["orig_h"])
    orig_w = int(crop_info["orig_w"])
    x1, y1 = int(crop_info["x1"]), int(crop_info["y1"])

    pred_instance = np.zeros((orig_h, orig_w), dtype=np.int32)
    pred_instance[y1:y1+crop_h, x1:x1+crop_w] = pred_instance_crop

    return pred_instance


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    # --- CAMBIA ESTAS RUTAS POR TUS IMÁGENES DE PRUEBA ---
    TEST_IMAGE_PATH = "meibomio2.jpg" 
    
    # Opcional, pero recomendado para aislar el área del ojo
    TEST_CONTOUR_PATH = None
    
    # Si no tienes la máscara del eyelid a mano, simplemente pon: TEST_CONTOUR_PATH = None

    if os.path.exists(TEST_IMAGE_PATH):
        run_inference(
            image_path=TEST_IMAGE_PATH,
            model_path=DEFAULT_MODEL_WEIGHTS,
            contour_path=TEST_CONTOUR_PATH,
            output_dir="inference_outputs_panoptic6",
            save_outputs=True,
        )
    else:
        print(f"Por favor, actualiza TEST_IMAGE_PATH. No se encontró: {TEST_IMAGE_PATH}")