#!/usr/bin/env python3
# inference_tarsus.py — Inferencia de máscara del tarso para una nueva imagen

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms, models

# ============================================================
# CONFIGURACIÓN — ajusta estas rutas
# ============================================================

MODEL_WEIGHTS = "best_model_tarsus_improved.pth"   # Ruta al .pth entrenado con ddp_segmentation_tarsus.py
TEST_IMAGE_PATH = "meibomio2.jpg"              # Imagen de test
OUTPUT_PATH = "resultado_tarsus.png"       # Imagen de salida

INPUT_SIZE = (512, 512)   # Debe coincidir con el entrenamiento
THRESHOLD = 0.5           # Umbral para binarizar la predicción
USE_CLAHE = True          # Igual que en entrenamiento
USE_TTA = True            # Test-Time Augmentation (flip horizontal)

# ============================================================
# MODELO (copia de ddp_segmentation_tarsus.py)
# ============================================================

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.act(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetWithPretrainedEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        encoder = models.resnet34(weights=None)
        layers = list(encoder.children())
        self.layer0 = nn.Sequential(*layers[:3])
        self.layer1 = nn.Sequential(*layers[3:5])
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = self._double_conv(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = self._double_conv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = self._double_conv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = self._double_conv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        if self.use_attention:
            self.att3 = AttentionGate(256, 256, 128)
            self.att2 = AttentionGate(128, 128, 64)
            self.att1 = AttentionGate(64, 64, 32)
            self.att0 = AttentionGate(64, 64, 32)

    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        d4 = self.upconv4(x4)
        skip3 = self.att3(d4, x3) if self.use_attention else x3
        d4 = self.conv4(torch.cat((d4, skip3), dim=1))
        d3 = self.upconv3(d4)
        skip2 = self.att2(d3, x2) if self.use_attention else x2
        d3 = self.conv3(torch.cat((d3, skip2), dim=1))
        d2 = self.upconv2(d3)
        skip1 = self.att1(d2, x1) if self.use_attention else x1
        d2 = self.conv2(torch.cat((d2, skip1), dim=1))
        d1 = self.upconv1(d2)
        skip0 = self.att0(d1, x0) if self.use_attention else x0
        d1 = self.conv1(torch.cat((d1, skip0), dim=1))
        out = self.final_conv(d1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def preprocess(image_path, input_size=INPUT_SIZE, use_clahe=USE_CLAHE):
    img_pil = Image.open(image_path).convert("L")
    img_np = np.array(img_pil)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = clahe.apply(img_np)

    img_rgb = np.stack([img_np] * 3, axis=-1)
    img_rgb = cv2.resize(img_rgb, (input_size[1], input_size[0]), interpolation=cv2.INTER_AREA)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0)
    return tensor, img_rgb


# ============================================================
# INFERENCIA
# ============================================================

def infer(image_path, output_path=OUTPUT_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando: {device}")

    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_WEIGHTS}")

    model = UNetWithPretrainedEncoder(in_channels=3, out_channels=1, use_attention=True).to(device)
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
    # Soporta tanto checkpoint completo {"model": ..., "optimizer": ...} como state_dict directo
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    # Eliminar prefijo 'module.' si el modelo fue guardado con DDP
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Faltan pesos en el checkpoint: {missing[:5]}...")
    if unexpected:
        print(f"Claves ignoradas del checkpoint (redundantes): {len(unexpected)}")
    model.eval()
    print("Modelo cargado.")

    img_tensor, img_rgb = preprocess(image_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        if USE_TTA:
            img_flip = torch.flip(img_tensor, dims=[3])
            output_flip = model(img_flip)
            output_flip = torch.flip(output_flip, dims=[3])
            output = (output + output_flip) / 2.0

    pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_binary = (pred_prob > THRESHOLD).astype(np.uint8)

    # Overlay: contorno de la máscara sobre la imagen original
    img_norm = img_rgb.astype(np.float32) / 255.0
    contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img_norm.copy()
    cv2.drawContours(overlay, contours, -1, (0, 1, 0), 2)  # contorno verde

    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_norm)
    axes[0].set_title("Imagen de entrada")
    axes[0].axis("off")

    axes[1].imshow(pred_binary, cmap="gray")
    axes[1].set_title(f"Máscara predicha (thr={THRESHOLD})")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Contorno del tarso sobre imagen")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Resultado guardado en: {output_path}")
    plt.show()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"No se encontró la imagen: {TEST_IMAGE_PATH}")
        print("Actualiza TEST_IMAGE_PATH al inicio del script.")
    else:
        infer(TEST_IMAGE_PATH, OUTPUT_PATH)
