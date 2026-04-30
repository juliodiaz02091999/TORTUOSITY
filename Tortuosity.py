import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import random
import cv2  # Needed for contour operations in tortuosity calculation
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from Pipeline import SpatialCalibrator, GlandMorphometry, TortuosityAnalyzer

# Definir dispositivo (GPU o CPU)
device = torch.device("cpu")  # Force CPU for Cloud Run compatibility

# ------------------------------------------------------
# Función para cargar el modelo Mask R-CNN y sus pesos
# ------------------------------------------------------
def load_maskrcnn_model(model_path="final_model.pth", device=device):
    try:
        # Use modern PyTorch syntax without pretrained parameter
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        num_classes = 2  # fondo y glándula
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        # Add better error handling for model loading
        print(f"Loading model from: {model_path}")
        
        # Try different loading strategies for compatibility
        try:
            # First try: standard loading
            state_dict = torch.load(model_path, map_location=device)
        except Exception as e1:
            print(f"Standard loading failed: {e1}")
            try:
                # Second try: with pickle protocol 5 (for newer PyTorch versions)
                state_dict = torch.load(model_path, map_location=device, pickle_module=torch._utils._rebuild_tensor_v2)
            except Exception as e2:
                print(f"Pickle protocol 5 loading failed: {e2}")
                try:
                    # Third try: with weights_only (for newer PyTorch versions)
                    state_dict = torch.load(model_path, map_location=device, weights_only=True)
                except Exception as e3:
                    print(f"Weights-only loading failed: {e3}")
                    try:
                        # Fourth try: with different pickle protocol
                        import pickle
                        state_dict = torch.load(model_path, map_location=device, pickle_module=pickle)
                    except Exception as e4:
                        print(f"Pickle loading failed: {e4}")
                        # Last resort: try with different map_location
                        state_dict = torch.load(model_path, map_location='cpu')
        
        try:
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print("Mask R-CNN model loaded successfully")
            return model
        except Exception as e:
            print(f"Error setting model state: {e}")
            raise e
    except Exception as e:
        print(f"Error loading Mask R-CNN model: {e}")
        raise e

# ------------------------------------------------------
# Función para redimensionar a tamaño objetivo
# ------------------------------------------------------
def resize_to_target_size(img, target_size=(512, 512)):
    """
    Redimensiona la imagen al tamaño objetivo usado en entrenamiento.
    """
    return img.resize((target_size[1], target_size[0]), Image.BILINEAR)

# ------------------------------------------------------
# Función para aplicar CLAHE preprocessing
# ------------------------------------------------------
def apply_clahe_preprocessing(img_pil):
    """
    Aplica CLAHE igual que en entrenamiento
    """
    # Convertir a escala de grises
    img_gray = img_pil.convert("L")
    img_np_gray = np.array(img_gray)

    # Aplicar CLAHE con parámetros similares al entrenamiento
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_np_gray)

    # Replicar a 3 canales como en entrenamiento
    img_np_rgb = np.stack([img_eq]*3, axis=-1)
    return Image.fromarray(img_np_rgb, mode='RGB')

# ------------------------------------------------------
# Definición de Attention Gate (igual que en script.py)
# ------------------------------------------------------
class AttentionGate(torch.nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = torch.nn.Sequential(
            torch.nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )
        self.W_x = torch.nn.Sequential(
            torch.nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )
        self.psi = torch.nn.Sequential(
            torch.nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.Sigmoid()
        )
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.act(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ------------------------------------------------------
# Definición del modelo UNet con encoder preentrenado y Attention Gates
# ------------------------------------------------------
class UNetWithPretrainedEncoder(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, use_attention=True):
        super(UNetWithPretrainedEncoder, self).__init__()
        self.use_attention = use_attention
        # Usar los mismos pesos preentrenados que en entrenamiento (script.py)
        try:
            self.encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        except AttributeError:
            # Compatibilidad con versiones anteriores de torchvision
            self.encoder = models.resnet34(pretrained=True)
        self.encoder_layers = list(self.encoder.children())
        self.layer0 = torch.nn.Sequential(*self.encoder_layers[:3])   # Conv1, BN1, ReLU
        self.layer1 = torch.nn.Sequential(*self.encoder_layers[3:5])    # MaxPool + Layer1
        self.layer2 = self.encoder_layers[5]                      # Layer2
        self.layer3 = self.encoder_layers[6]                      # Layer3
        self.layer4 = self.encoder_layers[7]                      # Layer4

        # Decoder
        self.upconv4 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = self._double_conv(512, 256)
        self.upconv3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = self._double_conv(256, 128)
        self.upconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = self._double_conv(128, 64)
        self.upconv1 = torch.nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = self._double_conv(128, 64)
        self.final_conv = torch.nn.Conv2d(64, out_channels, kernel_size=1)

        # Attention Gates (igual que en script.py)
        if self.use_attention:
            self.att3 = AttentionGate(256, 256, 128)
            self.att2 = AttentionGate(128, 128, 64)
            self.att1 = AttentionGate(64, 64, 32)
            self.att0 = AttentionGate(64, 64, 32)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.upconv4(x4)
        skip3 = self.att3(d4, x3) if self.use_attention else x3
        d4 = torch.cat((d4, skip3), dim=1)
        d4 = self.conv4(d4)

        d3 = self.upconv3(d4)
        skip2 = self.att2(d3, x2) if self.use_attention else x2
        d3 = torch.cat((d3, skip2), dim=1)
        d3 = self.conv3(d3)

        d2 = self.upconv2(d3)
        skip1 = self.att1(d2, x1) if self.use_attention else x1
        d2 = torch.cat((d2, skip1), dim=1)
        d2 = self.conv2(d2)

        d1 = self.upconv1(d2)
        skip0 = self.att0(d1, x0) if self.use_attention else x0
        d1 = torch.cat((d1, skip0), dim=1)
        d1 = self.conv1(d1)

        out = self.final_conv(d1)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

    def _double_conv(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

# ------------------------------------------------------
# Pipeline de transformaciones: igual que en entrenamiento
# ------------------------------------------------------
transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_to_target_size(img, (512, 512))),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ------------------------------------------------------
# Funciones para cargar y predecir con UNet mejorado
# ------------------------------------------------------
def load_unet_model(model_path, device):
    try:
        model = UNetWithPretrainedEncoder(in_channels=3, out_channels=1, use_attention=True)
        print(f"Loading UNet model from: {model_path}")

        # Try different loading strategies for compatibility
        try:
            # First try: standard loading
            state_dict = torch.load(model_path, map_location=device)
        except Exception as e1:
            print(f"Standard loading failed: {e1}")
            try:
                # Second try: with pickle protocol 5 (for newer PyTorch versions)
                state_dict = torch.load(model_path, map_location=device, pickle_module=torch._utils._rebuild_tensor_v2)
            except Exception as e2:
                print(f"Pickle protocol 5 loading failed: {e2}")
                try:
                    # Third try: with weights_only (for newer PyTorch versions)
                    state_dict = torch.load(model_path, map_location=device, weights_only=True)
                except Exception as e3:
                    print(f"Weights-only loading failed: {e3}")
                    try:
                        # Fourth try: with different pickle protocol
                        import pickle
                        state_dict = torch.load(model_path, map_location=device, pickle_module=pickle)
                    except Exception as e4:
                        print(f"Pickle loading failed: {e4}")
                        # Last resort: try with different map_location
                        state_dict = torch.load(model_path, map_location='cpu')

        try:
            # Aceptar formatos: {'model': state_dict, ...} o state_dict plano
            if isinstance(state_dict, dict) and 'model' in state_dict:
                model_state_dict = state_dict['model']
            else:
                model_state_dict = state_dict
            missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
            if missing:
                print(f"[WARN] Missing keys: {len(missing)} (primeros 5) {missing[:5]}")
            if unexpected:
                print(f"[WARN] Unexpected keys: {len(unexpected)} (primeros 5) {unexpected[:5]}")
            model.to(device)
            model.eval()
            print("UNet model loaded successfully")
            return model
        except Exception as e:
            print(f"Error setting model state: {e}")
            raise e
    except Exception as e:
        print(f"Error loading UNet model: {e}")
        raise e

def predict_unet_model(model, image_path, device, use_clahe=True, use_tta=True):
    """
    Predice la máscara del tarso usando el modelo mejorado con CLAHE y TTA
    """
    # Cargar imagen original
    original_image = Image.open(image_path).convert("RGB")

    # Aplicar CLAHE preprocessing igual que en entrenamiento
    if use_clahe:
        processed_image = apply_clahe_preprocessing(original_image)
    else:
        processed_image = original_image

    # Aplicar transformaciones
    input_tensor = transform(processed_image).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_tta:
            # Test-Time Augmentation igual que en script.py
            output = model(input_tensor)
            # Flip horizontal
            img_flip = torch.flip(input_tensor, dims=[3])
            output_flip = model(img_flip)
            output_flip = torch.flip(output_flip, dims=[3])
            # Promedio
            output = (output + output_flip) / 2.0
        else:
            output = model(input_tensor)

        output = torch.sigmoid(output)  # Convertir logits a probabilidades
        prediction = (output > 0.5).float()  # Umbralización

    return original_image, prediction.cpu().squeeze(0)

def predict_unet_model_legacy(model, image_path, device):
    """
    Función legacy para compatibilidad con el código existente
    """
    image = Image.open(image_path).convert("RGB")
    transform_legacy = transforms.Compose([
        transforms.Lambda(resize_to_previous_multiple_of_32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform_legacy(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        prediction = (output > 0.5).float()  # Umbralización para obtener máscara binaria
    return prediction.cpu().squeeze(0)

# ------------------------------------------------------
# Función de predicción de instancias con Mask-RCNN
# ------------------------------------------------------
def predict_maskrcnn_model(model, image_path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    masks_np = prediction["masks"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    selected_masks = (masks_np[scores > 0.5] > 0.5).astype(np.uint8)
    if selected_masks.size == 0:
        return np.zeros(image_tensor.shape[2:], dtype=np.int32)
    pred_instance = np.zeros(selected_masks.shape[2:], dtype=np.int32)
    for i, mask in enumerate(selected_masks, start=1):
        mask_squeezed = mask.squeeze(0)
        pred_instance[mask_squeezed > 0.5] = i
    return pred_instance

# ------------------------------------------------------
# Función para generar un color aleatorio (RGB)
# ------------------------------------------------------
def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]

# ------------------------------------------------------
# Helpers para el cálculo robusto de tortuosidad
# ------------------------------------------------------
def _skeleton_to_path(skel):
    """
    Recorre el esqueleto binario y devuelve un array ordenado de puntos (y, x)
    representando la trayectoria más larga encontrada mediante traversal greedy.
    """
    ys, xs = np.where(skel)
    if len(xs) < 2:
        return None

    pts = list(zip(ys.tolist(), xs.tolist()))
    coord_set = set(pts)

    def get_nbrs(y, x):
        return [(y + dy, x + dx)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and (y + dy, x + dx) in coord_set]

    # Nodos de grado 1 = extremos del esqueleto
    endpoints = [p for p in pts if len(get_nbrs(*p)) == 1]
    start = endpoints[0] if endpoints else pts[0]

    path = [start]
    visited = {start}
    cur = start

    while True:
        nbrs = [p for p in get_nbrs(*cur) if p not in visited]
        if not nbrs:
            break
        # En bifurcaciones, elegir el vecino con más continuación (heurística greedy)
        cur = nbrs[0] if len(nbrs) == 1 else max(
            nbrs,
            key=lambda p: len([q for q in get_nbrs(*p) if q not in visited])
        )
        path.append(cur)
        visited.add(cur)

    return np.array(path)


def _smooth_path(path, window_length=11, polyorder=2):
    """
    Suaviza las coordenadas de una trayectoria con filtro Savitzky-Golay
    para eliminar el zig-zag digital del esqueleto binario.
    """
    n = len(path)
    wl = min(window_length, n)
    if wl % 2 == 0:
        wl -= 1
    if wl < polyorder + 2:
        wl = polyorder + 2
        if wl % 2 == 0:
            wl += 1
    if wl > n or n < 4:
        return path.astype(float)
    ys = savgol_filter(path[:, 0].astype(float), wl, polyorder)
    xs = savgol_filter(path[:, 1].astype(float), wl, polyorder)
    return np.column_stack([ys, xs])


def _path_length(path):
    """Longitud curvilínea real de una trayectoria (suma de distancias euclidianas)."""
    diffs = np.diff(path, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


# Longitud mínima de esqueleto para considerar una glándula válida
_MIN_SKEL_PX = 10

# ------------------------------------------------------
# Función para calcular la tortuosidad de una glándula de Meibomio
# ------------------------------------------------------
def calculate_gland_tortuosity(mask):
    """
    Calcula la tortuosidad, longitud y grosor de una glándula de Meibomio.

    Tortuosidad = (longitud_curvilínea_suavizada / distancia_euclidiana) - 1
                  0 = línea recta perfecta; valores mayores = más tortuosa.
                  El esqueleto se suaviza con Savitzky-Golay para eliminar
                  el zig-zag digital antes de medir.
    Longitud    = longitud curvilínea del esqueleto suavizado (px)
    Grosor      = 2 × media de la transformada de distancia sobre el esqueleto (px)

    Returns:
        dict con 'tortuosity', 'length_px', 'thickness_px',
        o None si la glándula es demasiado pequeña o fragmentada.
    """
    binary = mask.astype(np.uint8)

    skel = skeletonize(binary > 0)
    if int(np.sum(skel)) < _MIN_SKEL_PX:
        return None

    # --- Grosor: EDT sobre los píxeles del esqueleto (sin cambios) ---
    dist = distance_transform_edt(binary > 0)
    skel_pixels = dist[skel]
    thickness_px = float(2.0 * np.mean(skel_pixels)) if skel_pixels.size > 0 else 0.0

    # --- Longitud y Tortuosidad: basadas en esqueleto suavizado ---
    path = _skeleton_to_path(skel)
    if path is None or len(path) < 3:
        return None

    smooth = _smooth_path(path, window_length=11, polyorder=2)
    length_px = _path_length(smooth)

    euclidean_dist = float(np.linalg.norm(smooth[-1] - smooth[0]))
    if euclidean_dist > 1.0:
        tortuosity = max(0.0, (length_px / euclidean_dist) - 1.0)
    else:
        tortuosity = 0.0

    return {'tortuosity': tortuosity, 'length_px': length_px, 'thickness_px': thickness_px}

# ------------------------------------------------------
# Función de visualización y combinación final
# ------------------------------------------------------
def compute_results_from_instance_map(pred_instance, image_path, unet_model, device, um_per_px: float = 1.0):
    """
    Runs the full tortuosity pipeline given a pre-computed instance map.
    pred_instance: HxW int32 numpy array (0=background, 1..N=gland ids).
    Used by both Mask R-CNN and Mask2Former (panoptic) backends.
    """
    # Predicción de la máscara del contorno del párpado (Tarsus) con modelo mejorado
    original_image, mask_tarsus = predict_unet_model(unet_model, image_path, device, use_clahe=True, use_tta=True)

    # Redimensionar la máscara de Tarsus a las dimensiones de la máscara de instancias
    mask_tarsus_resized = F.interpolate(mask_tarsus.unsqueeze(0), size=pred_instance.shape, mode='bilinear', align_corners=True)
    mask_tarsus_resized = mask_tarsus_resized.squeeze(0).cpu().numpy()

    # Aquí, aún tenemos forma (1, H, W); eliminamos la dimensión extra:
    mask_tarsus_resized = np.squeeze(mask_tarsus_resized, axis=0)

    # Aplicar la máscara de Tarsus a la máscara de instancias (conservar solo las instancias dentro del contorno)
    pred_instance_cleaned = pred_instance * mask_tarsus_resized.astype(np.int32)
    pred_instance_cleaned = np.squeeze(pred_instance_cleaned)

    # Crear imagen de instancias: asignar un color aleatorio único a cada instancia
    colored_instance_image = np.zeros((pred_instance_cleaned.shape[0], pred_instance_cleaned.shape[1], 3), dtype=np.uint8)

    # Calcular métricas para cada glándula
    gland_tortuosities = []
    gland_lengths = []
    gland_thicknesses = []
    gland_ids = np.unique(pred_instance_cleaned)
    gland_ids = gland_ids[gland_ids > 0]  # Excluir el fondo (0)

    # Inicializar analizadores Pipeline (métricas calibradas + tortuosidad multi-índice)
    cal = SpatialCalibrator(um_per_px=um_per_px)
    morph_analyzer = GlandMorphometry(cal)
    tort_analyzer = TortuosityAnalyzer()
    pipeline_results = []

    for i in gland_ids:
        gland_mask = (pred_instance_cleaned == i).astype(np.uint8)
        metrics = calculate_gland_tortuosity(gland_mask)
        if metrics is None:
            # Glándula demasiado pequeña o fragmentada — se excluye
            continue
        # Misma morfometría que pipeline (esqueleto + SG) para que avg_length_px y avg_length_um cuadren
        morph_data = morph_analyzer.measure(gland_mask)
        gland_tortuosities.append(metrics['tortuosity'])
        gland_lengths.append(morph_data["length_px"])
        gland_thicknesses.append(morph_data["thickness_px"])

        # Análisis enriquecido: métricas en µm + ICM, ITA, DCF, IMCC, score, grade
        skel_p = skeletonize((gland_mask > 0).astype(np.uint8))
        ordered_pts = morph_analyzer._order_skeleton_points(skel_p)
        # Suavizado Savitzky-Golay: elimina el zig-zag digital del esqueleto
        # antes de calcular ITA y curvatura (igual que en calculate_gland_tortuosity)
        smooth_pts = _smooth_path(ordered_pts)
        tort_data = tort_analyzer.compute(smooth_pts, morph_data["length_um"])
        pipeline_results.append({**morph_data, **tort_data, "gland_id": f"G{len(pipeline_results)+1}"})

        color = generate_random_color()
        colored_instance_image[pred_instance_cleaned == i] = color

    # Filtrar outliers: tortuosidad > percentil 95 (cuando hay suficientes muestras)
    if len(gland_tortuosities) > 3:
        p95 = float(np.percentile(gland_tortuosities, 95))
        keep = [t <= p95 for t in gland_tortuosities]
        gland_tortuosities = [v for v, k in zip(gland_tortuosities, keep) if k]
        gland_lengths      = [v for v, k in zip(gland_lengths,      keep) if k]
        gland_thicknesses  = [v for v, k in zip(gland_thicknesses,  keep) if k]
        pipeline_results   = [v for v, k in zip(pipeline_results,   keep) if k]

    # Tortuosidad clásica: mediana robusta; longitud/grosor en px para tarjetas sin calibrar = media
    avg_tortuosity = float(np.median(gland_tortuosities)) if gland_tortuosities else 0.0
    avg_length     = float(np.mean(gland_lengths))       if gland_lengths      else 0.0
    avg_thickness  = float(np.mean(gland_thicknesses))   if gland_thicknesses  else 0.0

    # Resumen Pipeline (métricas calibradas en µm + clasificación clínica)
    if pipeline_results:
        from collections import Counter
        # Tarjetas ITA / longitud / grosor: promedio aritmético; ICM y score siguen con mediana
        avg_length_um    = float(np.mean([r['length_um']        for r in pipeline_results]))
        avg_thickness_um = float(np.mean([r['thickness_um']     for r in pipeline_results]))
        avg_ICM          = float(np.median([r['ICM']              for r in pipeline_results]))
        avg_ITA          = float(np.mean([r['ITA_deg']          for r in pipeline_results]))
        avg_score        = float(np.median([r['tortuosity_score'] for r in pipeline_results]))
        dominant_grade   = Counter([r['tortuosity_grade'] for r in pipeline_results]).most_common(1)[0][0]
    else:
        avg_length_um = avg_thickness_um = avg_ICM = avg_ITA = avg_score = 0.0
        dominant_grade = "Normal"

    # Abrir la imagen original
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Superponer las instancias (colores sólidos) sobre la imagen original
    result_image = image_np.copy()
    mask_inst = (colored_instance_image.sum(axis=-1) > 0)
    result_image[mask_inst] = colored_instance_image[mask_inst]

    # Redimensionar la máscara de Tarsus para que tenga el mismo tamaño que la imagen original para su visualización
    tarsus_mask_overlay = F.interpolate(mask_tarsus.unsqueeze(0), size=(image_np.shape[0], image_np.shape[1]), mode='bilinear', align_corners=True)
    tarsus_mask_overlay = tarsus_mask_overlay.squeeze(0).cpu().numpy()
    tarsus_mask_overlay = np.squeeze(tarsus_mask_overlay, axis=0)
    tarsus_mask_overlay = (tarsus_mask_overlay > 0.5).astype(np.uint8)

    plt.figure(figsize=(10, 10))
    plt.imshow(result_image)
    plt.imshow(tarsus_mask_overlay, cmap="jet", alpha=0.5)
    plt.title(f"Instancias (colores únicos) y contorno del párpado\nTortuosidad mediana: {avg_tortuosity:.3f}")
    plt.axis("off")
    # plt.show() # Commented out for Streamlit integration

    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = Image.open(buf)
    plt.close()

    binary_mask_glands = (pred_instance_cleaned > 0).astype(np.uint8)

    return img_arr, {
        'avg_tortuosity': avg_tortuosity,
        'individual_tortuosities': gland_tortuosities,
        'num_glands': len(gland_tortuosities),
        'avg_length_px': avg_length,
        'avg_thickness_px': avg_thickness,
        'individual_lengths': gland_lengths,
        'individual_thicknesses': gland_thicknesses,
        'binary_mask_glands': binary_mask_glands,
        # Pipeline: métricas calibradas + clasificación clínica
        'um_per_px': um_per_px,
        'avg_length_um': round(avg_length_um, 1),
        'avg_thickness_um': round(avg_thickness_um, 1),
        'avg_ICM': round(avg_ICM, 4),
        'avg_ITA_deg': round(avg_ITA, 2),
        'avg_tortuosity_score': round(avg_score, 1),
        'dominant_grade': dominant_grade,
        'individual_glands': pipeline_results,
    }


def show_combined_result_with_models(image_path, maskrcnn_model, unet_model, device, um_per_px: float = 1.0):
    pred_instance = predict_maskrcnn_model(maskrcnn_model, image_path, device)
    return compute_results_from_instance_map(pred_instance, image_path, unet_model, device, um_per_px)


def show_combined_result(image_path, maskrcnn_model_path, unet_model_path, device):
    # Cargar modelos
    maskrcnn_model = load_maskrcnn_model(maskrcnn_model_path)
    unet_model = load_unet_model(unet_model_path, device)

    # Predicción de la máscara de instancias (glándulas de Meibomio)
    pred_instance = predict_maskrcnn_model(maskrcnn_model, image_path, device)

    # Predicción de la máscara del contorno del párpado (Tarsus) con modelo mejorado
    original_image, mask_tarsus = predict_unet_model(unet_model, image_path, device, use_clahe=True, use_tta=True)

    # Redimensionar la máscara de Tarsus a las dimensiones de la máscara de instancias
    mask_tarsus_resized = F.interpolate(mask_tarsus.unsqueeze(0), size=pred_instance.shape, mode='bilinear', align_corners=True)
    mask_tarsus_resized = mask_tarsus_resized.squeeze(0).cpu().numpy()

    # Aquí, aún tenemos forma (1, H, W); eliminamos la dimensión extra:
    mask_tarsus_resized = np.squeeze(mask_tarsus_resized, axis=0)

    # Aplicar la máscara de Tarsus a la máscara de instancias (conservar solo las instancias dentro del contorno)
    pred_instance_cleaned = pred_instance * mask_tarsus_resized.astype(np.int32)
    pred_instance_cleaned = np.squeeze(pred_instance_cleaned)

    # Crear imagen de instancias: asignar un color aleatorio único a cada instancia
    colored_instance_image = np.zeros((pred_instance_cleaned.shape[0], pred_instance_cleaned.shape[1], 3), dtype=np.uint8)

    # Calcular métricas para cada glándula
    gland_tortuosities = []
    gland_lengths = []
    gland_thicknesses = []
    gland_ids = np.unique(pred_instance_cleaned)
    gland_ids = gland_ids[gland_ids > 0]  # Excluir el fondo (0)

    for i in gland_ids:
        gland_mask = (pred_instance_cleaned == i).astype(np.uint8)
        metrics = calculate_gland_tortuosity(gland_mask)
        if metrics is None:
            continue
        gland_tortuosities.append(metrics['tortuosity'])
        gland_lengths.append(metrics['length_px'])
        gland_thicknesses.append(metrics['thickness_px'])

        color = generate_random_color()
        colored_instance_image[pred_instance_cleaned == i] = color

    # Filtrar outliers: tortuosidad > percentil 95
    if len(gland_tortuosities) > 3:
        p95 = float(np.percentile(gland_tortuosities, 95))
        keep = [t <= p95 for t in gland_tortuosities]
        gland_tortuosities = [v for v, k in zip(gland_tortuosities, keep) if k]
        gland_lengths      = [v for v, k in zip(gland_lengths,      keep) if k]
        gland_thicknesses  = [v for v, k in zip(gland_thicknesses,  keep) if k]

    # Promedio robusto: mediana
    avg_tortuosity = float(np.median(gland_tortuosities)) if gland_tortuosities else 0.0
    avg_length     = float(np.median(gland_lengths))      if gland_lengths      else 0.0
    avg_thickness  = float(np.median(gland_thicknesses))  if gland_thicknesses  else 0.0

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    result_image = image_np.copy()
    mask_inst = (colored_instance_image.sum(axis=-1) > 0)
    result_image[mask_inst] = colored_instance_image[mask_inst]

    tarsus_mask_overlay = F.interpolate(mask_tarsus.unsqueeze(0), size=(image_np.shape[0], image_np.shape[1]), mode='bilinear', align_corners=True)
    tarsus_mask_overlay = tarsus_mask_overlay.squeeze(0).cpu().numpy()
    tarsus_mask_overlay = np.squeeze(tarsus_mask_overlay, axis=0)
    tarsus_mask_overlay = (tarsus_mask_overlay > 0.5).astype(np.uint8)

    plt.figure(figsize=(10, 10))
    plt.imshow(result_image)
    plt.imshow(tarsus_mask_overlay, cmap="jet", alpha=0.5)
    plt.title(f"Instancias (colores únicos) y contorno del párpado\nTortuosidad mediana: {avg_tortuosity:.3f}")
    plt.axis("off")
    # plt.show() # Commented out for Streamlit integration

    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = Image.open(buf)
    plt.close()

    return img_arr, {
        'avg_tortuosity': avg_tortuosity,
        'individual_tortuosities': gland_tortuosities,
        'num_glands': len(gland_tortuosities),
        'avg_length_px': avg_length,
        'avg_thickness_px': avg_thickness,
        'individual_lengths': gland_lengths,
        'individual_thicknesses': gland_thicknesses,
    }


# ------------------------------------------------------
# Función para redimensionar la imagen a múltiplos de 32
# ------------------------------------------------------
def resize_to_previous_multiple_of_32(img):
    """
    Redimensiona la imagen a dimensiones que sean múltiplos de 32, redondeando hacia abajo.
    Esto preserva la relación de aspecto, pero puede recortar algunos píxeles.
    """
    w, h = img.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    return img.resize((new_w, new_h), Image.BILINEAR)

# ------------------------------------------------------
# Bloque principal (Commented out for Streamlit integration)
# ------------------------------------------------------
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     maskrcnn_model_path = "final_model (11).pth"
#     unet_model_path = "final_model_tarsus_improved (6).pth"  # Usar el modelo mejorado
#     image_path = "meibomio.jpg"

#     show_combined_result(image_path, maskrcnn_model_path, unet_model_path, device)
