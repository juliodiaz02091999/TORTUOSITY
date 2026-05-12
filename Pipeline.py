"""
Métricas de meibografía (µm) + tortuosidad multi-índice (ICM, ITA, DCF, IMCC).

En producción, Tortuosity.py importa SpatialCalibrator, GlandMorphometry y
TortuosityAnalyzer desde este módulo. Mantener este archivo en el repo (p. ej. para Docker/Cloud Run).

Score clínico: solo ICM + ITA (50 % / 50 %). DCF e IMCC se devuelven como auxiliares.

Uso rápido:
    pipeline = MeibographyPipeline(um_per_px=11.76)
    results = pipeline.analyze(masks_tensor, image)
    df = results.to_dataframe()
"""

import torch
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings


# ---------------------------------------------------------------------------
# MÓDULO 1 – CALIBRACIÓN ESPACIAL
# ---------------------------------------------------------------------------

class SpatialCalibrator:
    """
    Convierte píxeles a micrómetros usando tres estrategias:
      A) Factor directo conocido del equipo (recomendado)
      B) Reglilla de calibración en la imagen
      C) Metadata DICOM (campo PixelSpacing)
    """

    def __init__(self, um_per_px: float = None):
        self.um_per_px = um_per_px

    @classmethod
    def from_reference_bar(
        cls,
        image: np.ndarray,
        bar_real_um: float = 1000.0,
        bar_color_lower: Tuple = (200, 200, 200),
        bar_color_upper: Tuple = (255, 255, 255),
    ) -> "SpatialCalibrator":
        """
        Detecta una reglilla blanca/gris en la imagen y calcula K.
        bar_real_um: longitud real conocida de la barra en µm.
        """
        mask = cv2.inRange(image, np.array(bar_color_lower), np.array(bar_color_upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No se encontró barra de referencia en la imagen.")
        bar = max(contours, key=cv2.contourArea)
        x, _, w, _ = cv2.boundingRect(bar)
        um_per_px = bar_real_um / w
        print(f"[Calibración] Barra detectada: {w} px → K = {um_per_px:.4f} µm/px")
        return cls(um_per_px=um_per_px)

    @classmethod
    def from_dicom(cls, dicom_path: str) -> "SpatialCalibrator":
        """Lee PixelSpacing del header DICOM (requiere pydicom)."""
        try:
            import pydicom
            ds = pydicom.dcmread(dicom_path)
            spacing_mm = float(ds.PixelSpacing[0])
            um_per_px = spacing_mm * 1000.0
            print(f"[Calibración DICOM] PixelSpacing = {spacing_mm} mm/px → K = {um_per_px:.4f} µm/px")
            return cls(um_per_px=um_per_px)
        except ImportError:
            raise ImportError("Instala pydicom: pip install pydicom")

    # Valores típicos por equipo (referencia)
    EQUIPMENT_PRESETS = {
        "keratograph_5m_10x":  8.0,
        "keratograph_5m_16x":  5.0,
        "lipiview_ii":         6.5,
        "slit_lamp_40x":      11.76,
    }

    @classmethod
    def from_equipment(cls, name: str) -> "SpatialCalibrator":
        if name not in cls.EQUIPMENT_PRESETS:
            raise KeyError(f"Equipo desconocido. Opciones: {list(cls.EQUIPMENT_PRESETS.keys())}")
        k = cls.EQUIPMENT_PRESETS[name]
        print(f"[Calibración] Equipo '{name}' → K = {k} µm/px")
        return cls(um_per_px=k)

    def px_to_um(self, value_px: float) -> float:
        if self.um_per_px is None:
            raise ValueError("Factor K no calibrado. Usa from_reference_bar, from_dicom o from_equipment.")
        return value_px * self.um_per_px

    def __repr__(self):
        return f"SpatialCalibrator(K={self.um_per_px} µm/px)"


# ---------------------------------------------------------------------------
# MÓDULO 2 – MORFOMETRÍA (Longitud y Grosor reales)
# ---------------------------------------------------------------------------

class GlandMorphometry:
    """
    Extrae longitud y grosor de una glándula meibomiana a partir de su máscara binaria.
    Usa esqueleto + transformada de distancia para mayor precisión.
    """

    def __init__(self, calibrator: SpatialCalibrator):
        self.cal = calibrator

    def _extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """Esqueleto medial de la máscara binaria (Zhang-Suen)."""
        binary = (mask > 0).astype(np.uint8)
        return skeletonize(binary).astype(np.uint8)

    def _order_skeleton_points(self, skel: np.ndarray) -> np.ndarray:
        """
        Recorre el esqueleto como una sola polilínea continua (8-vecinos), extremo a extremo.

        El BFS por niveles que había antes podía, en bifurcaciones, colocar píxeles
        consecutivos en la lista que no son vecinos en el grafo del esqueleto; eso
        introduce saltos y ángulos espurios y sube ITA/ICM → grados clínicos inflados.
        Misma idea que el recorrido greedy de Tortuosity._skeleton_to_path.
        """
        ys, xs = np.where(skel > 0)
        if len(xs) < 2:
            return np.argwhere(skel > 0)

        pts = list(zip(ys.tolist(), xs.tolist()))
        coord_set = set(pts)

        def get_nbrs(y, x):
            return [
                (y + dy, x + dx)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if (dy or dx) and (y + dy, x + dx) in coord_set
            ]

        endpoints = [p for p in pts if len(get_nbrs(*p)) == 1]
        start = endpoints[0] if endpoints else pts[0]

        path = [start]
        visited = {start}
        cur = start
        while True:
            nbrs = [p for p in get_nbrs(*cur) if p not in visited]
            if not nbrs:
                break
            cur = (
                nbrs[0]
                if len(nbrs) == 1
                else max(
                    nbrs,
                    key=lambda p: len([q for q in get_nbrs(*p) if q not in visited]),
                )
            )
            path.append(cur)
            visited.add(cur)

        return np.array(path)

    def _smooth_polyline(
        self, path: np.ndarray, window_length: int = 11, polyorder: int = 2
    ) -> np.ndarray:
        """
        Suaviza la polilínea del esqueleto (Savitzky–Golay) para quitar el zig-zag
        de la malla en píxeles. Sin esto, la suma de segmentos sobreestima la longitud
        real del eje glandular (valores absurdos en µm con K correcto).
        """
        if path is None or len(path) < 4:
            return path.astype(float) if path is not None else path
        n = len(path)
        wl = min(window_length, n)
        if wl % 2 == 0:
            wl -= 1
        if wl < polyorder + 2:
            wl = polyorder + 2
            if wl % 2 == 0:
                wl += 1
        if wl > n:
            return path.astype(float)
        ys = savgol_filter(path[:, 0].astype(float), wl, polyorder)
        xs = savgol_filter(path[:, 1].astype(float), wl, polyorder)
        return np.column_stack([ys, xs])

    def _polyline_length(self, pts: np.ndarray) -> float:
        """Longitud acumulada del polígono de puntos (conectividad 8)."""
        if len(pts) < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        return float(dists.sum())

    def measure(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Retorna:
            length_px, length_um, thickness_px, thickness_um,
            aspect_ratio, area_um2
        """
        binary = (mask > 0).astype(np.uint8)
        if binary.sum() == 0:
            return {k: 0.0 for k in
                    ["length_px","length_um","thickness_px","thickness_um",
                     "aspect_ratio","area_um2"]}

        skel = self._extract_skeleton(binary)
        pts  = self._order_skeleton_points(skel)
        smooth = self._smooth_polyline(pts)

        # Longitud = eje sobre polilínea suavizada (alinea con Tortuosity.calculate_gland_tortuosity)
        length_px = self._polyline_length(smooth)

        # Grosor = 2 × media de la transformada de distancia sobre el esqueleto
        dist_map  = distance_transform_edt(binary)
        skel_mask = skel > 0
        thickness_px = float(2.0 * dist_map[skel_mask].mean()) if skel_mask.any() else 0.0

        # Área
        area_px2 = float(binary.sum())

        length_um    = self.cal.px_to_um(length_px)
        thickness_um = self.cal.px_to_um(thickness_px)
        area_um2     = area_px2 * (self.cal.um_per_px ** 2)

        return {
            "length_px":    round(length_px, 2),
            "length_um":    round(length_um, 1),
            "thickness_px": round(thickness_px, 2),
            "thickness_um": round(thickness_um, 1),
            "aspect_ratio": round(length_um / max(thickness_um, 1e-9), 2),
            "area_um2":     round(area_um2, 1),
        }


# ---------------------------------------------------------------------------
# MÓDULO 3 – TORTUOSIDAD MULTI-ÍNDICE
# ---------------------------------------------------------------------------

class TortuosityAnalyzer:
    """
    Calcula 4 índices complementarios de tortuosidad sobre el esqueleto ordenado.

    ICM  – Índice de Curva Media (longitud real / distancia directa)
    ITA  – Índice de Ángulo Tangente acumulado (°)
    DCF  – Densidad de Curvatura e Inflexiones (inflexiones / µm)
    IMCC – Índice de Máxima Curvatura local × longitud media
    """

    WINDOW = 5  # puntos para curvatura local

    def _local_curvature(self, pts: np.ndarray) -> np.ndarray:
        """
        Curvatura de Menger: κ = 4·Area_triángulo / (|AB|·|BC|·|CA|)
        para cada triplete de puntos consecutivos.
        """
        if len(pts) < 3:
            return np.array([0.0])

        w = self.WINDOW
        curv = []
        for i in range(w, len(pts) - w):
            A = pts[i - w].astype(float)
            B = pts[i].astype(float)
            C = pts[i + w].astype(float)

            ab = np.linalg.norm(B - A)
            bc = np.linalg.norm(C - B)
            ca = np.linalg.norm(A - C)

            # Área del triángulo por producto cruzado
            area = abs((B[0]-A[0])*(C[1]-A[1]) - (C[0]-A[0])*(B[1]-A[1])) / 2.0
            denom = ab * bc * ca
            kappa = (4.0 * area / denom) if denom > 1e-9 else 0.0
            curv.append(kappa)

        return np.array(curv) if curv else np.array([0.0])

    def _tangent_angles(self, pts: np.ndarray) -> np.ndarray:
        """Ángulo tangente en cada punto (radianes)."""
        if len(pts) < 2:
            return np.array([0.0])
        diffs = np.diff(pts.astype(float), axis=0)
        return np.arctan2(diffs[:, 0], diffs[:, 1])

    @staticmethod
    def _resample_equidistant(pts: np.ndarray, n_points: int = 40) -> np.ndarray:
        """
        Resample a path to n_points equidistant points using linear interpolation.
        Eliminates pixel-level staircase noise before angle computation.
        """
        if len(pts) < 2:
            return pts
        diffs = np.diff(pts.astype(float), axis=0)
        seg_len = np.sqrt((diffs ** 2).sum(axis=1))
        cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cumlen[-1]
        if total < 1e-9:
            return pts
        positions = np.linspace(0, total, n_points)
        new_y = np.interp(positions, cumlen, pts[:, 0].astype(float))
        new_x = np.interp(positions, cumlen, pts[:, 1].astype(float))
        return np.column_stack([new_y, new_x])

    def compute(self, ordered_pts: np.ndarray, length_um: float) -> Dict[str, float]:
        """
        Recibe los puntos ordenados del esqueleto y la longitud real.
        Retorna diccionario con los 4 índices + clasificación clínica.
        """
        if len(ordered_pts) < 4 or length_um < 1:
            return {
                "ICM": 1.0, "ITA_deg": 0.0, "DCF": 0.0, "IMCC": 0.0,
                "tortuosity_grade": "Normal", "tortuosity_score": 0.0
            }

        pts = ordered_pts.astype(float)

        # --- ICM ---
        direct_dist = np.linalg.norm(pts[-1] - pts[0])
        diffs = np.diff(pts, axis=0)
        real_length = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
        ICM = real_length / max(direct_dist, 1e-9)

        # --- ITA ---
        # Resample to 40 equidistant points before angle computation.
        # This removes pixel-level staircase noise that accumulates to hundreds
        # of spurious degrees even on near-straight glands.
        pts_ita = self._resample_equidistant(pts, n_points=40)
        angles = self._tangent_angles(pts_ita)
        delta_angles = np.abs(np.diff(np.unwrap(angles)))
        ITA_deg = float(np.degrees(delta_angles.sum()))

        # --- DCF ---
        curv = self._local_curvature(pts)
        # Inflexiones = cambios de signo en la curvatura derivada
        if len(curv) > 2:
            curv_sign = np.sign(np.diff(curv))
            inflections = int(np.sum(np.abs(np.diff(curv_sign)) > 0))
        else:
            inflections = 0
        DCF = inflections / max(length_um, 1.0)  # inflexiones por µm

        # --- IMCC ---
        kappa_max = float(curv.max()) if len(curv) > 0 else 0.0
        IMCC = kappa_max * (length_um / 1000.0)  # normalizado a mm

        # --- Score clínico (solo ICM) ---
        # ICM saturado en 1.0 (100% más largo que la cuerda = tortuosidad muy alta).
        score = min((ICM - 1.0) / 1.0, 1.0) * 100

        # Clasificación
        if score < 20:
            grade = "Normal"
        elif score < 45:
            grade = "Leve"
        elif score < 70:
            grade = "Moderada"
        else:
            grade = "Severa"

        return {
            "ICM":               round(ICM, 4),
            "ITA_deg":           round(ITA_deg, 2),
            "DCF":               round(DCF, 6),
            "IMCC":              round(IMCC, 4),
            "tortuosity_score":  round(score, 1),
            "tortuosity_grade":  grade,
        }


# ---------------------------------------------------------------------------
# MÓDULO 4 – PIPELINE PRINCIPAL (integra todo)
# ---------------------------------------------------------------------------

@dataclass
class GlandResult:
    gland_id:       str
    morphometry:    Dict[str, float]
    tortuosity:     Dict[str, float]

    def to_dict(self) -> Dict:
        return {"gland_id": self.gland_id, **self.morphometry, **self.tortuosity}


@dataclass
class PipelineResult:
    glands:         List[GlandResult] = field(default_factory=list)
    summary:        Dict[str, float]  = field(default_factory=dict)
    calibration_k:  float = 0.0

    def to_dataframe(self):
        """Convierte resultados a pandas DataFrame."""
        try:
            import pandas as pd
            rows = [g.to_dict() for g in self.glands]
            return pd.DataFrame(rows)
        except ImportError:
            raise ImportError("Instala pandas: pip install pandas")

    def print_summary(self):
        print("\n" + "="*60)
        print(f"  RESUMEN MEIBOGRAFÍA  (K = {self.calibration_k:.2f} µm/px)")
        print("="*60)
        for k, v in self.summary.items():
            print(f"  {k:<35} {v}")
        print("="*60)
        print(f"  Glándulas analizadas: {len(self.glands)}")


class MeibographyPipeline:
    """
    Pipeline principal. Acepta máscaras de instancia en tres formatos:
      - torch.Tensor  [N, H, W]  (salida de Mask R-CNN, SAM, etc.)
      - np.ndarray    [N, H, W]
      - List[np.ndarray]  (lista de máscaras binarias individuales)

    Ejemplo:
        pipeline = MeibographyPipeline(um_per_px=11.76)
        result = pipeline.analyze(masks, image)
        df = result.to_dataframe()
        result.print_summary()
    """

    # Rangos clínicos de referencia (literatura)
    CLINICAL_NORMS = {
        "length_um":    (3000, 8000),   # µm
        "thickness_um": (200,  600),    # µm
        "ICM":          (1.0,  1.10),   # ratio
        "ITA_deg":      (0,    30),     # grados
    }

    def __init__(
        self,
        um_per_px: float = None,
        calibrator: SpatialCalibrator = None,
        equipment: str = None,
        min_area_px: int = 100,
    ):
        """
        Proporciona um_per_px directo, un SpatialCalibrator, o el nombre del equipo.
        min_area_px: ignora máscaras más pequeñas (ruido).
        """
        if calibrator:
            self.cal = calibrator
        elif um_per_px:
            self.cal = SpatialCalibrator(um_per_px=um_per_px)
        elif equipment:
            self.cal = SpatialCalibrator.from_equipment(equipment)
        else:
            warnings.warn(
                "No se especificó calibración. Las medidas serán en píxeles (K=1).",
                UserWarning
            )
            self.cal = SpatialCalibrator(um_per_px=1.0)

        self.morph   = GlandMorphometry(self.cal)
        self.tort    = TortuosityAnalyzer()
        self.min_area = min_area_px

    def _to_mask_list(self, masks) -> List[np.ndarray]:
        """Normaliza la entrada a lista de máscaras binarias numpy."""
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                masks = [masks]
            else:
                masks = [masks[i] for i in range(masks.shape[0])]
        return [m.astype(np.uint8) for m in masks]

    def analyze(
        self,
        masks,
        image: np.ndarray = None,
        ids: List[str] = None,
    ) -> PipelineResult:
        """
        Analiza todas las instancias de glándula.

        Args:
            masks: Tensor [N,H,W], ndarray [N,H,W] o list de máscaras binarias
            image: imagen original (opcional, para visualización futura)
            ids:   lista de IDs; si None usa G1, G2, ...

        Returns:
            PipelineResult con métricas por glándula + resumen global
        """
        mask_list = self._to_mask_list(masks)
        results   = PipelineResult(calibration_k=self.cal.um_per_px)

        valid_idx = 0
        for i, mask in enumerate(mask_list):
            # Filtrar glándulas demasiado pequeñas
            if mask.sum() < self.min_area:
                continue

            gland_id = ids[i] if ids else f"G{valid_idx + 1}"
            valid_idx += 1

            # ---- Morfometría ----
            morph_data = self.morph.measure(mask)

            # ---- Tortuosidad ----
            skel = skeletonize((mask > 0).astype(np.uint8))
            ordered = self.morph._order_skeleton_points(skel)
            smooth = self.morph._smooth_polyline(ordered)
            tort_data = self.tort.compute(smooth, morph_data["length_um"])

            results.glands.append(GlandResult(
                gland_id    = gland_id,
                morphometry = morph_data,
                tortuosity  = tort_data,
            ))

        # ---- Resumen global ----
        if results.glands:
            results.summary = self._compute_summary(results.glands)

        return results

    def _compute_summary(self, glands: List[GlandResult]) -> Dict:
        lengths    = [g.morphometry["length_um"]    for g in glands]
        thicknesses= [g.morphometry["thickness_um"] for g in glands]
        ICMs       = [g.tortuosity["ICM"]           for g in glands]
        ITAs       = [g.tortuosity["ITA_deg"]       for g in glands]
        scores     = [g.tortuosity["tortuosity_score"] for g in glands]
        grades     = [g.tortuosity["tortuosity_grade"] for g in glands]

        def mean(lst): return round(float(np.mean(lst)), 2)
        def rng(lst):  return f"{min(lst):.1f} – {max(lst):.1f}"

        # Cobertura palpebral (% glándulas con longitud normal)
        normal_count = sum(
            1 for l in lengths
            if self.CLINICAL_NORMS["length_um"][0] <= l <= self.CLINICAL_NORMS["length_um"][1]
        )
        coverage = round(100 * normal_count / len(glands), 1)

        # Grado predominante
        from collections import Counter
        dominant_grade = Counter(grades).most_common(1)[0][0]

        return {
            "N glándulas analizadas":    len(glands),
            "Longitud media (µm)":       mean(lengths),
            "Longitud rango (µm)":       rng(lengths),
            "Grosor medio (µm)":         mean(thicknesses),
            "Grosor rango (µm)":         rng(thicknesses),
            "ICM medio":                 mean(ICMs),
            "ITA medio (°)":             mean(ITAs),
            "Score tortuosidad medio":   mean(scores),
            "Grado predominante":        dominant_grade,
            "Cobertura palpebral (%)":   coverage,
        }

    def flag_abnormal(self, result: PipelineResult) -> List[Dict]:
        """
        Devuelve lista de glándulas fuera de rangos clínicos normales.
        Útil para resaltar en el dashboard.
        """
        flagged = []
        for g in result.glands:
            alerts = []
            m, t = g.morphometry, g.tortuosity

            if m["length_um"] < self.CLINICAL_NORMS["length_um"][0]:
                alerts.append(f"Longitud baja ({m['length_um']} µm)")
            if not (self.CLINICAL_NORMS["thickness_um"][0] <= m["thickness_um"] <= self.CLINICAL_NORMS["thickness_um"][1]):
                alerts.append(f"Grosor anormal ({m['thickness_um']} µm)")
            if t["ICM"] > self.CLINICAL_NORMS["ICM"][1]:
                alerts.append(f"ICM elevado ({t['ICM']})")
            if t["ITA_deg"] > self.CLINICAL_NORMS["ITA_deg"][1]:
                alerts.append(f"ITA elevado ({t['ITA_deg']}°)")

            if alerts:
                flagged.append({"gland_id": g.gland_id, "alerts": alerts})

        return flagged


# ---------------------------------------------------------------------------
# EJEMPLO DE USO COMPLETO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("  DEMO: Pipeline de Meibografía")
    print("="*60)

    # --- Simular máscaras de instancia (salida de Mask R-CNN) ---
    np.random.seed(42)
    H, W = 512, 512
    N = 5  # 5 glándulas simuladas

    fake_masks = []
    for i in range(N):
        m = np.zeros((H, W), dtype=np.uint8)
        # Elipse alargada (simula una glándula meibomiana)
        cy = 100 + i * 70
        cx = 256
        # Añadir ligera curvatura desplazando el centro
        cv2.ellipse(m, (cx + i*10, cy), (80, 15), i * 10, 0, 360, 1, -1)
        fake_masks.append(m)

    masks_tensor = torch.tensor(np.stack(fake_masks))  # [N, H, W]

    # --- Inicializar pipeline ---
    # Opción A: factor conocido del equipo
    pipeline = MeibographyPipeline(equipment="slit_lamp_40x")

    # Opción B: factor manual
    # pipeline = MeibographyPipeline(um_per_px=11.76)

    # Opción C: desde reglilla en imagen
    # cal = SpatialCalibrator.from_reference_bar(image, bar_real_um=1000)
    # pipeline = MeibographyPipeline(calibrator=cal)

    # --- Analizar ---
    result = pipeline.analyze(masks_tensor)

    # --- Mostrar resumen ---
    result.print_summary()

    # --- DataFrame por glándula ---
    df = result.to_dataframe()
    print("\nMétricas por glándula:")
    print(df[["gland_id","length_um","thickness_um","ICM","ITA_deg",
              "tortuosity_score","tortuosity_grade"]].to_string(index=False))

    # --- Glándulas anómalas ---
    flagged = pipeline.flag_abnormal(result)
    if flagged:
        print(f"\nAlertas clínicas ({len(flagged)} glándulas):")
        for f in flagged:
            print(f"  {f['gland_id']}: {', '.join(f['alerts'])}")
    else:
        print("\nTodas las glándulas dentro de rangos normales.")
