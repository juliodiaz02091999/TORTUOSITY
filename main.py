from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import base64
import io
from PIL import Image
import json
from typing import Optional, Dict, Any
import numpy as np
from skimage import exposure, img_as_ubyte, img_as_float32
import mgda as mgda

# Import the tortuosity analysis functions
from Tortuosity import (
    load_maskrcnn_model,
    load_unet_model,
    predict_maskrcnn_model,
    predict_unet_model,
    show_combined_result,
    show_combined_result_with_models,
    compute_results_from_instance_map,
    resize_to_previous_multiple_of_32,
    device
)
from panoptic import build_model as build_panoptic_model, load_checkpoint as load_panoptic_checkpoint, predict_panoptic_in_memory
from transformers import AutoImageProcessor
import maskcrnn as maskrcnn_v2

# Create FastAPI app
app = FastAPI(
    title="Análisis de Tortuosidad Avanzado API",
    description="API para análisis de tortuosidad de glándulas de Meibomio usando PyTorch (Mask R-CNN & UNet)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for temporary files and results
TEMP_DIR = Path("temp")
RESULTS_DIR = Path("results")
STATIC_DIR = Path("static")
TEMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files only if directory has content
if any(STATIC_DIR.iterdir()) if STATIC_DIR.exists() else False:
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Model paths - with fallbacks
MASK_RCNN_MODEL_PATH = "final_model (11).pth"
UNET_MODEL_PATH = "final_model_tarsus_improved (6).pth"  # Mejor modelo con attention gates y CLAHE

# Fallback model paths
FALLBACK_MASK_RCNN_PATHS = [
    "final_model (11).pth",
    "final_model_tarsus.pth",
    "final_model.pth"
]

FALLBACK_UNET_PATHS = [
    "final_model_tarsus_improved (6).pth",  # Prioridad al mejor modelo
    "final_model_tarsus_improved.pth",      # Backup
    "final_model_tarsus.pth"               # Último recurso
]

PANOPTIC_CKPT_PATH = "best_model (13).pth"
PANOPTIC_MODEL_ID = "facebook/mask2former-swin-small-cityscapes-instance"
MASKRCNN2_CKPT_PATH = "best_model (17).pth"

# Global model instances (loaded once at startup)
maskrcnn_model = None
unet_model = None
meibomio_model = None
panoptic_model = None
panoptic_processor = None
maskrcnn2_model = None

def try_load_model_with_fallbacks(load_function, model_paths, model_name):
    """Try to load a model from multiple possible paths"""
    for path in model_paths:
        try:
            print(f"Trying to load {model_name} from: {path}")
            if os.path.exists(path):
                model = load_function(path)
                print(f"Successfully loaded {model_name} from: {path}")
                return model
            else:
                print(f"Model file not found: {path}")
        except Exception as e:
            print(f"Failed to load {model_name} from {path}: {e}")
            continue
    
    raise Exception(f"Could not load {model_name} from any of the provided paths: {model_paths}")

def clahe_like_imagej(img, block_radius=63, bins=255, slope=3.0, convert_to_gray=True):
    """
    Replica el CLAHE de ImageJ con un error ≤ 1 nivel de gris.
    img : uint8 ó RGB uint8
    convert_to_gray: Si True, convierte RGB a gris antes de aplicar CLAHE
    """
    tile        = 2*block_radius + 1
    clip_limit  = slope / bins      # mapeo exacto
    nbins       = bins + 1

    if img.ndim == 3 and convert_to_gray:
        # Convertir RGB a gris usando la fórmula estándar
        gray_img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        out = exposure.equalize_adapthist(gray_img,
                                          kernel_size=tile,
                                          nbins=nbins,
                                          clip_limit=clip_limit)
        return img_as_ubyte(out)

    elif img.ndim == 2:                        # ya es gris
        out = exposure.equalize_adapthist(img,
                                          kernel_size=tile,
                                          nbins=nbins,
                                          clip_limit=clip_limit)
        return img_as_ubyte(out)

    elif img.ndim == 3 and not convert_to_gray:  # color: trata cada canal
        ch = [clahe_like_imagej(img[..., c],
                                block_radius, bins, slope, convert_to_gray=False)
              for c in range(img.shape[2])]
        return np.stack(ch, axis=-1)

    else:
        raise ValueError("Solo imágenes 2D o RGB.")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global maskrcnn_model, unet_model, meibomio_model, panoptic_model, panoptic_processor, maskrcnn2_model
    try:
        print("Starting model loading process...")
        print(f"Mask R-CNN model path: {MASK_RCNN_MODEL_PATH}")
        print(f"UNet model path: {UNET_MODEL_PATH}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        
        # Check if model files exist
        for path in FALLBACK_MASK_RCNN_PATHS + FALLBACK_UNET_PATHS:
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"✓ Model file exists: {path} ({size:,} bytes)")
                # Try to read first few bytes to check if file is readable
                try:
                    with open(path, 'rb') as f:
                        header = f.read(16)
                        print(f"  File header (hex): {header.hex()}")
                except Exception as e:
                    print(f"  Warning: Could not read file header: {e}")
            else:
                print(f"✗ Model file missing: {path}")
        
        # Print system information
        import platform
        print(f"Python version: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"Architecture: {platform.architecture()}")
        
        # Print PyTorch information
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
        
        # Print memory information
        import psutil
        try:
            memory = psutil.virtual_memory()
            print(f"Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
        except ImportError:
            print("psutil not available for memory info")
        
        # Add a small delay to ensure system is ready
        import asyncio
        await asyncio.sleep(3)
        
        print("Loading Mask R-CNN model...")
        try:
            maskrcnn_model = try_load_model_with_fallbacks(lambda path: load_maskrcnn_model(path, device), FALLBACK_MASK_RCNN_PATHS, "Mask R-CNN")
            print("✓ Mask R-CNN model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Mask R-CNN model: {e}")
            maskrcnn_model = None
        
        print("Loading UNet model...")
        try:
            unet_model = try_load_model_with_fallbacks(lambda path: load_unet_model(path, device), FALLBACK_UNET_PATHS, "UNet")
            print("✓ UNet model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load UNet model: {e}")
            unet_model = None

        # Load MGDA meibomian glands model (separate UNet)
        try:
            meibo_paths = [
                "final_model_improved_fixed.pth",
            ]
            def load_meibo(path):
                # Use pretrained weights for better performance
                return mgda.load_model(path, device, encoder_pretrained=True)
            print("Loading Meibomian glands model (MGDA)...")
            meibomio_model = try_load_model_with_fallbacks(load_meibo, meibo_paths, "MGDA Meibomian")
            print("✓ MGDA meibomian model loaded successfully")
        except Exception as e:
            print(f"⚠ Failed to load MGDA meibomian model: {e}")
            meibomio_model = None
        
        # Load Mask2Former panoptic model
        try:
            print("Loading Mask2Former (panoptic) model...")
            if os.path.exists(PANOPTIC_CKPT_PATH):
                _pan = build_panoptic_model()
                _pan = load_panoptic_checkpoint(_pan, PANOPTIC_CKPT_PATH, device)
                _pan.to(device).eval()
                panoptic_model = _pan
                panoptic_processor = AutoImageProcessor.from_pretrained(PANOPTIC_MODEL_ID, use_fast=False)
                print("✓ Mask2Former (panoptic) model loaded successfully")
            else:
                print(f"✗ Panoptic checkpoint not found: {PANOPTIC_CKPT_PATH}")
        except Exception as e:
            print(f"⚠ Failed to load panoptic model: {e}")
            panoptic_model = None
            panoptic_processor = None

        # Load Mask R-CNN v2 (maskcrnn.py — CLAHE LAB + EXIF + contour crop)
        try:
            print("Loading Mask R-CNN v2 model...")
            if os.path.exists(MASKRCNN2_CKPT_PATH):
                import torch as _torch
                _m2 = maskrcnn_v2.build_model(pretrained=False)
                try:
                    _state = _torch.load(MASKRCNN2_CKPT_PATH, map_location=device, weights_only=True)
                except TypeError:
                    _state = _torch.load(MASKRCNN2_CKPT_PATH, map_location=device)
                _state = maskrcnn_v2._normalize_state_dict(_state)
                _m2.load_state_dict(_state, strict=True)
                _m2.to(device).eval()
                maskrcnn2_model = _m2
                print("✓ Mask R-CNN v2 model loaded successfully")
            else:
                print(f"✗ Mask R-CNN v2 checkpoint not found: {MASKRCNN2_CKPT_PATH}")
        except Exception as e:
            print(f"⚠ Failed to load Mask R-CNN v2 model: {e}")
            maskrcnn2_model = None

        if maskrcnn_model is not None and unet_model is not None:
            print("✓ All models loaded successfully!")
        else:
            print("⚠ Some models failed to load, but continuing startup...")
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise the error immediately, let the app start and handle it gracefully
        print("Warning: Models failed to load, but continuing startup...")
        maskrcnn_model = None
        unet_model = None
    
    # Final status check
    if maskrcnn_model is not None and unet_model is not None:
        print("🎉 Application startup completed successfully with all models loaded!")
    else:
        print("⚠ Application startup completed with degraded functionality - some models failed to load")
        print("   The application will continue to run but may not be able to process images")

@app.post("/analyze-mgda")
async def analyze_mgda(
    file: UploadFile = File(...),
    expansion_mode: str = Form("inferior"),
    background_tasks: BackgroundTasks = None
):
    """Run MGDA analysis (meibomian dysfunction) using preloaded models"""
    if expansion_mode not in ("inferior", "superior"):
        raise HTTPException(status_code=400, detail="expansion_mode must be 'inferior' or 'superior'")

    if meibomio_model is None or unet_model is None:
        raise HTTPException(status_code=503, detail="Required models not loaded (meibomio and/or tarsus)")

    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = Path(file.filename or "").suffix.lower() or ".jpg"
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File extension {file_extension} not allowed. Use: {allowed_extensions}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Analyze
        print(f"[MGDA] Starting analysis. temp_file_path={temp_file_path}, expansion_mode={expansion_mode}")

        # Validate expansion_mode
        if expansion_mode not in ["inferior", "superior"]:
            expansion_mode = "inferior"  # Default fallback

        result_image, metrics = mgda.analyze_mgda_with_models(
            temp_file_path,
            meibomio_model,
            unet_model,
            device,
            expansion_mode=expansion_mode
        )

        # Convert image to base64
        buf = io.BytesIO()
        result_image.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()

        background_tasks.add_task(os.unlink, temp_file_path) if background_tasks else os.unlink(temp_file_path)

        return {
            "success": True,
            "message": "MGDA analysis completed successfully",
            "data": {
                "processed_image": f"data:image/png;base64,{img_base64}",
                "metrics": metrics,
                "expansion_mode": expansion_mode,
                "used_expansion_mode": expansion_mode
            }
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[MGDA] ERROR: {e}\n{tb}")
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail={"error": "MGDA analysis failed", "message": str(e), "trace": tb[-2000:]})

@app.get("/")
async def root():
    """API root — frontend is served from Vercel"""
    return JSONResponse({
        "message": "Tortuosity Analysis API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "frontend": "https://tortuosity2.vercel.app"
    })

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Análisis de Tortuosidad Avanzado API",
        "version": "1.0.0",
        "endpoints": {
            "/": "Main interface",
            "/api": "API information",
            "/health": "Health check",
            "/models/status": "Detailed model status",
            "/analyze": "Analyze image for tortuosity",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = (maskrcnn_model is not None) and (unet_model is not None) and (meibomio_model is not None)

    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": {
            "maskrcnn": maskrcnn_model is not None,
            "unet": unet_model is not None,
            "meibomio": meibomio_model is not None,
            "panoptic": panoptic_model is not None,
            "maskrcnn2": maskrcnn2_model is not None,
        },
        "device": str(device),
        "message": "Models loaded successfully" if models_loaded else "Some models failed to load"
    }

@app.get("/models/status")
async def model_status():
    """Detailed model status endpoint"""
    return {
        "models": {
            "maskrcnn": {
                "status": "loaded" if maskrcnn_model is not None else "failed",
                "path": MASK_RCNN_MODEL_PATH,
                "fallback_paths": FALLBACK_MASK_RCNN_PATHS
            },
            "unet": {
                "status": "loaded" if unet_model is not None else "failed",
                "path": UNET_MODEL_PATH,
                "fallback_paths": FALLBACK_UNET_PATHS
            },
            "meibomio": {
                "status": "loaded" if meibomio_model is not None else "failed",
                "path": "final_model_improved_fixed.pth",
                "fallback_paths": ["final_model_improved_fixed.pth"]
            },
            "panoptic": {
                "status": "loaded" if panoptic_model is not None else "failed",
                "path": PANOPTIC_CKPT_PATH,
            },
            "maskrcnn2": {
                "status": "loaded" if maskrcnn2_model is not None else "failed",
                "path": MASKRCNN2_CKPT_PATH,
            },
        },
        "device": str(device),
        "working_directory": os.getcwd(),
        "available_files": os.listdir('.'),
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    um_per_px: float = Form(1.0),
    background_tasks: BackgroundTasks = None
):
    """
    Analyze an uploaded image for gland tortuosity
    
    Args:
        file: Image file (jpg, jpeg, png)
        
    Returns:
        JSON with analysis results including:
        - processed_image: Base64 encoded processed image
        - avg_tortuosity: Average tortuosity value
        - num_glands: Number of detected glands
        - individual_tortuosities: List of individual gland tortuosities
    """

    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Validate file extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = Path(file.filename or "").suffix.lower() or ".jpg"
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File extension {file_extension} not allowed. Use: {allowed_extensions}"
        )

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Check if models are loaded
        if maskrcnn_model is None or unet_model is None:
            model_status = {
                "maskrcnn": "loaded" if maskrcnn_model is not None else "failed",
                "unet": "loaded" if unet_model is not None else "failed"
            }
            raise HTTPException(
                status_code=503, 
                detail={
                    "error": "Service temporarily unavailable",
                    "message": "AI models are not currently loaded. Please try again later.",
                    "model_status": model_status,
                    "suggestion": "Contact administrator if the problem persists."
                }
            )
        
        # Perform analysis using pre-loaded models
        result_image, tortuosity_data = show_combined_result_with_models(
            temp_file_path,
            maskrcnn_model,
            unet_model,
            device,
            um_per_px=um_per_px
        )
        
        # Convert result image to base64
        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        # Convert binary mask to base64 (PNG) for frontend
        binary_mask = tortuosity_data.get('binary_mask_glands')
        binary_mask_base64 = None
        if binary_mask is not None:
            # Convert binary mask (0/1) to image (0/255) for PNG encoding
            mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8), mode='L')
            mask_buffer = io.BytesIO()
            mask_image.save(mask_buffer, format='PNG')
            mask_buffer.seek(0)
            binary_mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()

        # Clean up temporary file
        background_tasks.add_task(os.unlink, temp_file_path) if background_tasks else os.unlink(temp_file_path)

        # Return results
        individual_tortuosities = tortuosity_data['individual_tortuosities']
        individual_lengths = tortuosity_data.get('individual_lengths', [])
        individual_thicknesses = tortuosity_data.get('individual_thicknesses', [])
        # jsonable_encoder: convierte numpy / tipos no estándar en JSON válido (evita pérdida de campos anidados)
        return jsonable_encoder({
            "success": True,
            "message": "Analysis completed successfully",
            "data": {
                "processed_image": f"data:image/png;base64,{img_base64}",
                "avg_tortuosity": round(tortuosity_data['avg_tortuosity'], 3),
                "num_glands": tortuosity_data['num_glands'],
                "individual_tortuosities": [round(t, 3) for t in individual_tortuosities],
                "avg_length_px": round(tortuosity_data.get('avg_length_px', 0.0), 1),
                "avg_thickness_px": round(tortuosity_data.get('avg_thickness_px', 0.0), 1),
                "individual_lengths": [round(l, 1) for l in individual_lengths],
                "individual_thicknesses": [round(t, 1) for t in individual_thicknesses],
                "binary_mask_glands": f"data:image/png;base64,{binary_mask_base64}" if binary_mask_base64 else None,
                "analysis_info": {
                    "total_glands_analyzed": len(individual_tortuosities),
                    "tortuosity_range": {
                        "min": round(min(individual_tortuosities), 3) if individual_tortuosities else 0,
                        "max": round(max(individual_tortuosities), 3) if individual_tortuosities else 0
                    }
                },
                "um_per_px": tortuosity_data.get('um_per_px', 1.0),
                "avg_length_um": tortuosity_data.get('avg_length_um', 0.0),
                "avg_thickness_um": tortuosity_data.get('avg_thickness_um', 0.0),
                "avg_ICM": tortuosity_data.get('avg_ICM', 0.0),
                "avg_ITA_deg": tortuosity_data.get('avg_ITA_deg', 0.0),
                "avg_tortuosity_score": tortuosity_data.get('avg_tortuosity_score', 0.0),
                "dominant_grade": tortuosity_data.get('dominant_grade', "Normal"),
                "individual_glands": tortuosity_data.get('individual_glands', []),
            }
        })
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Model file not found: {str(e)}"
        )
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/analyze-maskrcnn2")
async def analyze_maskrcnn2(
    file: UploadFile = File(...),
    um_per_px: float = Form(1.0),
    contour_mask: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks = None
):
    """Mask R-CNN v2 — CLAHE LAB + EXIF + contour crop support."""
    if maskrcnn2_model is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "Mask R-CNN v2 not loaded", "message": f"Checkpoint '{MASKRCNN2_CKPT_PATH}' missing or failed."}
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = Path(file.filename or "").suffix.lower() or ".jpg"
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Extension {file_extension} not allowed")

    contour_tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name

        # Optional user-drawn contour mask
        if contour_mask is not None and contour_mask.filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as ctmp:
                shutil.copyfileobj(contour_mask.file, ctmp)
                contour_tmp_path = ctmp.name

        pred_instance = maskrcnn_v2.infer_with_model(
            image_path=temp_file_path,
            model=maskrcnn2_model,
            device=device,
            contour_path=contour_tmp_path,
        )

        # Build tarsus mask for tortuosity pipeline
        if contour_tmp_path is not None:
            H_orig, W_orig = np.array(Image.open(temp_file_path).convert("RGB")).shape[:2]
            contour_arr = np.array(
                Image.open(contour_tmp_path).convert("L").resize((W_orig, H_orig), Image.NEAREST)
            )
            tarsus_bin = (contour_arr > 0).astype(np.float32)
        else:
            tarsus_bin = None  # compute_results_from_instance_map will run UNet

        result_image, tortuosity_data = compute_results_from_instance_map(
            pred_instance, temp_file_path, unet_model, device,
            um_per_px=um_per_px,
            precomputed_tarsus=tarsus_bin,
        )

        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        binary_mask = tortuosity_data.get('binary_mask_glands')
        binary_mask_base64 = None
        if binary_mask is not None:
            mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8), mode='L')
            mask_buffer = io.BytesIO()
            mask_image.save(mask_buffer, format='PNG')
            mask_buffer.seek(0)
            binary_mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()

        if background_tasks:
            background_tasks.add_task(os.unlink, temp_file_path)
        else:
            os.unlink(temp_file_path)

        if contour_tmp_path:
            os.unlink(contour_tmp_path)

        individual_tortuosities = tortuosity_data['individual_tortuosities']
        individual_lengths = tortuosity_data.get('individual_lengths', [])
        individual_thicknesses = tortuosity_data.get('individual_thicknesses', [])

        return jsonable_encoder({
            "success": True,
            "message": "Mask R-CNN v2 analysis completed successfully",
            "data": {
                "processed_image": f"data:image/png;base64,{img_base64}",
                "avg_tortuosity": round(tortuosity_data['avg_tortuosity'], 3),
                "num_glands": tortuosity_data['num_glands'],
                "individual_tortuosities": [round(t, 3) for t in individual_tortuosities],
                "avg_length_px": round(tortuosity_data.get('avg_length_px', 0.0), 1),
                "avg_thickness_px": round(tortuosity_data.get('avg_thickness_px', 0.0), 1),
                "individual_lengths": [round(l, 1) for l in individual_lengths],
                "individual_thicknesses": [round(t, 1) for t in individual_thicknesses],
                "binary_mask_glands": f"data:image/png;base64,{binary_mask_base64}" if binary_mask_base64 else None,
                "analysis_info": {
                    "total_glands_analyzed": len(individual_tortuosities),
                    "tortuosity_range": {
                        "min": round(min(individual_tortuosities), 3) if individual_tortuosities else 0,
                        "max": round(max(individual_tortuosities), 3) if individual_tortuosities else 0,
                    },
                },
                "um_per_px": tortuosity_data.get('um_per_px', 1.0),
                "avg_length_um": tortuosity_data.get('avg_length_um', 0.0),
                "avg_thickness_um": tortuosity_data.get('avg_thickness_um', 0.0),
                "avg_ICM": tortuosity_data.get('avg_ICM', 0.0),
                "avg_ITA_deg": tortuosity_data.get('avg_ITA_deg', 0.0),
                "avg_tortuosity_score": tortuosity_data.get('avg_tortuosity_score', 0.0),
                "dominant_grade": tortuosity_data.get('dominant_grade', "Normal"),
                "individual_glands": tortuosity_data.get('individual_glands', []),
            }
        })

    except Exception as e:
        for p in [locals().get('temp_file_path'), contour_tmp_path]:
            if p and os.path.exists(p):
                os.unlink(p)
        import traceback
        tb = traceback.format_exc()
        print(f"[MASKRCNN2] ERROR: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Mask R-CNN v2 analysis failed: {str(e)}")


@app.post("/analyze-panoptic")
async def analyze_panoptic(
    file: UploadFile = File(...),
    um_per_px: float = Form(1.0),
    contour_mask: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks = None
):
    """
    Analyze an uploaded image using Mask2Former (panoptic) for gland segmentation.
    Same response format as /analyze.
    """
    if panoptic_model is None or panoptic_processor is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Panoptic model not loaded",
                "message": f"Checkpoint '{PANOPTIC_CKPT_PATH}' may be missing or failed to load.",
            }
        )
    if unet_model is None:
        raise HTTPException(status_code=503, detail="UNet (tarsus) model not loaded")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = Path(file.filename or "").suffix.lower() or ".jpg"
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Extension {file_extension} not allowed")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name

        img_np = np.array(Image.open(temp_file_path).convert("RGB"))
        H_orig, W_orig = img_np.shape[:2]

        if contour_mask is not None and contour_mask.filename:
            # User-drawn contour mask — use exactly like reference.py contour_path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as mask_tmp:
                shutil.copyfileobj(contour_mask.file, mask_tmp)
                mask_tmp_path = mask_tmp.name
            contour_arr = np.array(
                Image.open(mask_tmp_path).convert("L").resize((W_orig, H_orig), Image.NEAREST)
            )
            os.unlink(mask_tmp_path)
            tarsus_bin = (contour_arr > 0).astype(np.uint8)
        else:
            # Fall back to UNet tarsus prediction
            _, mask_tarsus = predict_unet_model(unet_model, temp_file_path, device, use_clahe=True, use_tta=True)
            mask_np = mask_tarsus.squeeze().numpy()
            mask_full = np.array(
                Image.fromarray((mask_np * 255).astype(np.uint8)).resize((W_orig, H_orig), Image.BILINEAR)
            ) / 255.0
            tarsus_bin = (mask_full > 0.5).astype(np.uint8)

        CROP_MARGIN = 20
        ys, xs = np.where(tarsus_bin > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1 = max(0, int(xs.min()) - CROP_MARGIN)
            x2 = min(W_orig, int(xs.max()) + 1 + CROP_MARGIN)
            y1 = max(0, int(ys.min()) - CROP_MARGIN)
            y2 = min(H_orig, int(ys.max()) + 1 + CROP_MARGIN)
        else:
            x1, y1, x2, y2 = 0, 0, W_orig, H_orig

        # Apply mask then crop — mirrors reference.py preprocess_image exactly:
        #   img_np = img_np * contour_np[..., None]
        #   img_np = img_np[y1:y2, x1:x2]
        img_masked = img_np * tarsus_bin[..., None]
        img_cropped = img_masked[y1:y2, x1:x2]

        pred_instance_crop, _ = predict_panoptic_in_memory(
            img_cropped, panoptic_model, panoptic_processor, device
        )

        # Place crop result back into full-image coordinates
        pred_instance = np.zeros((H_orig, W_orig), dtype=np.int32)
        h_c, w_c = pred_instance_crop.shape
        pred_instance[y1:y1+h_c, x1:x1+w_c] = pred_instance_crop

        result_image, tortuosity_data = compute_results_from_instance_map(
            pred_instance, temp_file_path, unet_model, device,
            um_per_px=um_per_px,
            precomputed_tarsus=tarsus_bin.astype(np.float32),
        )

        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        binary_mask = tortuosity_data.get('binary_mask_glands')
        binary_mask_base64 = None
        if binary_mask is not None:
            mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8), mode='L')
            mask_buffer = io.BytesIO()
            mask_image.save(mask_buffer, format='PNG')
            mask_buffer.seek(0)
            binary_mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()

        background_tasks.add_task(os.unlink, temp_file_path) if background_tasks else os.unlink(temp_file_path)

        individual_tortuosities = tortuosity_data['individual_tortuosities']
        individual_lengths = tortuosity_data.get('individual_lengths', [])
        individual_thicknesses = tortuosity_data.get('individual_thicknesses', [])

        return jsonable_encoder({
            "success": True,
            "message": "Panoptic analysis completed successfully",
            "data": {
                "processed_image": f"data:image/png;base64,{img_base64}",
                "avg_tortuosity": round(tortuosity_data['avg_tortuosity'], 3),
                "num_glands": tortuosity_data['num_glands'],
                "individual_tortuosities": [round(t, 3) for t in individual_tortuosities],
                "avg_length_px": round(tortuosity_data.get('avg_length_px', 0.0), 1),
                "avg_thickness_px": round(tortuosity_data.get('avg_thickness_px', 0.0), 1),
                "individual_lengths": [round(l, 1) for l in individual_lengths],
                "individual_thicknesses": [round(t, 1) for t in individual_thicknesses],
                "binary_mask_glands": f"data:image/png;base64,{binary_mask_base64}" if binary_mask_base64 else None,
                "analysis_info": {
                    "total_glands_analyzed": len(individual_tortuosities),
                    "tortuosity_range": {
                        "min": round(min(individual_tortuosities), 3) if individual_tortuosities else 0,
                        "max": round(max(individual_tortuosities), 3) if individual_tortuosities else 0,
                    },
                },
                "um_per_px": tortuosity_data.get('um_per_px', 1.0),
                "avg_length_um": tortuosity_data.get('avg_length_um', 0.0),
                "avg_thickness_um": tortuosity_data.get('avg_thickness_um', 0.0),
                "avg_ICM": tortuosity_data.get('avg_ICM', 0.0),
                "avg_ITA_deg": tortuosity_data.get('avg_ITA_deg', 0.0),
                "avg_tortuosity_score": tortuosity_data.get('avg_tortuosity_score', 0.0),
                "dominant_grade": tortuosity_data.get('dominant_grade', "Normal"),
                "individual_glands": tortuosity_data.get('individual_glands', []),
            }
        })

    except Exception as e:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        import traceback
        tb = traceback.format_exc()
        print(f"[PANOPTIC] ERROR: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Panoptic analysis failed: {str(e)}\n{tb[-2000:]}")


@app.post("/apply-clahe")
async def apply_clahe_filter(
    file: UploadFile = File(...),
    block_radius: int = 63,
    bins: int = 255,
    slope: float = 3.0,
    convert_to_gray: bool = True
):
    """
    Apply CLAHE filter to an uploaded image
    
    Args:
        file: Image file (jpg, jpeg, png)
        block_radius: Block radius for CLAHE (default: 63)
        bins: Number of bins for histogram (default: 255)
        slope: Slope for clip limit (default: 3.0)
        
    Returns:
        JSON with processed image as base64
    """
    
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Validate file extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = Path(file.filename or "").suffix.lower() or ".jpg"
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File extension {file_extension} not allowed. Use: {allowed_extensions}"
        )

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Load image and convert to numpy array
        image = Image.open(temp_file_path).convert("RGB")
        img_array = np.array(image)
        
        # Apply CLAHE filter
        clahe_img = clahe_like_imagej(img_array, block_radius, bins, slope, convert_to_gray)
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(clahe_img)
        
        # Convert result image to base64
        img_buffer = io.BytesIO()
        processed_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Return results
        return {
            "success": True,
            "message": "CLAHE filter applied successfully",
            "data": {
                "processed_image": f"data:image/png;base64,{img_base64}",
                "parameters": {
                    "block_radius": block_radius,
                    "bins": bins,
                    "slope": slope,
                    "convert_to_gray": convert_to_gray
                }
            }
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(
            status_code=500, 
            detail=f"CLAHE processing failed: {str(e)}"
        )

@app.get("/debug/force-load-models")
async def force_load_models():
    """Debug endpoint to force model loading and get detailed error info"""
    global maskrcnn_model, unet_model
    debug_info = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "models": {},
        "system_info": {},
        "errors": []
    }
    
    try:
        # System info
        import platform, psutil
        memory = psutil.virtual_memory()
        debug_info["system_info"] = {
            "python_version": platform.python_version(),
            "pytorch_version": device.__module__.split('.')[0] if hasattr(device, '__module__') else "unknown",
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_percent": memory.percent,
            "device": str(device)
        }
        
        # Try loading Mask R-CNN
        debug_info["models"]["maskrcnn"] = {"status": "attempting"}
        try:
            import torch
            debug_info["system_info"]["pytorch_version"] = torch.__version__
            
            for i, path in enumerate(FALLBACK_MASK_RCNN_PATHS):
                if os.path.exists(path):
                    size_mb = round(os.path.getsize(path) / (1024**2), 1)
                    debug_info["models"]["maskrcnn"][f"attempt_{i+1}"] = {
                        "path": path,
                        "size_mb": size_mb,
                        "status": "trying"
                    }
                    try:
                        model = load_maskrcnn_model(path, device)
                        maskrcnn_model = model
                        debug_info["models"]["maskrcnn"][f"attempt_{i+1}"]["status"] = "SUCCESS"
                        debug_info["models"]["maskrcnn"]["status"] = "loaded"
                        break
                    except Exception as e:
                        debug_info["models"]["maskrcnn"][f"attempt_{i+1}"]["status"] = f"FAILED: {str(e)}"
                        debug_info["errors"].append(f"Mask R-CNN {path}: {str(e)}")
            else:
                debug_info["models"]["maskrcnn"]["status"] = "failed_all_paths"
        except Exception as e:
            debug_info["models"]["maskrcnn"]["status"] = f"critical_error: {str(e)}"
            debug_info["errors"].append(f"Mask R-CNN critical: {str(e)}")
        
        # Try loading UNet
        debug_info["models"]["unet"] = {"status": "attempting"}
        try:
            for i, path in enumerate(FALLBACK_UNET_PATHS):
                if os.path.exists(path):
                    size_mb = round(os.path.getsize(path) / (1024**2), 1)
                    debug_info["models"]["unet"][f"attempt_{i+1}"] = {
                        "path": path,
                        "size_mb": size_mb,
                        "status": "trying"
                    }
                    try:
                        model = load_unet_model(path, device)
                        unet_model = model
                        debug_info["models"]["unet"][f"attempt_{i+1}"]["status"] = "SUCCESS"
                        debug_info["models"]["unet"]["status"] = "loaded"
                        break
                    except Exception as e:
                        debug_info["models"]["unet"][f"attempt_{i+1}"]["status"] = f"FAILED: {str(e)}"
                        debug_info["errors"].append(f"UNet {path}: {str(e)}")
            else:
                debug_info["models"]["unet"]["status"] = "failed_all_paths"
        except Exception as e:
            debug_info["models"]["unet"]["status"] = f"critical_error: {str(e)}"
            debug_info["errors"].append(f"UNet critical: {str(e)}")
            
    except Exception as e:
        debug_info["critical_error"] = str(e)
        debug_info["errors"].append(f"Critical system error: {str(e)}")
    
    return debug_info

@app.get("/info")
async def get_analysis_info():
    """Get information about the tortuosity analysis"""
    return {
        "description": "Análisis de Tortuosidad de Glándulas de Meibomio",
        "methodology": {
            "tortuosity_formula": "Tortuosidad = (Perímetro / (2 × Altura del rectángulo mínimo externo)) - 1",
            "interpretation": {
                "low": "0.0 - 0.1: Tortuosidad baja (generalmente normal)",
                "moderate": "0.1 - 0.2: Tortuosidad moderada (puede indicar cambios iniciales)",
                "high": "> 0.2: Tortuosidad alta (sugestivo de MGD, requiere correlación clínica)"
            }
        },
        "models_used": {
            "mask_rcnn": "Detección y segmentación de glándulas individuales",
            "unet": "Segmentación del contorno del párpado (Tarsus)"
        },
        "note": "Los rangos de interpretación son aproximados y la interpretación final debe ser realizada por un especialista."
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 