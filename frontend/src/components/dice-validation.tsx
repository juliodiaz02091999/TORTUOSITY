"use client";

import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { UploadZone } from "@/components/upload-zone";
import {
  Paintbrush,
  Eraser,
  RotateCcw,
  Sparkles,
  Calculator,
  AlertCircle,
  Loader2,
  Upload,
  Eye,
  EyeOff,
  FileImage
} from "lucide-react";

interface DiceValidationProps {
  onTabChange?: (tab: 'upload' | 'capture' | 'results' | 'info' | 'dice') => void;
}

interface SegmentationMetrics {
  dice: number;
  f1: number;
  iou: number;
  accuracy: number;
  recall: number;
  precision: number;
}

function calculateMetrics(groundTruth: Uint8ClampedArray, prediction: Uint8ClampedArray): SegmentationMetrics {
  if (groundTruth.length !== prediction.length) {
    throw new Error("Masks must have the same size");
  }

  let tp = 0, fp = 0, fn = 0, tn = 0;
  const totalPixels = groundTruth.length / 4;

  for (let i = 0; i < groundTruth.length; i += 4) {
    const gtPixel = groundTruth[i + 1] > 127 ? 1 : 0;
    const predGray = Math.max(prediction[i], prediction[i + 1], prediction[i + 2]);
    const predPixel = predGray > 127 ? 1 : 0;

    if (gtPixel === 1 && predPixel === 1) tp++;
    else if (gtPixel === 0 && predPixel === 1) fp++;
    else if (gtPixel === 1 && predPixel === 0) fn++;
    else tn++;
  }

  const dice      = (2 * tp + fp + fn) === 0 ? 0 : (2 * tp) / (2 * tp + fp + fn);
  const iou       = (tp + fp + fn) === 0 ? 0 : tp / (tp + fp + fn);
  const accuracy  = totalPixels === 0 ? 0 : (tp + tn) / totalPixels;
  const recall    = (tp + fn) === 0 ? 0 : tp / (tp + fn);
  const precision = (tp + fp) === 0 ? 0 : tp / (tp + fp);

  return { dice, f1: dice, iou, accuracy, recall, precision };
}

export function DiceValidation({ onTabChange }: DiceValidationProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  const [claheImageUrl, setClaheImageUrl] = useState<string | null>(null);
  const [showClahe, setShowClahe] = useState(true);
  const [brushSize, setBrushSize] = useState(10);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawMode, setDrawMode] = useState<"brush" | "eraser">("brush");
  const [isApplyingClahe, setIsApplyingClahe] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<SegmentationMetrics | null>(null);
  const [predictedMaskUrl, setPredictedMaskUrl] = useState<string | null>(null);
  const [groundTruthMaskUrl, setGroundTruthMaskUrl] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const cachedImageRef = useRef<HTMLImageElement | null>(null);
  const cachedClaheImageRef = useRef<HTMLImageElement | null>(null);
  const redrawRequestRef = useRef<number | null>(null);

  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setOriginalImageUrl(url);

      // Cleanup only when selectedFile changes or component unmounts
      return () => {
        URL.revokeObjectURL(url);
        setOriginalImageUrl(null);
      };
    }
  }, [selectedFile]);

  // Load and cache original image
  useEffect(() => {
    if (originalImageUrl && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const img = new Image();
      img.onload = () => {
        cachedImageRef.current = img; // Cache the image
        canvas.width = img.width;
        canvas.height = img.height;

        // Initialize mask canvas
        if (maskCanvasRef.current) {
          maskCanvasRef.current.width = img.width;
          maskCanvasRef.current.height = img.height;
        }

        redrawCanvas();
      };
      img.src = originalImageUrl;
    }
  }, [originalImageUrl]);

  // Load and cache CLAHE image
  useEffect(() => {
    if (claheImageUrl) {
      const img = new Image();
      img.onload = () => {
        cachedClaheImageRef.current = img; // Cache the CLAHE image
        redrawCanvas();
      };
      img.src = claheImageUrl;
    }
  }, [claheImageUrl]);

  // Redraw when showClahe changes
  useEffect(() => {
    redrawCanvas();
  }, [showClahe]);

  const loadExampleImage = async () => {
    try {
      const response = await fetch('/meibomio.jpg');
      const blob = await response.blob();
      const file = new File([blob], 'meibomio.jpg', { type: 'image/jpeg' });
      setSelectedFile(file);
      setError(null);
    } catch (error) {
      console.error('Error loading example image:', error);
      setError('No se pudo cargar la imagen de ejemplo');
    }
  };

  const redrawCanvas = () => {
    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    if (!canvas || !maskCanvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw base image (CLAHE if enabled and available, otherwise original)
    if (showClahe && cachedClaheImageRef.current) {
      // Show CLAHE image as the base (complete, not overlay)
      ctx.drawImage(cachedClaheImageRef.current, 0, 0);
    } else if (cachedImageRef.current) {
      // Show original image if CLAHE is not enabled or not available
      ctx.drawImage(cachedImageRef.current, 0, 0);
    }

    // Draw mask on top
    ctx.drawImage(maskCanvas, 0, 0);
  };

  const applyClaheFilter = async () => {
    if (!selectedFile) return;

    setIsApplyingClahe(true);
    setError(null);

    try {
      // Usar la implementación optimizada del frontend (igual que en Análisis de Imagen)
      const { applyCLAHEToImage, fileToImageData } = await import("../lib/clahe-optimized");
      
      // Si ya hay una imagen CLAHE procesada, usarla como base (permite aplicar CLAHE múltiples veces)
      // Si no, usar la imagen original
      let imageData: ImageData;
      if (claheImageUrl && cachedClaheImageRef.current) {
        // Convertir la imagen CLAHE actual directamente a ImageData
        const canvas = document.createElement("canvas");
        canvas.width = cachedClaheImageRef.current.width;
        canvas.height = cachedClaheImageRef.current.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("No se pudo obtener el contexto del canvas");
        ctx.drawImage(cachedClaheImageRef.current, 0, 0);
        const imageDataFromCache = ctx.getImageData(0, 0, canvas.width, canvas.height);
        imageData = imageDataFromCache;
      } else {
        // Convertir archivo original a ImageData
        imageData = await fileToImageData(selectedFile);
      }
      
      // Aplicar CLAHE con parámetros optimizados (exactamente igual que en Análisis de Imagen)
      const processedImageData = await applyCLAHEToImage(imageData, {
        blockSize: 64,
        bins: 256,
        slope: 3,
        chunkSize: 64,
        workerCount: 2,
        onProgress: (progress, status) => {
          console.log(`CLAHE Progress: ${progress.toFixed(1)}% - ${status}`);
        }
      });

      // Crear data URL para mostrar la imagen procesada
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      canvas.width = processedImageData.width;
      canvas.height = processedImageData.height;
      ctx?.putImageData(processedImageData, 0, 0);
      const dataUrl = canvas.toDataURL("image/png");

      setClaheImageUrl(dataUrl);
      setShowClahe(true);
      redrawCanvas();
      
      console.log('CLAHE filter applied successfully using frontend implementation');
    } catch (err) {
      console.error('Error applying CLAHE filter:', err);
      setError(err instanceof Error ? err.message : "Error aplicando filtro CLAHE");
    } finally {
      setIsApplyingClahe(false);
    }
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    draw(e);
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    draw(e);
  };

  const handleCanvasMouseUp = () => {
    setIsDrawing(false);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    if (!canvas || !maskCanvas) return;

    const rect = canvas.getBoundingClientRect();
    // Calculate scale factor between visual size and internal canvas size
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    // Scale mouse coordinates to match canvas internal coordinates
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    const maskCtx = maskCanvas.getContext("2d");
    if (!maskCtx) return;

    maskCtx.globalCompositeOperation = drawMode === "brush" ? "source-over" : "destination-out";
    maskCtx.fillStyle = "rgba(0, 255, 0, 0.5)"; // Green for glands
    // Scale brush size to match canvas scale
    const scale = (scaleX + scaleY) / 2;
    const scaledBrushSize = brushSize * scale;
    maskCtx.beginPath();
    maskCtx.arc(x, y, scaledBrushSize, 0, 2 * Math.PI);
    maskCtx.fill();

    // Use requestAnimationFrame for smoother rendering
    if (redrawRequestRef.current !== null) {
      cancelAnimationFrame(redrawRequestRef.current);
    }
    redrawRequestRef.current = requestAnimationFrame(() => {
      redrawCanvas();
      redrawRequestRef.current = null;
    });
  };

  const clearMask = () => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;

    const maskCtx = maskCanvas.getContext("2d");
    if (!maskCtx) return;

    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    setGroundTruthMaskUrl(null);
    setMetrics(null);
    redrawCanvas();
  };

  const analyzeAndCalculateDice = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);
    setMetrics(null);
    setPredictedMaskUrl(null);
    setGroundTruthMaskUrl(null);

    try {
      // Get ground truth mask from canvas first
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas) {
        throw new Error("No se ha dibujado la máscara ground truth");
      }

      // Convert mask canvas to image URL for display
      const groundTruthMaskDataUrl = maskCanvas.toDataURL("image/png");
      setGroundTruthMaskUrl(groundTruthMaskDataUrl);

      const maskCtx = maskCanvas.getContext("2d");
      if (!maskCtx) {
        throw new Error("No se pudo obtener el contexto del canvas de máscara");
      }

      const groundTruthData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);

      // Send image to backend for analysis (CLAHE if active, otherwise original)
      let fileToSend: File;
      if (showClahe && claheImageUrl && cachedClaheImageRef.current) {
        // Convert CLAHE image to File
        const { imageDataToFile } = await import("../lib/clahe-optimized");
        const canvas = document.createElement("canvas");
        canvas.width = cachedClaheImageRef.current.width;
        canvas.height = cachedClaheImageRef.current.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("No se pudo obtener el contexto del canvas");
        ctx.drawImage(cachedClaheImageRef.current, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        fileToSend = await imageDataToFile(imageData, "clahe_processed.png");
      } else {
        // Use original image
        fileToSend = selectedFile;
      }

      const formData = new FormData();
      formData.append("file", fileToSend);

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error del servidor: ${response.status}`);
      }

      const result = await response.json();
      const predictedMaskDataUrl = result.data.binary_mask_glands;

      if (!predictedMaskDataUrl) {
        throw new Error("El backend no devolvió la máscara binaria");
      }

      setPredictedMaskUrl(predictedMaskDataUrl);

      // Load predicted mask
      const predImg = new Image();
      await new Promise((resolve, reject) => {
        predImg.onload = resolve;
        predImg.onerror = reject;
        predImg.src = predictedMaskDataUrl;
      });

      const predCanvas = document.createElement("canvas");
      predCanvas.width = predImg.width;
      predCanvas.height = predImg.height;
      const predCtx = predCanvas.getContext("2d");
      if (!predCtx) {
        throw new Error("No se pudo obtener el contexto del canvas de predicción");
      }

      predCtx.drawImage(predImg, 0, 0);
      const predictedData = predCtx.getImageData(0, 0, predCanvas.width, predCanvas.height);

      // Ensure masks have the same dimensions
      if (groundTruthData.width !== predictedData.width || groundTruthData.height !== predictedData.height) {
        // Resize ground truth to match predicted mask dimensions
        const resizedCanvas = document.createElement("canvas");
        resizedCanvas.width = predictedData.width;
        resizedCanvas.height = predictedData.height;
        const resizedCtx = resizedCanvas.getContext("2d");
        if (!resizedCtx) {
          throw new Error("No se pudo redimensionar la máscara ground truth");
        }
        resizedCtx.drawImage(maskCanvas, 0, 0, predictedData.width, predictedData.height);
        const resizedGroundTruthData = resizedCtx.getImageData(0, 0, resizedCanvas.width, resizedCanvas.height);
        
        const m = calculateMetrics(resizedGroundTruthData.data, predictedData.data);
        setMetrics(m);
      } else {
        const m = calculateMetrics(groundTruthData.data, predictedData.data);
        setMetrics(m);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error en el análisis");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Section */}
      {!selectedFile && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Cargar Meibografía
            </CardTitle>
          </CardHeader>
          <CardContent>
            <UploadZone onFileSelect={setSelectedFile} selectedFile={selectedFile} />

            {/* Example Image Button */}
            {!selectedFile && (
              <div className="mt-4 text-center">
                <Button
                  variant="outline"
                  onClick={loadExampleImage}
                  className="w-full border-border"
                >
                  <FileImage className="mr-2 h-4 w-4" />
                  Cargar Imagen de Ejemplo
                </Button>
                <p className="text-xs text-muted-foreground mt-2">
                  Prueba la validación Dice con una imagen de glándulas de Meibomio
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Drawing and Analysis Section */}
      {selectedFile && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Paintbrush className="h-5 w-5" />
                Dibujar Máscara Ground Truth
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Toolbar */}
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  onClick={() => setDrawMode("brush")}
                  variant={drawMode === "brush" ? "default" : "outline"}
                  size="sm"
                >
                  <Paintbrush className="mr-2 h-4 w-4" />
                  Pincel
                </Button>
                <Button
                  onClick={() => setDrawMode("eraser")}
                  variant={drawMode === "eraser" ? "default" : "outline"}
                  size="sm"
                >
                  <Eraser className="mr-2 h-4 w-4" />
                  Borrador
                </Button>
                <Button onClick={clearMask} variant="outline" size="sm">
                  <RotateCcw className="mr-2 h-4 w-4" />
                  Limpiar
                </Button>
                <Button
                  onClick={applyClaheFilter}
                  variant="outline"
                  size="sm"
                  disabled={isApplyingClahe}
                >
                  {isApplyingClahe ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Sparkles className="mr-2 h-4 w-4" />
                  )}
                  Aplicar CLAHE
                </Button>
                <Button
                  onClick={() => setShowClahe(!showClahe)}
                  variant="outline"
                  size="sm"
                  disabled={!claheImageUrl}
                >
                  {showClahe ? (
                    <EyeOff className="mr-2 h-4 w-4" />
                  ) : (
                    <Eye className="mr-2 h-4 w-4" />
                  )}
                  {showClahe ? "Ocultar" : "Mostrar"} CLAHE
                </Button>
              </div>

              {/* Brush Size Slider */}
              <div className="space-y-2">
                <Label>Tamaño del pincel: {brushSize}px</Label>
                <Slider
                  value={[brushSize]}
                  onValueChange={(value) => setBrushSize(value[0])}
                  min={1}
                  max={50}
                  step={1}
                  className="w-full"
                />
              </div>

              {/* Canvas */}
              <div className="border rounded-lg overflow-hidden">
                <canvas
                  ref={canvasRef}
                  onMouseDown={handleCanvasMouseDown}
                  onMouseMove={handleCanvasMouseMove}
                  onMouseUp={handleCanvasMouseUp}
                  onMouseLeave={handleCanvasMouseUp}
                  className="w-full cursor-crosshair"
                  style={{ maxWidth: "100%", height: "auto" }}
                />
                <canvas ref={maskCanvasRef} style={{ display: "none" }} />
              </div>

              {/* Analyze Button */}
              <Button
                onClick={analyzeAndCalculateDice}
                className="w-full"
                disabled={isAnalyzing}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Calculando métricas...
                  </>
                ) : (
                  <>
                    <Calculator className="mr-2 h-4 w-4" />
                    Calcular Métricas de Segmentación
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Results Section */}
          {metrics !== null && (
            <Card>
              <CardHeader>
                <CardTitle>Métricas de Segmentación</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Metric cards */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {[
                    { label: "Dice", value: metrics.dice, desc: "Coeficiente de similitud" },
                    { label: "F1 Score", value: metrics.f1, desc: "= Dice (segm. binaria)" },
                    { label: "IoU", value: metrics.iou, desc: "Jaccard Index" },
                    { label: "Accuracy", value: metrics.accuracy, desc: "Exactitud global" },
                    { label: "Recall", value: metrics.recall, desc: "Sensibilidad (TP rate)" },
                    { label: "Precision", value: metrics.precision, desc: "Valor pred. positivo" },
                  ].map(({ label, value, desc }) => (
                    <div key={label} className="p-3 rounded-lg bg-muted text-center">
                      <div className="text-xs text-muted-foreground mb-1">{label}</div>
                      <div className="text-xl font-bold">{value.toFixed(4)}</div>
                      <div className="text-xs text-muted-foreground mt-1">{desc}</div>
                    </div>
                  ))}
                </div>

                {/* Mask images */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                  {groundTruthMaskUrl && (
                    <div>
                      <h4 className="font-semibold mb-2">Máscara Ground Truth</h4>
                      <div className="border rounded-lg overflow-hidden">
                        <img src={groundTruthMaskUrl} alt="Máscara ground truth" className="w-full" />
                      </div>
                    </div>
                  )}
                  {predictedMaskUrl && (
                    <div>
                      <h4 className="font-semibold mb-2">Máscara Predicha</h4>
                      <div className="border rounded-lg overflow-hidden">
                        <img src={predictedMaskUrl} alt="Máscara predicha" className="w-full" />
                      </div>
                    </div>
                  )}
                </div>

                {/* Interpretation guide */}
                <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-950">
                  <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                    Interpretación
                  </h4>
                  <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                    <li>• <strong>Dice/F1/IoU = 1.0:</strong> Segmentación perfecta</li>
                    <li>• <strong>Dice &gt; 0.7 / IoU &gt; 0.5:</strong> Buena segmentación</li>
                    <li>• <strong>Recall alto:</strong> pocas glándulas perdidas (FN bajos)</li>
                    <li>• <strong>Precision alta:</strong> pocas falsas detecciones (FP bajos)</li>
                    <li>• <strong>Accuracy:</strong> puede ser engañosa si el fondo domina la imagen</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Error */}
          {error && (
            <Alert className="border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950">
              <AlertCircle className="h-4 w-4 text-red-600" />
              <AlertDescription className="text-red-800 dark:text-red-200">
                {error}
              </AlertDescription>
            </Alert>
          )}
        </>
      )}
    </div>
  );
}
