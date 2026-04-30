"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { UploadZone } from "@/components/upload-zone";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { ResultsDisplay } from "@/components/results-display";
import { PhotoManager } from "@/components/photo-manager";
import { DiceValidation } from "@/components/dice-validation";
import { StoredPhoto } from "@/lib/photo-storage";
import {
  Eye,
  Brain,
  Zap,
  AlertCircle,
  Loader2,
  BarChart3,
  Upload,
  Settings,
  Home,
  FileText,
  Sparkles,
  FileImage,
  Camera,
  Menu,
  X,
  Calculator
} from "lucide-react";

interface GlandResult {
  gland_id: string;
  length_px: number;
  length_um: number;
  thickness_px: number;
  thickness_um: number;
  aspect_ratio: number;
  area_um2: number;
  ICM: number;
  ITA_deg: number;
  DCF: number;
  IMCC: number;
  tortuosity_score: number;
  tortuosity_grade: string;
}

interface AnalysisResult {
  success: boolean;
  message: string;
  data: {
    processed_image: string;
    avg_tortuosity: number;
    num_glands: number;
    individual_tortuosities: number[];
    avg_length_px: number;
    avg_thickness_px: number;
    individual_lengths: number[];
    individual_thicknesses: number[];
    analysis_info: {
      total_glands_analyzed: number;
      tortuosity_range: {
        min: number;
        max: number;
      };
    };
    um_per_px?: number;
    avg_length_um?: number;
    avg_thickness_um?: number;
    avg_ICM?: number;
    avg_ITA_deg?: number;
    avg_tortuosity_score?: number;
    dominant_grade?: string;
    individual_glands?: GlandResult[];
  };
}

interface MgdaResult {
  success: boolean;
  message: string;
  data: {
    processed_image: string;
    metrics: {
      mg_ratio: number;
      dysfunction_percentage: number;
      dysfunction_area: number;
    };
    expansion_mode: string;
    used_expansion_mode: "inferior" | "superior";
  };
}

export default function DashboardPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [mgdaResult, setMgdaResult] = useState<MgdaResult | null>(null);
  const [isAnalyzingMgda, setIsAnalyzingMgda] = useState(false);
  const [expansionMode, setExpansionMode] = useState<"inferior" | "superior">("inferior");

  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [activeTab, setActiveTab] = useState<'upload' | 'capture' | 'results' | 'info' | 'dice'>('upload');
  const [claheImage, setClaheImage] = useState<string | null>(null);
  const [isApplyingClahe, setIsApplyingClahe] = useState(false);
  const [convertToGray, setConvertToGray] = useState(true);
  const [selectedCapturedPhoto, setSelectedCapturedPhoto] = useState<StoredPhoto | null>(null);
  const [umPerPx, setUmPerPx] = useState<number>(1.0);
  const [segmentationModel, setSegmentationModel] = useState<"maskrcnn" | "panoptic">("maskrcnn");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  const EQUIPMENT_PRESETS = [
    { label: "Sin calibración (píxeles)", value: 1.0 },
    { label: "Keratograph 5M — 10x (8.0 µm/px)", value: 8.0 },
    { label: "Keratograph 5M — 16x (5.0 µm/px)", value: 5.0 },
    { label: "LipiView II (6.5 µm/px)", value: 6.5 },
    { label: "Lámpara de hendidura 40x (11.76 µm/px)", value: 11.76 },
  ];

  // Check if device is mobile on mount and resize
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
      if (window.innerWidth >= 768) {
        setSidebarOpen(false);
      }
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Close sidebar when clicking outside on mobile
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (isMobile && sidebarOpen) {
        const sidebar = document.getElementById('sidebar');
        const hamburger = document.getElementById('hamburger');
        if (sidebar && !sidebar.contains(event.target as Node) && 
            hamburger && !hamburger.contains(event.target as Node)) {
          setSidebarOpen(false);
        }
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isMobile, sidebarOpen]);

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
    setError(null);
    setResults(null);
    setMgdaResult(null);
    setClaheImage(null);
  };

  const handleCapturedPhotoSelect = (photo: StoredPhoto) => {
    setSelectedCapturedPhoto(photo);
    setSelectedFile(null); // Clear any uploaded file
    setError(null);
    setResults(null);
    setMgdaResult(null);
    setClaheImage(null);
  };

  const handleConvertToGrayChange = (checked: boolean) => {
    setConvertToGray(checked);
    setClaheImage(null); // Limpiar imagen anterior cuando se cambia la opción
  };

  const loadExampleImage = async () => {
    try {
      // Load the example image from the frontend public folder
      const response = await fetch('/meibomio.jpg');
      const blob = await response.blob();
      const file = new File([blob], 'meibomio.jpg', { type: 'image/jpeg' });
      handleFileSelect(file);
    } catch (error) {
      console.error('Error loading example image:', error);
      setError('No se pudo cargar la imagen de ejemplo');
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile && !selectedCapturedPhoto) return;

    setIsAnalyzing(true);
    setError(null);
    setProgress(0);

    // Simular progreso
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 500);

    try {
      let fileToAnalyze: File;

      if (selectedFile) {
        fileToAnalyze = selectedFile;
      } else if (selectedCapturedPhoto) {
        // Convert data URL to File object
        const response = await fetch(selectedCapturedPhoto.dataUrl);
        const blob = await response.blob();
        fileToAnalyze = new File([blob], selectedCapturedPhoto.fileName || 'captured-image.jpg', { type: 'image/jpeg' });
      } else {
        throw new Error('No hay imagen seleccionada para analizar');
      }

      const formData = new FormData();
      formData.append("file", fileToAnalyze);
      formData.append("um_per_px", umPerPx.toString());

      const endpoint = segmentationModel === "panoptic" ? "/api/analyze-panoptic" : "/api/analyze";
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        
        if (contentType && contentType.includes("application/json")) {
          try {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Error en el análisis");
          } catch {
            throw new Error(`Error del servidor: ${response.status}`);
          }
        } else {
          try {
            const errorText = await response.text();
            throw new Error(`Error del servidor: ${response.status} - ${errorText.substring(0, 100)}`);
          } catch {
            throw new Error(`Error del servidor: ${response.status}`);
          }
        }
      }

      const result: AnalysisResult = await response.json();

      // Save analysis results to local storage if this was a captured photo
      if (selectedCapturedPhoto) {
        try {
          // Update the photo in local storage with analysis results
          const analysisResults = {
            avgTortuosity: result.data.avg_tortuosity,
            numGlands: result.data.num_glands,
            individualTortuosities: result.data.individual_tortuosities
          };

          // Note: The photoStorage.updatePhotoAnalysis would need to be implemented
          // For now, we'll just log the results
          console.log('Analysis results for captured photo:', {
            photoId: selectedCapturedPhoto.id,
            results: analysisResults
          });
        } catch (error) {
          console.warn('Could not save analysis results to local storage:', error);
        }
      }

      setResults(result);
      setProgress(100);
      setActiveTab('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setIsAnalyzing(false);
      clearInterval(progressInterval);
    }
  };

  const analyzeMgda = async () => {
    if (!selectedFile && !selectedCapturedPhoto) return;

    setIsAnalyzingMgda(true);
    setError(null);

    try {
      let fileToAnalyze: File;

      if (selectedFile) {
        fileToAnalyze = selectedFile;
      } else if (selectedCapturedPhoto) {
        const response = await fetch(selectedCapturedPhoto.dataUrl);
        const blob = await response.blob();
        fileToAnalyze = new File([blob], selectedCapturedPhoto.fileName || 'captured-image.jpg', { type: 'image/jpeg' });
      } else {
        throw new Error('No hay imagen seleccionada para analizar');
      }

      const formData = new FormData();
      formData.append("file", fileToAnalyze);
      formData.append("expansion_mode", expansionMode);


      const response = await fetch("/api/analyze-mgda", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
          try {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Error en el análisis MGDA");
          } catch {
            throw new Error(`Error del servidor: ${response.status}`);
          }
        } else {
          const errorText = await response.text();
          throw new Error(`Error del servidor: ${response.status} - ${errorText.substring(0, 100)}`);
        }
      }

      const result: MgdaResult = await response.json();
      setMgdaResult(result);
      setResults(null); // Clear tortuosity results if any
      setActiveTab('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setIsAnalyzingMgda(false);
    }
  };

  const applyClaheFilter = async () => {
    if (!selectedFile && !selectedCapturedPhoto) return;

    setIsApplyingClahe(true);
    setError(null);

    try {
      let fileToProcess: File;

      if (selectedFile) {
        fileToProcess = selectedFile;
      } else if (selectedCapturedPhoto) {
        // Convert data URL to File object
        const response = await fetch(selectedCapturedPhoto.dataUrl);
        const blob = await response.blob();
        fileToProcess = new File([blob], selectedCapturedPhoto.fileName || 'captured-image.jpg', { type: 'image/jpeg' });
      } else {
        throw new Error('No hay imagen seleccionada para procesar');
      }

      // Usar la implementación optimizada del frontend
      const { applyCLAHEToImage, fileToImageData, imageDataToFile } = await import('../lib/clahe-optimized');
      
      // Convertir archivo a ImageData
      const imageData = await fileToImageData(fileToProcess);
      
      // Aplicar CLAHE con parámetros optimizados
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

      // Convertir resultado a File
      const baseName = fileToProcess.name.replace(/\.[^/.]+$/, "");
      const processedFile = await imageDataToFile(processedImageData, `clahe_${baseName}.png`);
      
      // Crear data URL para mostrar la imagen procesada
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = processedImageData.width;
      canvas.height = processedImageData.height;
      ctx?.putImageData(processedImageData, 0, 0);
      const dataUrl = canvas.toDataURL('image/png');
      
      setClaheImage(dataUrl);

      // Reemplazar el archivo seleccionado con la imagen procesada
          setSelectedFile(processedFile);
      if (selectedCapturedPhoto) {
          setSelectedCapturedPhoto(null); // Clear captured photo since we're now using the processed file
        }

        // Clear previous results if any
        setResults(null);
      
      console.log('CLAHE filter applied successfully using frontend implementation');
    } catch (err) {
      console.error('Error applying CLAHE filter:', err);
      setError(err instanceof Error ? err.message : "Error aplicando filtro CLAHE");
    } finally {
      setIsApplyingClahe(false);
    }
  };

  const SidebarItem = ({ icon: Icon, label, active, onClick }: {
    icon: React.ComponentType<{ className?: string }>;
    label: string;
    active: boolean;
    onClick: () => void;
  }) => (
    <button
      onClick={() => {
        onClick();
        if (isMobile) {
          setSidebarOpen(false);
        }
      }}
      className={`flex items-center gap-3 w-full p-3 rounded-lg transition-all ${
        active 
          ? 'bg-primary text-primary-foreground shadow-sm' 
          : 'text-muted-foreground hover:text-white hover:bg-muted'
      }`}
    >
      <Icon className="h-5 w-5" />
      <span className="font-medium">{label}</span>
    </button>
  );

  return (
    <div className="min-h-screen bg-background flex w-full">
      {/* Mobile Overlay */}
      {isMobile && sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden transition-opacity duration-100 ease-out"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div 
        id="sidebar"
        className={`fixed left-0 top-0 h-full z-50 transition-transform duration-150 ease-out ${
          isMobile 
            ? `w-64 transform ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`
            : 'w-64'
        } dashboard-sidebar p-4`}
      >
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center gap-2 mb-8">
            <div className="p-2 bg-primary rounded-lg">
              <Eye className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="font-bold text-lg">Tortuosity AI</h1>
              <p className="text-xs text-muted-foreground">Análisis Avanzado</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-2">
            <SidebarItem
              icon={Home}
              label="Inicio"
              active={activeTab === 'upload'}
              onClick={() => setActiveTab('upload')}
            />
            <SidebarItem
              icon={Camera}
              label="Capturar"
              active={activeTab === 'capture'}
              onClick={() => setActiveTab('capture')}
            />
            <SidebarItem
              icon={Calculator}
              label="Validación Dice"
              active={activeTab === 'dice'}
              onClick={() => setActiveTab('dice')}
            />
            <SidebarItem
              icon={BarChart3}
              label="Resultados"
              active={activeTab === 'results'}
              onClick={() => setActiveTab('results')}
            />
            <SidebarItem
              icon={FileText}
              label="Información"
              active={activeTab === 'info'}
              onClick={() => setActiveTab('info')}
            />
          </nav>

          {/* Status */}
          <div className="pt-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span>Sistema Activo</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className={`transition-all duration-150 ease-out p-6 dashboard-content min-h-screen bg-background w-full ${
        isMobile ? 'ml-0' : 'ml-64'
      }`}>
        {/* Mobile Header with Hamburger */}
        {isMobile && (
          <div className="flex items-center gap-4 mb-6 lg:hidden">
            <button
              id="hamburger"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg hover:bg-muted transition-colors"
            >
              {sidebarOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
            <div className="flex items-center gap-2">
              <div className="p-2 bg-primary rounded-lg">
                <Eye className="h-5 w-5 text-primary-foreground" />
              </div>
              <div>
                <h1 className="font-bold text-lg">Tortuosity AI</h1>
                <p className="text-xs text-muted-foreground">Análisis Avanzado</p>
              </div>
            </div>
          </div>
        )}

        {/* Header */}
        <div className="mb-6">
          <h2 className="text-2xl font-bold">
            {activeTab === 'upload' && 'Análisis de Imagen'}
            {activeTab === 'capture' && 'Captura de Imagen IR - Módulo Especializado'}
            {activeTab === 'dice' && 'Validación Dice - Segmentación Manual'}
            {activeTab === 'results' && 'Resultados del Análisis'}
            {activeTab === 'info' && 'Información del Sistema'}
          </h2>
          <p className="text-muted-foreground">
            {activeTab === 'upload' && 'Sube una imagen del párpado para analizar la tortuosidad glandular'}
            {activeTab === 'capture' && 'Conecta tu módulo IR especializado y captura imágenes para análisis profesional de meibografía'}
            {activeTab === 'dice' && 'Dibuja manualmente la máscara ground truth y calcula el Dice Coefficient con la predicción del modelo'}
            {activeTab === 'results' && 'Visualiza los resultados detallados del análisis'}
            {activeTab === 'info' && 'Información sobre la metodología y modelos utilizados'}
          </p>
        </div>

        {/* Content */}
        {activeTab === 'upload' && (
          <div className="space-y-6 animate-fade-in bg-background">
            {/* Upload Zone */}
            <Card className="border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Cargar Imagen
                </CardTitle>
              </CardHeader>
              <CardContent>
                <UploadZone onFileSelect={handleFileSelect} selectedFile={selectedFile} />
                
                {/* Botón de ejemplo */}
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
                      Prueba la aplicación con una imagen de glándulas de Meibomio
                    </p>
                  </div>
                )}
                
                {(selectedFile || selectedCapturedPhoto) && (
                  <div className="mt-4 flex flex-col gap-4">
                    {/* CLAHE Options */}
                    <div className="flex flex-col items-center space-y-4">
                    <div className="flex flex-col items-center space-y-2">
                      <div className="flex items-center space-x-2">
                        <Switch
                          id="convert-to-gray"
                          checked={convertToGray}
                          onCheckedChange={handleConvertToGrayChange}
                        />
                        <Label htmlFor="convert-to-gray" className="text-sm">
                          {convertToGray ? "Procesar en escala de grises" : "Procesar en color"}
                        </Label>
                      </div>
                      <p className="text-xs text-muted-foreground text-center">
                        {convertToGray 
                          ? "Recomendado para imágenes con filtros de color o biomédicas"
                          : "Mantiene la información de color original"
                        }
                      </p>
                      </div>

                      <div className="flex flex-col items-center space-y-2">
                        <div className="flex items-center space-x-2">
                          <Switch
                            id="expansion-mode"
                            checked={expansionMode === "superior"}
                            onCheckedChange={(checked) => {
                              const newMode = checked ? "superior" : "inferior";
                              setExpansionMode(newMode);
                            }}
                          />
                          <Label htmlFor="expansion-mode" className="text-sm">
                            {expansionMode === "superior" ? "Superior" : "Inferior"}
                          </Label>
                        </div>
                        <div className="flex items-center justify-center space-x-2 text-xs text-muted-foreground">
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                          </svg>
                          <span>
                            {expansionMode === "inferior"
                              ? "Expande desde el borde inferior de glándulas hacia el tarso inferior"
                              : "Expande desde el borde superior de glándulas hacia el tarso superior"
                            }
                          </span>
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                          </svg>
                        </div>
                      </div>
                    </div>
                    
                    {/* Calibración espacial */}
                    <div className="flex flex-col items-center space-y-2 w-full">
                      <Label className="text-sm font-medium">Calibración espacial</Label>
                      <select
                        value={umPerPx}
                        onChange={(e) => setUmPerPx(parseFloat(e.target.value))}
                        className="w-full max-w-sm rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                      >
                        {EQUIPMENT_PRESETS.map((p) => (
                          <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                      </select>
                      <p className="text-xs text-muted-foreground text-center">
                        {umPerPx === 1.0
                          ? "Las medidas se mostrarán en píxeles"
                          : `Factor: ${umPerPx} µm/px — medidas en micrómetros reales`}
                      </p>
                    </div>

                    {/* Selector de modelo de segmentación */}
                    <div className="flex flex-col items-center space-y-2 w-full">
                      <Label className="text-sm font-medium">Modelo de segmentación</Label>
                      <div className="flex rounded-md border border-border overflow-hidden w-full max-w-sm">
                        <button
                          type="button"
                          onClick={() => setSegmentationModel("maskrcnn")}
                          className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                            segmentationModel === "maskrcnn"
                              ? "bg-primary text-primary-foreground"
                              : "bg-background text-muted-foreground hover:bg-muted"
                          }`}
                        >
                          Mask R-CNN
                        </button>
                        <button
                          type="button"
                          onClick={() => setSegmentationModel("panoptic")}
                          className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                            segmentationModel === "panoptic"
                              ? "bg-primary text-primary-foreground"
                              : "bg-background text-muted-foreground hover:bg-muted"
                          }`}
                        >
                          Mask2Former
                        </button>
                      </div>
                      <p className="text-xs text-muted-foreground text-center">
                        {segmentationModel === "panoptic"
                          ? "Mask2Former — segmentación panóptica de glándulas"
                          : "Mask R-CNN — detección de instancias clásica"}
                      </p>
                    </div>

                    <div className="flex flex-col sm:flex-row gap-2 justify-center">
                      <Button
                        onClick={analyzeImage}
                        disabled={isAnalyzing}
                        className="flex-1"
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Analizando...
                          </>
                        ) : (
                          <>
                            <Zap className="mr-2 h-4 w-4" />
                            Analizar Imagen
                          </>
                        )}
                      </Button>
                      <Button 
                        onClick={analyzeMgda} 
                        disabled={isAnalyzingMgda}
                        variant="secondary"
                        className="flex-1"
                      >
                        {isAnalyzingMgda ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            MGDA...
                          </>
                        ) : (
                          <>
                            <BarChart3 className="mr-2 h-4 w-4" />
                            Analizar MGDA
                          </>
                        )}
                      </Button>
                      <Button 
                        onClick={applyClaheFilter} 
                        disabled={isApplyingClahe}
                        variant="outline"
                        className="flex-1 border-border"
                      >
                        {isApplyingClahe ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Aplicando...
                          </>
                        ) : (
                          <>
                            <Sparkles className="mr-2 h-4 w-4" />
                            Aplicar CLAHE
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Progress */}
            {isAnalyzing && (
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm font-medium">Procesando imagen...</span>
                    </div>
                    <Progress value={progress} className="w-full" />
                    <p className="text-xs text-muted-foreground text-center">
                      Cargando modelos y analizando glándulas...
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* CLAHE Result */}
            {claheImage && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5" />
                    Imagen con CLAHE Aplicado
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <img
                    src={claheImage}
                    alt="Imagen con filtro CLAHE"
                    className="w-full rounded-lg shadow-lg"
                  />
                  <p className="text-sm text-muted-foreground mt-2 text-center">
                    Filtro CLAHE aplicado para mejorar el contraste local
                  </p>
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
          </div>
        )}

        {activeTab === 'capture' && (
          <div className="animate-fade-in bg-background">
            <PhotoManager
              onPhotoSelect={handleCapturedPhotoSelect}
              onAnalysisComplete={(results) => {
                setResults(results);
                setActiveTab('results');
              }}
              onTabChange={setActiveTab}
            />
          </div>
        )}

        {activeTab === 'dice' && (
          <div className="animate-fade-in bg-background">
            <DiceValidation onTabChange={setActiveTab} />
          </div>
        )}

        {activeTab === 'results' && (
          <div className="animate-fade-in bg-background">
            {results ? (
              <ResultsDisplay 
                data={results.data} 
                processedImage={results.data.processed_image} 
              />
            ) : mgdaResult ? (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Resultados MGDA - Análisis de Disfunción Glandular
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <img
                    src={mgdaResult.data.processed_image}
                    alt="Resultado MGDA"
                    className="w-full rounded-lg shadow-lg"
                  />
                  <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
                    <div className="p-3 rounded-lg bg-muted">
                      <div className="text-muted-foreground">MG Ratio</div>
                      <div className="font-semibold">{mgdaResult.data.metrics.mg_ratio.toFixed(4)}</div>
                    </div>
                    <div className="p-3 rounded-lg bg-muted">
                      <div className="text-muted-foreground">Disfunción (%)</div>
                      <div className="font-semibold">{mgdaResult.data.metrics.dysfunction_percentage.toFixed(2)}%</div>
                    </div>
                    <div className="p-3 rounded-lg bg-muted">
                      <div className="text-muted-foreground">Área disfuncional</div>
                      <div className="font-semibold">{mgdaResult.data.metrics.dysfunction_area}</div>
                    </div>
                  </div>
                  <div className="mt-4 p-3 rounded-lg bg-blue-50 dark:bg-blue-950">
                    <h4 className="font-semibold text-blue-900 dark:text-blue-100">Interpretación</h4>
                    <div className="text-sm text-blue-700 dark:text-blue-300 mt-1 space-y-1">
                      <p>• MG Ratio bajo: Posible disfunción glandular</p>
                      <p>• Disfunción alta: Sugestivo de enfermedad de glándulas de Meibomio</p>
                      <p>• Modo usado: {mgdaResult.data.used_expansion_mode === "inferior" ? "Expansión inferior" : "Expansión superior"}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center py-12">
                    <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <h3 className="text-lg font-medium mb-2">No hay resultados</h3>
                    <p className="text-muted-foreground">
                      Analiza una imagen primero para ver los resultados aquí.
                    </p>
                    <Button 
                      onClick={() => setActiveTab('upload')} 
                      className="mt-4"
                    >
                      Ir a Análisis
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {activeTab === 'info' && (
          <div className="space-y-6 animate-fade-in bg-background">
            {/* Methodology */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Metodología
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <h4 className="font-semibold text-green-600">Fórmula de Tortuosidad</h4>
                    <p className="text-sm text-muted-foreground">
                      Tortuosidad = (Perímetro / (2 × Altura del rectángulo mínimo externo)) - 1
                    </p>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold text-blue-600">Interpretación</h4>
                    <div className="text-sm text-muted-foreground space-y-1">
                      <p><strong>0.0 - 0.1:</strong> Tortuosidad baja (normal)</p>
                      <p><strong>0.1 - 0.2:</strong> Tortuosidad moderada</p>
                      <p><strong>&gt; 0.2:</strong> Tortuosidad alta (sugestivo de MGD)</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold text-purple-600">Modelos Utilizados</h4>
                    <div className="text-sm text-muted-foreground space-y-1">
                      <p><strong>Mask R-CNN:</strong> Detección de glándulas individuales</p>
                      <p><strong>UNet:</strong> Segmentación del contorno del párpado</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* System Info */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Información del Sistema
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <h4 className="font-semibold">Tecnologías</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Frontend: Next.js 15 con TypeScript</li>
                      <li>• Backend: FastAPI con Python</li>
                      <li>• IA: PyTorch (Mask R-CNN & UNet)</li>
                      <li>• UI: Tailwind CSS + shadcn/ui</li>
                    </ul>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold">Características</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• Análisis en tiempo real</li>
                      <li>• Visualización interactiva</li>
                      <li>• Métricas detalladas</li>
                      <li>• Interfaz responsiva</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
