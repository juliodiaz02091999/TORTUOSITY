"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import {
  Camera,
  Download,
  Trash2,
  FileImage,
  AlertCircle,
  CheckCircle,
  BarChart3,
  Image as ImageIcon,
  Loader2,
  Brain,
  Lightbulb
} from "lucide-react";
import { CameraCapture } from "./camera-capture";
import { photoStorage, StoredPhoto } from "@/lib/photo-storage";

interface PhotoManagerProps {
  onPhotoSelect?: (photo: StoredPhoto) => void;
  onAnalysisComplete?: (results: AnalysisResults) => void;
  onTabChange?: (tab: 'upload' | 'capture' | 'results' | 'info') => void;
}

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

interface AnalysisResults {
  success: boolean;
  message: string;
  data: {
    avg_tortuosity: number;
    num_glands: number;
    individual_tortuosities: number[];
    avg_length_px: number;
    avg_thickness_px: number;
    individual_lengths: number[];
    individual_thicknesses: number[];
    processed_image: string;
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
  processedImage?: string;
  timestamp?: Date;
}

export function PhotoManager({ onPhotoSelect, onAnalysisComplete, onTabChange }: PhotoManagerProps) {
  const [storedPhotos, setStoredPhotos] = useState<StoredPhoto[]>([]);
  const [selectedPhoto, setSelectedPhoto] = useState<StoredPhoto | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);

  // Load photos from storage
  const loadPhotos = useCallback(async () => {
    try {
      setIsLoading(true);
      setError('');
      const photos = await photoStorage.getAllPhotos();
      setStoredPhotos(photos);
    } catch (err) {
      console.error('Error loading photos:', err);
      setError('Error al cargar las fotos guardadas');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Handle photo capture from camera
  const handlePhotoCapture = useCallback(async (dataUrl: string) => {
    try {
      const photo: StoredPhoto = {
        id: Date.now().toString(),
        dataUrl,
        timestamp: new Date(),
        fileName: `meibomio-capture-${Date.now()}.jpg`
      };

      await photoStorage.savePhoto(photo);
      await loadPhotos(); // Refresh the list
    } catch (err) {
      console.error('Error saving captured photo:', err);
      setError('Error al guardar la foto capturada');
    }
  }, [loadPhotos]);

  // Handle photo selection
  const handlePhotoSelect = useCallback((photo: StoredPhoto) => {
    setSelectedPhoto(photo);
    if (onPhotoSelect) {
      onPhotoSelect(photo);
    }
  }, [onPhotoSelect]);

  // Delete photo
  const deletePhoto = useCallback(async (photoId: string) => {
    try {
      await photoStorage.deletePhoto(photoId);
      await loadPhotos();
      if (selectedPhoto?.id === photoId) {
        setSelectedPhoto(null);
      }
    } catch (err) {
      console.error('Error deleting photo:', err);
      setError('Error al eliminar la foto');
    }
  }, [loadPhotos, selectedPhoto]);

  // Download photo
  const downloadPhoto = useCallback((photo: StoredPhoto) => {
    const link = document.createElement('a');
    link.href = photo.dataUrl;
    link.download = photo.fileName || `meibomio-${photo.timestamp.getTime()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  // Analyze selected photo
  const analyzeSelectedPhoto = useCallback(async () => {
    if (!selectedPhoto) return;

    setIsAnalyzing(true);
    setError('');
    setAnalysisProgress(0);
    setAnalysisResults(null);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setAnalysisProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 500);

    try {
      // Convert data URL to File object
      const response = await fetch(selectedPhoto.dataUrl);
      const blob = await response.blob();
      const fileToAnalyze = new File([blob], selectedPhoto.fileName || 'captured-image.jpg', { type: 'image/jpeg' });

      const formData = new FormData();
      formData.append("file", fileToAnalyze);

      console.log('Enviando imagen para análisis...');
      const response2 = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      console.log('📡 Respuesta del servidor:', response2.status, response2.statusText);

      if (!response2.ok) {
        const contentType = response2.headers.get("content-type");
        let errorMessage = '';
        
        if (contentType && contentType.includes("application/json")) {
          try {
            const errorData = await response2.json();
            errorMessage = errorData.detail || errorData.message || `Error ${response2.status}`;
          } catch {
            errorMessage = `Error del servidor: ${response2.status}`;
          }
        } else {
          try {
            const errorText = await response2.text();
            errorMessage = `Error del servidor: ${response2.status} - ${errorText.substring(0, 100)}`;
          } catch {
            errorMessage = `Error del servidor: ${response2.status}`;
          }
        }

        // Handle specific HTTP errors
        if (response2.status === 503) {
          errorMessage = 'Servicio temporalmente no disponible. Los modelos de IA están cargando. Intenta de nuevo en unos segundos.';
        } else if (response2.status === 500) {
          errorMessage = 'Error interno del servidor. Contacta al administrador si persiste.';
        } else if (response2.status === 413) {
          errorMessage = 'Imagen demasiado grande. Intenta con una imagen de menor resolución.';
        }

        throw new Error(errorMessage);
      }

      const result = await response2.json();
      console.log('Análisis completado exitosamente:', result);
      
      // Ensure the result has the required properties for AnalysisResults interface
      const analysisResult: AnalysisResults = {
        success: true,
        message: 'Análisis completado exitosamente',
        data: {
          avg_tortuosity: result.data.avg_tortuosity,
          num_glands: result.data.num_glands,
          individual_tortuosities: result.data.individual_tortuosities,
          avg_length_px: result.data.avg_length_px ?? 0,
          avg_thickness_px: result.data.avg_thickness_px ?? 0,
          individual_lengths: result.data.individual_lengths ?? [],
          individual_thicknesses: result.data.individual_thicknesses ?? [],
          processed_image: result.data.processed_image || '',
          analysis_info: result.data.analysis_info || {
            total_glands_analyzed: result.data.num_glands || 0,
            tortuosity_range: {
              min: Math.min(...(result.data.individual_tortuosities || [0])),
              max: Math.max(...(result.data.individual_tortuosities || [0]))
            }
          },
          um_per_px: result.data.um_per_px,
          avg_length_um: result.data.avg_length_um,
          avg_thickness_um: result.data.avg_thickness_um,
          avg_ICM: result.data.avg_ICM,
          avg_ITA_deg: result.data.avg_ITA_deg,
          avg_tortuosity_score: result.data.avg_tortuosity_score,
          dominant_grade: result.data.dominant_grade,
          individual_glands: result.data.individual_glands,
        },
        processedImage: result.data.processed_image || '',
        timestamp: new Date()
      };
      
      setAnalysisResults(analysisResult);

      // Save analysis results to local storage
      try {
        const localAnalysisResults = {
          avgTortuosity: result.data.avg_tortuosity,
          numGlands: result.data.num_glands,
          individualTortuosities: result.data.individual_tortuosities
        };

        // Update the photo with analysis results
        await photoStorage.updatePhotoAnalysis(selectedPhoto.id, localAnalysisResults);
        
        // Refresh the photos list to show updated analysis
        await loadPhotos();
        
        // Update selected photo with analysis results
        setSelectedPhoto({ ...selectedPhoto, analysisResults: localAnalysisResults });

        // Notify parent component about analysis completion
        if (onAnalysisComplete) {
          onAnalysisComplete(analysisResult);
        }

        // Change to results tab after successful analysis
        if (onTabChange) {
          onTabChange('results');
        }

      } catch (error) {
        console.warn('Could not save analysis results to local storage:', error);
      }

      setAnalysisProgress(100);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Error desconocido";
      console.error('Error durante el análisis:', errorMessage);
      setError(errorMessage);
    } finally {
      setIsAnalyzing(false);
      clearInterval(progressInterval);
    }
  }, [selectedPhoto, loadPhotos, onAnalysisComplete, onTabChange]);

  // Export all photos
  const exportAllPhotos = useCallback(async () => {
    try {
      const jsonData = await photoStorage.exportPhotos();
      const blob = new Blob([jsonData], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `meibomio-photos-backup-${Date.now()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Error exporting photos:', err);
      setError('Error al exportar las fotos');
    }
  }, []);

  // Clear all photos
  const clearAllPhotos = useCallback(async () => {
    if (confirm('¿Estás seguro de que quieres eliminar todas las fotos? Esta acción no se puede deshacer.')) {
      try {
        await photoStorage.clearAllPhotos();
        await loadPhotos();
        setSelectedPhoto(null);
      } catch (err) {
        console.error('Error clearing photos:', err);
        setError('Error al eliminar todas las fotos');
      }
    }
  }, [loadPhotos]);

  // Load photos on component mount
  useEffect(() => {
    loadPhotos();
  }, [loadPhotos]);

  return (
    <div className="space-y-4 sm:space-y-6 overflow-visible">
      <Tabs defaultValue="capture" className="w-full overflow-visible">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="capture" className="flex items-center gap-2 text-xs sm:text-sm">
            <Camera className="h-3 w-3 sm:h-4 sm:w-4" />
            Capturar
          </TabsTrigger>
          <TabsTrigger value="gallery" className="flex items-center gap-2 text-xs sm:text-sm">
            <ImageIcon className="h-3 w-3 sm:h-4 sm:w-4" />
            Galería ({storedPhotos.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="capture" className="space-y-4 overflow-visible">
          <CameraCapture onPhotoCapture={handlePhotoCapture} />
        </TabsContent>

        <TabsContent value="gallery" className="space-y-4 overflow-visible">
          {/* Gallery Header */}
          <Card className="border-border">
            <CardHeader>
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                  <FileImage className="h-4 w-4 sm:h-5 sm:w-5" />
                  Fotos Guardadas ({storedPhotos.length})
                </CardTitle>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={exportAllPhotos}
                    disabled={storedPhotos.length === 0}
                    className="border-border text-xs sm:text-sm"
                  >
                    <Download className="mr-1 sm:mr-2 h-3 w-3 sm:h-4 sm:w-4" />
                    Exportar
                  </Button>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={clearAllPhotos}
                    disabled={storedPhotos.length === 0}
                    className="text-xs sm:text-sm"
                  >
                    <Trash2 className="mr-1 sm:mr-2 h-3 w-3 sm:h-4 sm:w-4" />
                    Limpiar Todo
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {error && (
                <Alert className="border-red-200 bg-red-50 mb-4">
                  <AlertCircle className="h-4 w-4 text-red-600" />
                  <AlertDescription className="text-red-800">{error}</AlertDescription>
                </Alert>
              )}

              {isLoading ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
                  <p className="text-muted-foreground">Cargando fotos...</p>
                </div>
              ) : (
                <>
                  {/* Analysis functionality info */}
                  <div className="p-3 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800/30 rounded-lg">
                    <div className="flex items-start gap-2">
                      <Brain className="h-4 w-4 sm:h-5 sm:w-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
                      <div className="text-xs sm:text-sm text-blue-800 dark:text-blue-200">
                        <p className="font-medium mb-1 flex items-center gap-2">
                          <Lightbulb className="h-3 w-3 sm:h-4 sm:w-4" />
                          Nueva Funcionalidad: Análisis Directo
                        </p>
                        <p>Ahora puedes analizar cualquier foto guardada directamente desde la galería:</p>
                        <ul className="list-disc list-inside mt-1 space-y-1">
                          <li>Haz clic en <strong>&quot;Analizar&quot;</strong> en cualquier foto para procesarla</li>
                          <li>Los resultados se guardan automáticamente con la foto</li>
                          <li>Las fotos analizadas muestran un badge verde &quot;Analizada&quot;</li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  {storedPhotos.length === 0 ? (
                    <div className="text-center py-8 sm:py-12">
                      <FileImage className="h-8 w-8 sm:h-12 sm:w-12 text-muted-foreground mx-auto mb-4" />
                      <h3 className="text-base sm:text-lg font-medium mb-2">No hay fotos guardadas</h3>
                      <p className="text-muted-foreground mb-4 text-sm">
                        Captura algunas fotos usando la pestaña &quot;Capturar&quot;
                      </p>
                      <Button onClick={() => {
                        const captureTab = document.querySelector('[value="capture"]') as HTMLButtonElement;
                        if (captureTab) captureTab.click();
                      }} size="sm">
                        <Camera className="mr-2 h-3 w-3 sm:h-4 sm:w-4" />
                        Ir a Capturar
                      </Button>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
                      {storedPhotos.map((photo) => (
                        <Card
                          key={photo.id}
                          className={`cursor-pointer transition-all hover:shadow-md border-border ${
                            selectedPhoto?.id === photo.id ? 'ring-2 ring-primary' : ''
                          }`}
                          onClick={() => handlePhotoSelect(photo)}
                        >
                          <CardContent className="p-3 sm:p-4">
                            <div className="relative mb-3">
                              <img
                                src={photo.dataUrl}
                                alt={`Foto ${photo.timestamp.toLocaleDateString()}`}
                                className="w-full h-24 sm:h-32 object-cover rounded-lg"
                              />
                              {photo.analysisResults && (
                                <Badge className="absolute top-1 right-1 sm:top-2 sm:right-2 bg-green-500 text-xs">
                                  <BarChart3 className="h-2 w-2 sm:h-3 sm:w-3 mr-1" />
                                  Analizada
                                </Badge>
                              )}
                              {isAnalyzing && selectedPhoto?.id === photo.id && (
                                <Badge className="absolute top-1 right-1 sm:top-2 sm:right-2 bg-blue-500 animate-pulse text-xs">
                                  <Loader2 className="h-2 w-2 sm:h-3 sm:w-3 animate-spin" />
                                  Analizando
                                </Badge>
                              )}
                            </div>

                            <div className="space-y-2">
                              <p className="text-xs sm:text-sm font-medium">
                                {photo.timestamp.toLocaleDateString()} {photo.timestamp.toLocaleTimeString()}
                              </p>

                              {photo.analysisResults && (
                                <div className="text-xs text-muted-foreground">
                                  <p>Tortuosidad: {photo.analysisResults.avgTortuosity.toFixed(3)}</p>
                                  <p>Glándulas: {photo.analysisResults.numGlands}</p>
                                </div>
                              )}

                              <div className="grid grid-cols-2 gap-1">
                                {!photo.analysisResults && (
                                  <Button
                                    size="sm"
                                    variant="default"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setSelectedPhoto(photo);
                                      analyzeSelectedPhoto();
                                    }}
                                    className="w-full border-border text-xs"
                                    disabled={isAnalyzing}
                                  >
                                    {isAnalyzing && selectedPhoto?.id === photo.id ? (
                                      <Loader2 className="h-2 w-2 sm:h-3 sm:w-3 mr-1 animate-spin" />
                                    ) : (
                                      <Brain className="h-2 w-2 sm:h-3 sm:w-3 mr-1" />
                                    )}
                                    Analizar
                                  </Button>
                                )}
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    downloadPhoto(photo);
                                  }}
                                  className="w-full border-border text-xs"
                                >
                                  <Download className="h-2 w-2 sm:h-3 sm:w-3 mr-1" />
                                  Descargar
                                </Button>
                                <Button
                                  size="sm"
                                  variant="destructive"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    deletePhoto(photo.id);
                                  }}
                                  className="w-full text-xs"
                                >
                                  <Trash2 className="h-2 w-2 sm:h-3 sm:w-3 mr-1" />
                                  Eliminar
                                </Button>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>

          {/* Selected Photo Details */}
          {selectedPhoto && (
            <Card className="border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                  <CheckCircle className="h-4 w-4 sm:h-5 sm:w-5" />
                  Foto Seleccionada
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <img
                    src={selectedPhoto.dataUrl}
                    alt="Foto seleccionada"
                    className="w-full max-h-64 sm:max-h-96 object-contain rounded-lg border"
                  />

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="font-medium">Fecha:</p>
                      <p className="text-muted-foreground text-xs sm:text-sm">
                        {selectedPhoto.timestamp.toLocaleDateString()} {selectedPhoto.timestamp.toLocaleTimeString()}
                      </p>
                    </div>

                    {selectedPhoto.analysisResults ? (
                      <div className="space-y-2">
                        <p className="font-medium">Resultados del Análisis:</p>
                        <div className="text-muted-foreground text-xs sm:text-sm">
                          <p>Tortuosidad promedio: {selectedPhoto.analysisResults.avgTortuosity.toFixed(3)}</p>
                          <p>Número de glándulas: {selectedPhoto.analysisResults.numGlands}</p>
                          <p>Glándulas analizadas: {selectedPhoto.analysisResults.individualTortuosities.length}</p>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <p className="font-medium">Estado:</p>
                        <Badge variant="outline">Sin analizar</Badge>
                      </div>
                    )}
                  </div>

                  <div className="space-y-3">
                    {/* Analysis Progress */}
                    {isAnalyzing && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-sm">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          <span>Analizando imagen...</span>
                        </div>
                        <Progress value={analysisProgress} className="w-full" />
                      </div>
                    )}

                    {/* Error Display */}
                    {error && (
                      <Alert className="border-red-200 bg-red-50">
                        <AlertCircle className="h-4 w-4 text-red-600" />
                        <AlertDescription className="text-red-800">
                          <div className="space-y-2">
                            <p className="font-medium text-sm">{error}</p>
                            
                            {/* Helpful tips for common errors */}
                            {error.includes('503') && (
                              <div className="text-xs sm:text-sm bg-red-100 p-2 rounded">
                                <p className="font-medium flex items-center gap-2">
                                  <Lightbulb className="h-3 w-3 sm:h-4 sm:w-4" />
                                  Solución:
                                </p>
                                <ul className="list-disc list-inside mt-1">
                                  <li>Los modelos de IA están cargando</li>
                                  <li>Espera 10-30 segundos y vuelve a intentar</li>
                                  <li>Este es un proceso normal al iniciar la aplicación</li>
                                </ul>
                              </div>
                            )}
                            
                            {error.includes('500') && (
                              <div className="text-xs sm:text-sm bg-red-100 p-2 rounded">
                                <p className="font-medium flex items-center gap-2">
                                  <Lightbulb className="h-3 w-3 sm:h-4 sm:w-4" />
                                  Solución:
                                </p>
                                <ul className="list-disc list-inside mt-1">
                                  <li>Error interno del servidor</li>
                                  <li>Intenta con una imagen diferente</li>
                                  <li>Si persiste, contacta al administrador</li>
                                </ul>
                              </div>
                            )}
                            
                            {error.includes('413') && (
                              <div className="text-xs sm:text-sm bg-red-100 p-2 rounded">
                                <p className="font-medium flex items-center gap-2">
                                  <Lightbulb className="h-3 w-3 sm:h-4 sm:w-4" />
                                  Solución:
                                </p>
                                <ul className="list-inside mt-1">
                                  <li>La imagen es demasiado grande</li>
                                  <li>Usa una imagen de menor resolución</li>
                                  <li>Recomendado: máximo 1920x1080</li>
                                </ul>
                              </div>
                            )}
                          </div>
                        </AlertDescription>
                      </Alert>
                    )}

                    {/* Analysis Results */}
                    {analysisResults && (
                      <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                        <h4 className="font-medium text-green-800 mb-2 flex items-center gap-2 text-sm">
                          <CheckCircle className="h-3 w-3 sm:h-4 sm:w-4" />
                          Análisis Completado
                        </h4>
                        <div className="text-xs sm:text-sm text-green-700 space-y-1">
                          <p>Tortuosidad promedio: {analysisResults.data.avg_tortuosity.toFixed(3)}</p>
                          <p>Número de glándulas: {analysisResults.data.num_glands}</p>
                          <p>Glándulas analizadas: {analysisResults.data.individual_tortuosities.length}</p>
                        </div>
                        <div className="mt-3 pt-3 border-t border-green-200">
                          <Button 
                            onClick={() => onTabChange?.('results')}
                            size="sm"
                            className="w-full bg-green-600 hover:bg-green-700 text-white border-border text-xs sm:text-sm"
                          >
                            <BarChart3 className="h-3 w-3 sm:h-4 sm:w-4 mr-2" />
                            Ver Resultados Completos
                          </Button>
                        </div>
                      </div>
                    )}

                    <div className="flex flex-col sm:flex-row gap-2">
                      <Button 
                        onClick={analyzeSelectedPhoto} 
                        disabled={isAnalyzing}
                        className="flex-1 border-border text-xs sm:text-sm"
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="mr-2 h-3 w-3 sm:h-4 sm:w-4 animate-spin" />
                            Analizando...
                          </>
                        ) : (
                          <>
                            <Brain className="mr-2 h-3 w-3 sm:h-4 sm:w-4" />
                            Analizar Esta Foto
                          </>
                        )}
                      </Button>
                      <Button variant="outline" onClick={() => downloadPhoto(selectedPhoto)} className="border-border text-xs sm:text-sm">
                        <Download className="mr-2 h-3 w-3 sm:h-4 sm:w-4" />
                        Descargar
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
