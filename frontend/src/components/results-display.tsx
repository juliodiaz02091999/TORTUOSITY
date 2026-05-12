"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Eye, Activity, TrendingUp, Info, Ruler, AlignVerticalJustifyCenter, Microscope, Award } from "lucide-react";

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

interface TortuosityData {
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
}

interface ResultsDisplayProps {
  data: TortuosityData;
  processedImage: string;
}

function toFiniteNumber(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function median(nums: number[]): number | undefined {
  const a = nums.filter(Number.isFinite).sort((x, y) => x - y);
  if (!a.length) return undefined;
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

function minMax(nums: number[]): { min?: number; max?: number } {
  const a = nums.filter(Number.isFinite);
  if (!a.length) return {};
  return { min: Math.min(...a), max: Math.max(...a) };
}

function arithmeticMean(nums: number[]): number | undefined {
  const a = nums.filter(Number.isFinite);
  if (!a.length) return undefined;
  return a.reduce((s, x) => s + x, 0) / a.length;
}

export function ResultsDisplay({ data, processedImage }: ResultsDisplayProps) {
  const calibrated = (data.um_per_px ?? 1.0) > 1.0;
  const unitLabel = calibrated ? "µm" : "px";

  const glands = data.individual_glands ?? [];

  const { meanITA, itaRange, avgScore, avgICM } = useMemo(() => {
    const list = data.individual_glands ?? [];
    const itas = list.map((g) => toFiniteNumber(g.ITA_deg)).filter((n): n is number => n !== undefined);
    const scores = list.map((g) => toFiniteNumber(g.tortuosity_score)).filter((n): n is number => n !== undefined);
    const icms = list.map((g) => toFiniteNumber(g.ICM)).filter((n): n is number => n !== undefined);
    const backendMeanIta = toFiniteNumber(data.avg_ITA_deg);
    return {
      meanITA: backendMeanIta ?? (itas.length ? arithmeticMean(itas) : undefined),
      itaRange: minMax(itas),
      avgScore: toFiniteNumber(data.avg_tortuosity_score) ?? (scores.length ? median(scores) : undefined),
      avgICM: toFiniteNumber(data.avg_ICM) ?? (icms.length ? median(icms) : undefined),
    };
  }, [data.avg_ITA_deg, data.avg_tortuosity_score, data.avg_ICM, data.individual_glands]);

  const tableRows = useMemo(() => {
    if (glands.length > 0) {
      return glands.map((g, i) => ({
        g,
        legacyTort: data.individual_tortuosities[i] ?? 0,
      }));
    }
    return data.individual_tortuosities.map((legacyTort, i) => ({
      g: undefined as GlandResult | undefined,
      legacyTort,
    }));
  }, [glands, data.individual_tortuosities]);

  const getGradeStyle = (grade: string) => {
    switch (grade) {
      case "Normal":   return { color: "bg-green-500",  label: "Normal" };
      case "Leve":     return { color: "bg-yellow-400", label: "Leve" };
      case "Moderada": return { color: "bg-orange-500", label: "Moderada" };
      case "Severa":   return { color: "bg-red-600",    label: "Severa" };
      default:         return { color: "bg-gray-400",   label: grade };
    }
  };

  // Grado derivado del score promedio (coherente con la escala 0-100)
  const scoreToGrade = (score: number) => {
    if (score < 20) return "Normal";
    if (score < 45) return "Leve";
    if (score < 70) return "Moderada";
    return "Severa";
  };

  const globalGrade =
    avgScore != null
      ? scoreToGrade(avgScore)
      : data.dominant_grade ?? "Normal";

  const chartData = tableRows.map(({ g, legacyTort }, index) => {
    const rawScore = toFiniteNumber(g?.tortuosity_score);
    return {
      gland: g?.gland_id ?? `G${index + 1}`,
      score: rawScore ?? 0,
      ICM: toFiniteNumber(g?.ICM) ?? legacyTort,
    };
  });

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Main Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 sm:gap-4">
        <Card className="border-border">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs sm:text-sm font-medium">Tortuosidad ICM</CardTitle>
            <Activity className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-lg sm:text-2xl font-bold">
              {avgICM != null ? avgICM.toFixed(4) : data.avg_tortuosity?.toFixed(3) ?? "—"}
            </div>
            <p className="text-xs text-muted-foreground">Índice curva media</p>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs sm:text-sm font-medium">Glándulas</CardTitle>
            <TrendingUp className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-lg sm:text-2xl font-bold">{data.num_glands}</div>
            <p className="text-xs text-muted-foreground">Total identificadas</p>
          </CardContent>
        </Card>


        <Card className="border-border">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs sm:text-sm font-medium">Grado Clínico</CardTitle>
            <Award className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-lg sm:text-2xl font-bold">{globalGrade}</div>
            <p className="text-xs text-muted-foreground">
              Score: {avgScore != null ? `${avgScore.toFixed(1)}/100` : "—"}
            </p>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs sm:text-sm font-medium">Longitud (promedio)</CardTitle>
            <Ruler className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-lg sm:text-2xl font-bold">
              {calibrated ? (data.avg_length_um?.toFixed(0) ?? "—") : (data.avg_length_px?.toFixed(1) ?? "—")}
            </div>
            <p className="text-xs text-muted-foreground">{unitLabel}</p>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xs sm:text-sm font-medium">Grosor (promedio)</CardTitle>
            <AlignVerticalJustifyCenter className="h-3 w-3 sm:h-4 sm:w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-lg sm:text-2xl font-bold">
              {calibrated ? (data.avg_thickness_um?.toFixed(0) ?? "—") : (data.avg_thickness_px?.toFixed(1) ?? "—")}
            </div>
            <p className="text-xs text-muted-foreground">{unitLabel}</p>
          </CardContent>
        </Card>
      </div>

      {/* Pipeline Clinical Summary */}
      {data.individual_glands && data.individual_glands.length > 0 && (
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
              <Microscope className="h-4 w-4 sm:h-5 sm:w-5" />
              Resumen Clínico
              {calibrated && <span className="text-xs font-normal text-muted-foreground ml-2">Factor: {data.um_per_px} µm/px</span>}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 rounded-lg bg-muted text-center">
                <div className="text-xs text-muted-foreground mb-1">ICM (mediana)</div>
                <div className="font-bold text-lg">{avgICM != null ? avgICM.toFixed(4) : "—"}</div>
                <div className="text-xs text-muted-foreground">longitud / distancia directa</div>
              </div>
              <div className="p-3 rounded-lg bg-muted text-center">
                <div className="text-xs text-muted-foreground mb-1">Score Clínico</div>
                <div className="font-bold text-lg">
                  {avgScore != null ? avgScore.toFixed(1) : "—"}
                  <span className="text-sm font-normal">/100</span>
                </div>
                <div className="text-xs text-muted-foreground">ponderado multi-índice</div>
              </div>
              <div className="p-3 rounded-lg bg-muted text-center">
                <div className="text-xs text-muted-foreground mb-1">Grado Global</div>
                <div className="flex justify-center mt-1">
                  <Badge className={`${getGradeStyle(globalGrade).color} text-white text-sm px-3 py-1`}>
                    {globalGrade}
                  </Badge>
                </div>
                {data.dominant_grade && data.dominant_grade !== globalGrade && (
                  <div className="text-xs text-muted-foreground mt-1">
                    Moda: {data.dominant_grade}
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Image and Table Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        {/* Processed Image */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
              <Eye className="h-4 w-4 sm:h-5 sm:w-5" />
              Imagen Procesada
            </CardTitle>
          </CardHeader>
          <CardContent>
            <img
              src={processedImage}
              alt="Imagen procesada"
              className="w-full rounded-lg shadow-lg"
            />
          </CardContent>
        </Card>

        {/* Individual Tortuosity Table */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base sm:text-lg">Métricas por Glándula</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="max-h-64 sm:max-h-96 overflow-y-auto custom-scrollbar">
              <Table>
                <TableHeader>
                  <TableRow className="bg-card hover:bg-card border-b-2 border-border">
                    <TableHead className="text-xs sm:text-sm font-medium text-muted-foreground">ID</TableHead>
                    <TableHead className="text-xs sm:text-sm font-medium text-muted-foreground">ICM</TableHead>
                    <TableHead className="text-xs sm:text-sm font-medium text-muted-foreground">Long. ({unitLabel})</TableHead>
                    <TableHead className="text-xs sm:text-sm font-medium text-muted-foreground">Grosor ({unitLabel})</TableHead>
                    <TableHead className="text-xs sm:text-sm font-medium text-muted-foreground">Score</TableHead>
                    <TableHead className="text-xs sm:text-sm font-medium text-muted-foreground">Grado</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {tableRows.map(({ g, legacyTort: value }, index) => {
                    const icmN = toFiniteNumber(g?.ICM);
                    const itaN = toFiniteNumber(g?.ITA_deg);
                    const scoreN = toFiniteNumber(g?.tortuosity_score);
                    const grade = g?.tortuosity_grade ?? (value <= 0.1 ? "Normal" : value <= 0.2 ? "Leve" : "Severa");
                    const gradeStyle = getGradeStyle(grade);
                    return (
                      <TableRow key={g?.gland_id ?? index}>
                        <TableCell className="font-medium text-xs sm:text-sm">{g?.gland_id ?? `G${index + 1}`}</TableCell>
                        <TableCell className="text-xs sm:text-sm">
                          {icmN != null ? icmN.toFixed(4) : value.toFixed(3)}
                        </TableCell>
                        <TableCell className="text-xs sm:text-sm">
                          {calibrated
                            ? (g?.length_um != null ? Number(g.length_um).toFixed(0) : "—")
                            : (data.individual_lengths?.[index]?.toFixed(1) ?? "—")}
                        </TableCell>
                        <TableCell className="text-xs sm:text-sm">
                          {calibrated
                            ? (g?.thickness_um != null ? Number(g.thickness_um).toFixed(0) : "—")
                            : (data.individual_thicknesses?.[index]?.toFixed(1) ?? "—")}
                        </TableCell>
                        <TableCell className="text-xs sm:text-sm">
                          {scoreN != null ? scoreN.toFixed(1) : "—"}
                        </TableCell>
                        <TableCell>
                          <Badge className={`${gradeStyle.color} text-white text-xs`}>
                            {gradeStyle.label}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Chart */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle className="text-base sm:text-lg">Score Clínico por Glándula</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] sm:h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gland" />
                <YAxis domain={[0, 100]} tickCount={6} />
                <Tooltip
                  formatter={(value: unknown, name: string) => {
                    const n = typeof value === "number" && Number.isFinite(value) ? value : NaN;
                    if (name === "score") {
                      return [
                        Number.isFinite(n) ? `${n.toFixed(1)} / 100` : "—",
                        "Score clínico",
                      ];
                    }
                    return [Number.isFinite(n) ? n.toFixed(4) : "—", "ICM"];
                  }}
                  labelFormatter={(label) => `Glándula ${label}`}
                />
                <Bar dataKey="score" fill="#3b82f6" radius={[4, 4, 0, 0]} name="score" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-4 mt-3 text-xs text-muted-foreground">
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-green-500"></span> Normal (0–20)</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-yellow-400"></span> Leve (20–45)</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-orange-500"></span> Moderada (45–70)</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-red-600"></span> Severa (&gt;70)</span>
          </div>
        </CardContent>
      </Card>

      {/* Interpretation Guide */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle className="text-base sm:text-lg">Guía de Interpretación Clínica</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
            <div className="text-center p-3 sm:p-4 bg-green-50 dark:bg-green-950 rounded-lg">
              <div className="text-lg sm:text-2xl font-bold text-green-600">0 – 20</div>
              <div className="text-xs sm:text-sm font-semibold text-green-700 dark:text-green-300">Normal</div>
              <div className="text-xs text-green-600 dark:text-green-400 mt-1">ICM ≈ 1.0</div>
            </div>
            <div className="text-center p-3 sm:p-4 bg-yellow-50 dark:bg-yellow-950 rounded-lg">
              <div className="text-lg sm:text-2xl font-bold text-yellow-600">20 – 45</div>
              <div className="text-xs sm:text-sm font-semibold text-yellow-700 dark:text-yellow-300">Leve</div>
              <div className="text-xs text-yellow-600 dark:text-yellow-400 mt-1">Cambios incipientes</div>
            </div>
            <div className="text-center p-3 sm:p-4 bg-orange-50 dark:bg-orange-950 rounded-lg">
              <div className="text-lg sm:text-2xl font-bold text-orange-600">45 – 70</div>
              <div className="text-xs sm:text-sm font-semibold text-orange-700 dark:text-orange-300">Moderada</div>
              <div className="text-xs text-orange-600 dark:text-orange-400 mt-1">Seguimiento clínico</div>
            </div>
            <div className="text-center p-3 sm:p-4 bg-red-50 dark:bg-red-950 rounded-lg">
              <div className="text-lg sm:text-2xl font-bold text-red-600">&gt; 70</div>
              <div className="text-xs sm:text-sm font-semibold text-red-700 dark:text-red-300">Severa</div>
              <div className="text-xs text-red-600 dark:text-red-400 mt-1">Sugestivo de MGD</div>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-muted-foreground">
            <div className="p-3 rounded-lg bg-muted">
              <span className="font-semibold">ICM</span> — Índice de Curva Media: longitud real / distancia directa (1.0 = recto)
            </div>
            <div className="p-3 rounded-lg bg-muted">
              <span className="font-semibold">DCF</span> — Densidad de inflexiones por µm de longitud
            </div>
            <div className="p-3 rounded-lg bg-muted">
              <span className="font-semibold">Score</span> — Basado en ICM: (ICM − 1) × 100, saturado en 100
            </div>
          </div>
          <p className="text-xs text-muted-foreground italic text-center">
            La interpretación final debe ser realizada por un especialista. Rangos basados en literatura (Arita et al., Pult &amp; Nichols).
          </p>
        </CardContent>
      </Card>
    </div>
  );
} 