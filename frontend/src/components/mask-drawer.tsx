"use client";

import { useRef, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Trash2, Check, X } from "lucide-react";

interface Point { x: number; y: number }

interface MaskDrawerProps {
  imageUrl: string;
  onConfirm: (maskDataUrl: string) => void;
  onClose: () => void;
}

const MAX_DIM = 680;

export function MaskDrawer({ imageUrl, onConfirm, onClose }: MaskDrawerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [cursor, setCursor] = useState<Point | null>(null);
  const [cw, setCw] = useState(0);
  const [ch, setCh] = useState(0);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const image = new Image();
    image.onload = () => {
      const s = Math.min(MAX_DIM / image.naturalWidth, MAX_DIM / image.naturalHeight, 1);
      setScale(s);
      setCw(Math.round(image.naturalWidth * s));
      setCh(Math.round(image.naturalHeight * s));
      setImg(image);
    };
    image.src = imageUrl;
  }, [imageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img || cw === 0) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, cw, ch);
    ctx.drawImage(img, 0, 0, cw, ch);
    if (points.length === 0) return;

    // Semi-transparent fill preview
    if (points.length >= 3) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      ctx.closePath();
      ctx.fillStyle = "rgba(59,130,246,0.25)";
      ctx.fill();
    }

    // Polygon outline + cursor line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
    if (cursor) ctx.lineTo(cursor.x, cursor.y);
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.stroke();

    // Dashed closing line
    if (cursor && points.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(cursor.x, cursor.y);
      ctx.lineTo(points[0].x, points[0].y);
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "rgba(59,130,246,0.5)";
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw vertices
    points.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, i === 0 ? 7 : 4, 0, Math.PI * 2);
      ctx.fillStyle = i === 0 ? "#ef4444" : "#3b82f6";
      ctx.fill();
      ctx.strokeStyle = "white";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }, [img, points, cursor, cw, ch]);

  const toCanvas = (e: React.MouseEvent<HTMLCanvasElement>): Point => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (cw / rect.width),
      y: (e.clientY - rect.top) * (ch / rect.height),
    };
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pt = toCanvas(e);
    if (points.length >= 3) {
      const dx = pt.x - points[0].x;
      const dy = pt.y - points[0].y;
      if (Math.sqrt(dx * dx + dy * dy) < 14) { finalize(); return; }
    }
    setPoints(prev => [...prev, pt]);
  };

  const finalize = () => {
    if (points.length < 3 || !img) return;
    const off = document.createElement("canvas");
    off.width = img.naturalWidth;
    off.height = img.naturalHeight;
    const ctx = off.getContext("2d")!;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, off.width, off.height);
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.moveTo(points[0].x / scale, points[0].y / scale);
    points.slice(1).forEach(p => ctx.lineTo(p.x / scale, p.y / scale));
    ctx.closePath();
    ctx.fill();
    onConfirm(off.toDataURL("image/png"));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4">
      <div className="bg-background rounded-xl shadow-2xl flex flex-col gap-3 p-4 max-w-[95vw] max-h-[95vh]">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-sm">Contorno del párpado</h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Clic para añadir vértices · Clic sobre el punto rojo para cerrar
            </p>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {img ? (
          <canvas
            ref={canvasRef}
            width={cw}
            height={ch}
            onClick={handleClick}
            onMouseMove={e => setCursor(toCanvas(e))}
            onMouseLeave={() => setCursor(null)}
            className="rounded border border-border cursor-crosshair"
            style={{ maxWidth: "100%", maxHeight: "65vh", objectFit: "contain" }}
          />
        ) : (
          <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
            Cargando imagen…
          </div>
        )}

        <div className="flex gap-2 justify-end">
          <Button variant="outline" size="sm" onClick={() => setPoints([])} disabled={points.length === 0}>
            <Trash2 className="h-3 w-3 mr-1" /> Limpiar
          </Button>
          <Button size="sm" onClick={finalize} disabled={points.length < 3}>
            <Check className="h-3 w-3 mr-1" /> Confirmar ({points.length} pts)
          </Button>
        </div>
      </div>
    </div>
  );
}
