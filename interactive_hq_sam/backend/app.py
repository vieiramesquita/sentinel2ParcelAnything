"""
Servidor FastAPI para Aplicação de Segmentação Interativa por Amostragem HQ-SAM
LAPIG - Laboratório de Processamento de Imagens e Geoprocessamento (UFG)
"""

import os
import io
import json
import zipfile
import tempfile
import rasterio
from rasterio.warp import transform_bounds
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from .hq_sam_engine import HQSamEngine

app = FastAPI(title="LAPIG HQ-SAM Interactive Paddock Segmenter", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o motor HQ-SAM Singleton
engine = HQSamEngine(model_type="vit_h")

# Schemas Pydantic
class CoordsRequest(BaseModel):
    lat: float
    lon: float
    buffer_m: Optional[float] = 2000.0
    source: Optional[str] = "Satellite"

class PresetRequest(BaseModel):
    preset_id: str

class CarIdRequest(BaseModel):
    car_id: str

class PointPrompt(BaseModel):
    lon: float
    lat: float
    label: int = 1  # 1 = positivo, 0 = negativo

class PredictRequest(BaseModel):
    points: List[PointPrompt]
    multimask: bool = False

class SavePaddockRequest(BaseModel):
    name: Optional[str] = None
    area_ha: float
    confidence_score: float
    geojson: dict
    points: Optional[list] = []

class UpdatePaddockRequest(BaseModel):
    name: Optional[str] = None
    area_ha: float
    confidence_score: float
    geojson: dict
    points: Optional[list] = []


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model": "HQ-SAM ViT-H",
        "device": engine.sam.device if engine.sam else "unknown",
        "has_active_roi": engine.image_path is not None,
        "active_roi_id": engine.active_roi_id,
        "active_roi_name": engine.active_roi_name,
        "saved_paddocks_count": len(engine.saved_paddocks)
    }

@app.get("/api/progress")
def get_progress():
    return engine.progress

@app.get("/api/presets")
def get_presets():
    return engine.get_presets()

@app.post("/api/roi/preset")
def load_preset(req: PresetRequest):
    try:
        res = engine.load_preset(req.preset_id)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/roi/coords")
def load_coords(req: CoordsRequest):
    try:
        res = engine.load_coordinates(lat=req.lat, lon=req.lon, buffer_m=req.buffer_m, source=req.source)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/roi/car_id")
def load_car_id(req: CarIdRequest):
    try:
        res = engine.load_car_id(req.car_id.strip())
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/roi/upload")
async def upload_roi(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".gpkg", ".geojson", ".json", ".zip"]:
            raise HTTPException(status_code=400, detail="Formato não suportado. Use .gpkg, .geojson ou .zip com Shapefile.")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        import geopandas as gpd
        gdf = gpd.read_file(tmp_path)
        if gdf.empty:
            raise ValueError("O arquivo vetorial enviado não possui geometrias.")

        gdf_4326 = gdf.to_crs("EPSG:4326")
        bounds = gdf_4326.total_bounds
        c_lon = (bounds[0] + bounds[2]) / 2
        c_lat = (bounds[1] + bounds[3]) / 2

        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        buffer_ratio = 0.15
        bbox = [
            bounds[0] - width * buffer_ratio,
            bounds[1] - height * buffer_ratio,
            bounds[2] + width * buffer_ratio,
            bounds[3] + height * buffer_ratio
        ]

        engine.clear_session()
        engine.active_roi_id = f"upload_{os.path.basename(file.filename)[:12]}"
        engine.active_roi_name = file.filename
        engine.car_geojson = json.loads(gdf_4326.to_json())['features'][0]

        engine.set_progress(5, "Processando arquivo vetorial...", f"Arquivo: {file.filename} (calculando extensão territorial)...")
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        tif_path = os.path.join(cache_dir, f"s2_{engine.active_roi_id}.tif")
        geom_shp = gdf_4326.geometry.iloc[0]
        engine.acquire_sentinel2_image(
            roi_geometry_or_point=geom_shp,
            output_tif_path=tif_path,
            buffer_m=500,
            fallback_bbox=bbox,
            year=2024
        )

        engine.image_path = tif_path
        with rasterio.open(tif_path) as src:
            engine.image_crs = str(src.crs)
            engine.image_bounds_wgs84 = list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))

        engine.set_progress(88, "Gerando camada espectral...", "Normalizando contraste RGB para visualização no mapa...")
        engine.overlay_png_path = os.path.join(cache_dir, f"overlay_{engine.active_roi_id}.png")
        engine._generate_overlay_png(tif_path, engine.overlay_png_path)

        if engine.sam is not None:
            engine.set_progress(92, "Calculando embeddings na GPU...", "Processando tensores no ViT-H (RTX 2080 Ti)...")
            engine.sam.set_image(tif_path)

        engine.set_progress(100, "Concluído!", "Vetor e Sentinel-2 carregados com sucesso!", status="done")
        os.remove(tmp_path)

        return {
            "roi_id": engine.active_roi_id,
            "roi_name": engine.active_roi_name,
            "bounds_wgs84": engine.image_bounds_wgs84,
            "center": [c_lat, c_lon],
            "car_geojson": engine.car_geojson,
            "has_overlay": True
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/overlay")
def get_overlay_image():
    if not engine.overlay_png_path or not os.path.exists(engine.overlay_png_path):
        raise HTTPException(status_code=404, detail="Nenhum overlay disponível")
    return FileResponse(engine.overlay_png_path, media_type="image/png")

@app.post("/api/predict")
def predict_paddock(req: PredictRequest):
    try:
        points = [{"lon": p.lon, "lat": p.lat, "label": p.label} for p in req.points]
        results = engine.predict_prompts(points, multimask=req.multimask)
        return {
            "status": "success",
            "candidates": results,
            "best_mask": results[0] if results else None,
            "points": points
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/paddock/save")
def save_paddock(req: SavePaddockRequest):
    rec = engine.save_paddock(req.dict())
    return {
        "status": "success",
        "paddock": rec,
        "total_paddocks": len(engine.saved_paddocks)
    }

@app.put("/api/paddock/{paddock_id}")
def update_paddock(paddock_id: str, req: UpdatePaddockRequest):
    try:
        p = engine.update_paddock(paddock_id, req.dict())
        return {"status": "success", "paddock": p}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/paddock/{paddock_id}")
def delete_paddock(paddock_id: str):
    count = engine.delete_paddock(paddock_id)
    return {"status": "deleted", "remaining": count}

@app.post("/api/paddocks/clear")
def clear_paddocks():
    engine.saved_paddocks = []
    engine.current_prompt_points = []
    return {"status": "cleared"}

@app.get("/api/paddocks")
def get_paddocks():
    return {
        "paddocks": engine.saved_paddocks,
        "total_area_ha": round(sum(p['area_ha'] for p in engine.saved_paddocks), 2),
        "count": len(engine.saved_paddocks)
    }

@app.get("/api/export/geojson")
def export_geojson():
    features = []
    for p in engine.saved_paddocks:
        features.append({
            "type": "Feature",
            "properties": {
                "id": p["id"],
                "name": p["name"],
                "area_ha": p["area_ha"],
                "confidence_score": p["confidence_score"]
            },
            "geometry": p["geojson"]
        })
    collection = {
        "type": "FeatureCollection",
        "features": features
    }
    content = json.dumps(collection, indent=2, ensure_ascii=False)
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="application/geo+json",
        headers={"Content-Disposition": f"attachment; filename=piquetes_hq_sam_{engine.active_roi_id or 'export'}.geojson"}
    )

# Monta arquivos estáticos do Frontend
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
