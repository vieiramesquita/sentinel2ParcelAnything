"""
FastAPI Server for HQ-SAM Prompt-Based Interactive Parcel Segmentation
LAPIG - Image Processing and Geoprocessing Laboratory (UFG)
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

app = FastAPI(title="LAPIG HQ-SAM Interactive Parcel Segmenter", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize HQ-SAM Engine Singleton
engine = HQSamEngine(model_type="vit_h")

# Pydantic Schemas
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
    label: int = 1  # 1 = positive (foreground), 0 = negative (background)

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
        if ext not in [".kml", ".kmz", ".gpkg", ".geojson", ".json", ".zip", ".shp"]:
            raise HTTPException(status_code=400, detail="Unsupported format. Use .kml, .kmz, .gpkg, .geojson or .zip with Shapefile.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        import geopandas as gpd
        import shapely
        from shapely.ops import unary_union

        target_path = tmp_path
        extracted_temp_dir = None

        # 1. KMZ handling (compressed archive containing internal KML)
        if ext == ".kmz":
            with zipfile.ZipFile(tmp_path, 'r') as z:
                kml_names = [f for f in z.namelist() if f.lower().endswith('.kml')]
                if not kml_names:
                    raise ValueError("No .kml file found inside the uploaded KMZ archive.")
                target_kml = "doc.kml" if "doc.kml" in kml_names else kml_names[0]
                extracted_temp_dir = tempfile.mkdtemp()
                target_path = z.extract(target_kml, extracted_temp_dir)
                ext = ".kml"

        # 2. Vector file reading via GeoPandas
        if ext == ".kml":
            try:
                import pyogrio
                layers = pyogrio.list_layers(target_path)
                gdfs = []
                for layer_name in layers[:, 0]:
                    try:
                        sub = gpd.read_file(target_path, layer=layer_name)
                        if not sub.empty and any(sub.geometry.notnull()):
                            gdfs.append(sub)
                    except Exception:
                        pass
                if gdfs:
                    import pandas as pd
                    gdf = pd.concat(gdfs, ignore_index=True)
                    gdf = gpd.GeoDataFrame(gdf, crs=gdfs[0].crs or "EPSG:4326")
                else:
                    gdf = gpd.read_file(target_path)
            except Exception:
                gdf = gpd.read_file(target_path)
        else:
            gdf = gpd.read_file(target_path)

        if gdf.empty:
            raise ValueError("The uploaded vector file contains no valid features or geometries.")

        # Ensure WGS84 CRS (standard for KML/KMZ and GeoJSON)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf_4326 = gdf.to_crs("EPSG:4326")
        gdf_4326 = gdf_4326[gdf_4326.geometry.notnull() & ~gdf_4326.geometry.is_empty]

        if gdf_4326.empty:
            raise ValueError("No valid geometry found in the uploaded file.")

        # Unify geometries (handles multiple placemarks/polygons inside vector file)
        union_geom = unary_union(list(gdf_4326.geometry))
        if union_geom.geom_type in ['LineString', 'MultiLineString']:
            union_geom = union_geom.convex_hull
        elif union_geom.geom_type == 'GeometryCollection':
            polys = [g for g in union_geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
            if polys:
                union_geom = unary_union(polys)
            else:
                union_geom = union_geom.convex_hull

        # Force 2D geometry (strips Z dimension for complete GEE / GeoJSON compatibility)
        import shapely.ops
        union_geom = shapely.ops.transform(lambda x, y, *args: (x, y), union_geom)

        bounds = union_geom.bounds
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
        engine.car_geojson = {
            "type": "Feature",
            "properties": {"name": file.filename},
            "geometry": shapely.geometry.mapping(union_geom)
        }

        engine.set_progress(5, "Processing vector file...", f"File: {file.filename} (calculating territorial bounds)...")
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        tif_path = os.path.join(cache_dir, f"s2_{engine.active_roi_id}.tif")
        geom_shp = union_geom
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

        engine.set_progress(88, "Generating spectral overlay...", "Normalizing RGB contrast for map rendering...")
        engine.overlay_png_path = os.path.join(cache_dir, f"overlay_{engine.active_roi_id}.png")
        engine._generate_overlay_png(tif_path, engine.overlay_png_path)

        if engine.sam is not None:
            engine.set_progress(92, "Computing embeddings on GPU...", "Processing image tensors on ViT-H (RTX 2080 Ti)...")
            engine.sam.set_image(tif_path)

        engine.set_progress(100, "Completed!", "Vector boundary and Sentinel-2 imagery successfully loaded!", status="done")

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
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        if extracted_temp_dir and os.path.exists(extracted_temp_dir):
            import shutil
            shutil.rmtree(extracted_temp_dir, ignore_errors=True)

@app.get("/api/overlay")
def get_overlay_image():
    if not engine.overlay_png_path or not os.path.exists(engine.overlay_png_path):
        raise HTTPException(status_code=404, detail="No overlay image available")
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

@app.post("/api/paddocks/regularize")
def regularize_paddocks(straight_tol: float = 18.0):
    try:
        res = engine.regularize_paddocks_topology(straight_tol=straight_tol)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/export/geojson")
def export_geojson(regularize: bool = True):
    if regularize and len(engine.saved_paddocks) > 1:
        try:
            engine.regularize_paddocks_topology()
        except Exception as e:
            print(f"Notice during automatic regularization prior to export: {e}")

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
        headers={"Content-Disposition": f"attachment; filename=parcels_hq_sam_{engine.active_roi_id or 'export'}.geojson"}
    )

# Mount static frontend files
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
