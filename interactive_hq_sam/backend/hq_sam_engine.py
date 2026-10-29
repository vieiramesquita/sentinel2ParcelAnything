"""
Interactive Segmentation Engine with HQ-SAM (Segment Anything High Quality)
LAPIG - Image Processing and Geoprocessing Laboratory (UFG)

Enables interactive parcel segmentation via prompt sampling (user click points)
utilizing the ViT-H architecture with NVIDIA GPU acceleration.
"""

import os
import sys
import io
import json
import glob
import uuid
import tempfile
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_bounds, transform
import shapely
from shapely.geometry import shape, mapping, MultiPolygon, Polygon, Point, LineString, MultiLineString
from shapely.ops import transform as shp_transform, unary_union, polygonize, linemerge
import pyproj
from PIL import Image

try:
    from samgeo.hq_sam import SamGeo
    from samgeo.common import tms_to_geotiff
except ImportError:
    SamGeo = None
    tms_to_geotiff = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

import math
import concurrent.futures
import requests

# Integration with official LAPIG s2_mosaic module
try:
    import ee
    from lapig_parcel_segmentation.s2_mosaic import initialize_earth_engine, get_sentinel2_ndvi_mosaic
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        import ee
        from lapig_parcel_segmentation.s2_mosaic import initialize_earth_engine, get_sentinel2_ndvi_mosaic
    except ImportError:
        ee = None
        initialize_earth_engine = None
        get_sentinel2_ndvi_mosaic = None

SENTINEL2_TMS_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg"
SENTINEL2_FALLBACK_TMS = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2022_3857/default/g/{z}/{y}/{x}.jpg"

class HQSamEngine:
    def __init__(self, model_type="vit_h", device=None):
        self.model_type = model_type
        self.device = device
        self.sam = None
        self._init_model()

        # Active Session State
        self.active_roi_id = None
        self.active_roi_name = "None"
        self.image_path = None
        self.image_crs = None
        self.image_bounds_wgs84 = None  # [min_lon, min_lat, max_lon, max_lat]
        self.overlay_png_path = None
        self.car_geojson = None
        self.saved_paddocks = []  # List of dictionaries: {id, name, area_ha, score, geojson}
        self.current_prompt_points = []  # [[lon, lat, label], ...]
        self.last_candidate_masks = None

        # Real-time progress bar state
        self.progress = {
            "percent": 0,
            "status": "idle",
            "title": "Waiting",
            "message": "",
            "detail": ""
        }

    def set_progress(self, percent, title, message="", detail="", status="running"):
        self.progress = {
            "percent": max(0, min(100, int(percent))),
            "status": status,
            "title": title,
            "message": message,
            "detail": detail
        }

    def download_sentinel2_with_progress(self, tif_path, bbox, zoom=14):
        """
        Downloads Sentinel-2 Cloudless mosaic updating progress tile-by-tile.
        bbox: [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
        """
        west, south, east, north = bbox
        self.set_progress(5, "Calculating Sentinel-2 orbital grid...", f"Bbox: [{west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}]", "Grid partition")

        def deg2num(lat, lon, z):
            lat_r = math.radians(lat)
            n = 2**z
            xtile = (lon + 180) / 360 * n
            ytile = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
            return (xtile, ytile)

        x0, y0 = deg2num(north, west, zoom)
        x1, y1 = deg2num(south, east, zoom)
        min_x, max_x = int(math.floor(min(x0, x1))), int(math.ceil(max(x0, x1)))
        min_y, max_y = int(math.floor(min(y0, y1))), int(math.ceil(max(y0, y1)))

        corners = [(x, y) for x in range(min_x, max_x) for y in range(min_y, max_y)]
        total_tiles = len(corners)

        self.set_progress(10, f"Downloading Sentinel-2 tiles (0/{total_tiles})...", "Starting parallel download of 10m tiles...", f"Zoom {zoom}")

        session = requests.Session()
        tiles_data = {}
        downloaded = 0

        def fetch_tile(xy):
            nonlocal downloaded
            x, y = xy
            url = SENTINEL2_TMS_URL.format(z=zoom, x=x, y=y)
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 200:
                    img = Image.open(io.BytesIO(r.content)).convert("RGB")
                    tiles_data[(x, y)] = img
                else:
                    # Fallback
                    r_fb = session.get(SENTINEL2_FALLBACK_TMS.format(z=zoom, x=x, y=y), timeout=15)
                    if r_fb.status_code == 200:
                        img = Image.open(io.BytesIO(r_fb.content)).convert("RGB")
                        tiles_data[(x, y)] = img
            except Exception as e:
                print(f"Notice downloading tile ({x}, {y}): {e}")

            downloaded += 1
            pct = 10 + int((downloaded / max(1, total_tiles)) * 60)  # 10% to 70%
            self.set_progress(
                pct,
                f"Downloading Sentinel-2 tiles ({downloaded}/{total_tiles})...",
                f"Received tile data ({x}, {y})",
                f"Progress: {pct}%"
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(fetch_tile, corners))

        self.set_progress(72, "Assembling Sentinel-2 mosaic...", "Stitching tiles and generating georeferenced GeoTIFF...", "RGB Composition")

        tile_w, tile_h = 256, 256
        cols = max_x - min_x
        rows = max_y - min_y
        big_img = Image.new("RGB", (cols * tile_w, rows * tile_h))
        for (x, y), img in tiles_data.items():
            dx = (x - min_x) * tile_w
            dy = (y - min_y) * tile_h
            big_img.paste(img, (dx, dy))

        frac_x0 = (min(x0, x1) - min_x) * tile_w
        frac_y0 = (min(y0, y1) - min_y) * tile_h
        frac_w = abs(x1 - x0) * tile_w
        frac_h = abs(y1 - y0) * tile_h
        cropped_img = big_img.crop((round(frac_x0), round(frac_y0), round(frac_x0 + frac_w), round(frac_y0 + frac_h)))

        self.set_progress(80, "Writing GeoTIFF file...", f"Dimensions: {cropped_img.width}x{cropped_img.height} pixels (3 RGB bands)", "EPSG:3857 Projection")

        EARTH_RADIUS = 6378137.0
        def from4326_to3857(lat, lon):
            return (math.radians(lon) * EARTH_RADIUS, math.log(math.tan(math.radians(45 + lat / 2.0))) * EARTH_RADIUS)

        xp0, yp0 = from4326_to3857(south, west)
        xp1, yp1 = from4326_to3857(north, east)
        min_xp, max_xp = min(xp0, xp1), max(xp0, xp1)
        min_yp, max_yp = min(yp0, yp1), max(yp0, yp1)
        pixel_w = (max_xp - min_xp) / max(1, cropped_img.width)
        pixel_h = (max_yp - min_yp) / max(1, cropped_img.height)

        arr = np.array(cropped_img)
        with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=cropped_img.height,
            width=cropped_img.width,
            count=3,
            dtype=arr.dtype,
            crs='EPSG:3857',
            transform=rasterio.transform.from_origin(min_xp, max_yp, pixel_w, pixel_h),
            compress='deflate'
        ) as dst:
            for i in range(3):
                dst.write(arr[:, :, i], i + 1)

        self.set_progress(85, "Sentinel-2 GeoTIFF written!", "Preparing spectral visualization overlay...", "Done")

    def _init_model(self):
        """Initializes the HQ-SAM model on GPU once."""
        if SamGeo is None:
            print("Notice: samgeo.hq_sam not found.")
            return

        print(f"Initializing HQ-SAM (Architecture: {self.model_type})...")
        try:
            self.sam = SamGeo(model_type=self.model_type, automatic=False, device=self.device)
            print("HQ-SAM initialized successfully on GPU!")
        except Exception as e:
            print(f"[-] Error initializing HQ-SAM: {e}")

    def get_presets(self, results_dir="results", gpkg_path="car_goias_area_imovel_epsg4674_hottest.gpkg"):
        """Returns list of pre-processed demonstration farms ready for immediate interaction."""
        presets = []
        car_dict = {}
        if os.path.exists(gpkg_path) and gpd is not None:
            try:
                gdf = gpd.read_file(gpkg_path)
                for _, row in gdf.iterrows():
                    car_dict[str(row['cod_imovel'])] = {
                        'municipio': str(row.get('municipio', 'Goiás')),
                        'num_area': float(row.get('num_area', 0.0))
                    }
            except Exception:
                pass

        # Known presets
        car_dict['MS-5006309-8DC93A48C2424FD7B3EDF235F8B531E1'] = {'municipio': 'Paranaíba (MS)', 'num_area': 1085.22}
        car_dict['GO-5211008-46C08C6235D74D36BE07B5384A17B156'] = {'municipio': 'Itapirapuã (GO)', 'num_area': 193.59}

        if os.path.exists(results_dir):
            for folder in sorted(os.listdir(results_dir)):
                fpath = os.path.join(results_dir, folder)
                s2_tif = os.path.join(fpath, "s2_ndvi_rgb.tif")
                if os.path.isdir(fpath) and os.path.exists(s2_tif):
                    info = car_dict.get(folder, {})
                    mun = info.get('municipio', 'Goiás')
                    area = info.get('num_area', 0.0)
                    presets.append({
                        "id": folder,
                        "title": f"{mun} ({area:,.1f} ha)" if area else mun,
                        "municipio": mun,
                        "car_area_ha": area,
                        "tif_path": s2_tif
                    })
        return presets

    def _generate_overlay_png(self, tif_path, output_png_path):
        """Generates a normalized PNG (RGBA or RGB) for Leaflet map overlay."""
        with rasterio.open(tif_path) as src:
            arr = src.read()
            if arr.shape[0] >= 4:
                # 4 bands (RGBA from s2_mosaic / geemap)
                rgba = np.transpose(arr[:4], (1, 2, 0))
                if rgba.dtype != np.uint8:
                    p2, p98 = np.percentile(rgba[:, :, :3], (2, 98))
                    rgb_scaled = np.clip((rgba[:, :, :3] - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
                    alpha = rgba[:, :, 3].astype(np.uint8)
                    rgba = np.dstack([rgb_scaled, alpha])
                img = Image.fromarray(rgba, "RGBA")
            elif arr.shape[0] >= 3:
                # 3 RGB bands
                rgb = np.transpose(arr[:3], (1, 2, 0))
                if rgb.dtype != np.uint8:
                    p2, p98 = np.percentile(rgb, (2, 98))
                    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
                img = Image.fromarray(rgb, "RGB")
            else:
                band = arr[0]
                if band.dtype != np.uint8:
                    p2, p98 = np.percentile(band, (2, 98))
                    band = np.clip((band - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
                img = Image.fromarray(np.repeat(band[:, :, None], 3, axis=2), "RGB")

            img.save(output_png_path, "PNG")

    def acquire_sentinel2_image(self, roi_geometry_or_point, output_tif_path, buffer_m=1000, fallback_bbox=None, year=2024):
        """
        Acquires Sentinel-2 imagery prioritizing LAPIG's s2_mosaic (Google Earth Engine)
        with multi-temporal NDVI statistical composite and Cloud Score Plus (CS+) mask.
        In case of Earth Engine unavailability, falls back to tile-by-tile download.
        """
        success = False
        if get_sentinel2_ndvi_mosaic is not None:
            try:
                self.set_progress(15, "Connecting to Google Earth Engine...", "Initializing ee-vieiramesquita project...", "s2_mosaic")
                initialize_earth_engine(project='ee-vieiramesquita')
                import ee

                if isinstance(roi_geometry_or_point, (tuple, list)):
                    lat, lon = roi_geometry_or_point
                    ee_geom = ee.Geometry.Point([lon, lat])
                    roi_ee = ee.Feature(ee_geom)
                    buf_meters = int(buffer_m)
                elif hasattr(roi_geometry_or_point, '__geo_interface__'):
                    import shapely.ops
                    geom_2d = shapely.ops.transform(lambda x, y, *args: (x, y), roi_geometry_or_point)
                    ee_geom = ee.Geometry(geom_2d.__geo_interface__)
                    roi_ee = ee.Feature(ee_geom)
                    buf_meters = max(200, int(buffer_m))
                elif isinstance(roi_geometry_or_point, dict):
                    ee_geom = ee.Geometry(roi_geometry_or_point)
                    roi_ee = ee.Feature(ee_geom)
                    buf_meters = max(200, int(buffer_m))
                else:
                    roi_ee = ee.Feature(roi_geometry_or_point)
                    buf_meters = int(buffer_m)

                self.set_progress(35, "Querying Sentinel-2 catalog (GEE)...", f"Multi-temporal NDVI {year} mosaic with Cloud Score Plus (cs_cdf >= 0.65)...", "s2_mosaic")
                print(f"Executing s2_mosaic via Earth Engine (year {year}, buffer {buf_meters}m)...")

                _, out_path, _ = get_sentinel2_ndvi_mosaic(
                    roi_ee=roi_ee,
                    year=year,
                    start_month=3,
                    end_month=10,
                    buffer_meters=buf_meters,
                    output_path=output_tif_path,
                    ee_project='ee-vieiramesquita'
                )

                if os.path.exists(output_tif_path) and os.path.getsize(output_tif_path) > 1000:
                    self.set_progress(78, "Sentinel-2 mosaic generated successfully!", f"File: {os.path.basename(output_tif_path)} ({os.path.getsize(output_tif_path)/1024:.1f} KB)", "s2_mosaic")
                    print(f"Sentinel-2 image acquired via s2_mosaic: {output_tif_path}")
                    success = True
            except Exception as e:
                print(f"Warning obtaining image via s2_mosaic: {e}. Triggering fallback...")

        if not success:
            if fallback_bbox is None:
                if isinstance(roi_geometry_or_point, (tuple, list)):
                    lat, lon = roi_geometry_or_point
                    delta_lat = buffer_m / 111320.0
                    delta_lon = buffer_m / (111320.0 * np.cos(np.radians(lat)))
                    fallback_bbox = [lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat]
                else:
                    fallback_bbox = [-49.5, -16.5, -49.4, -16.4]

            self.set_progress(20, "Obtaining Sentinel-2 via orbital tile service...", f"Downloading tiles for bbox {fallback_bbox}...", "Fallback TMS")
            self.download_sentinel2_with_progress(output_tif_path, bbox=fallback_bbox, zoom=14)

    def load_preset(self, preset_id, results_dir="results", gpkg_path="car_goias_area_imovel_epsg4674_hottest.gpkg"):
        """Loads a pre-processed farm into the interactive session."""
        self.set_progress(10, "Loading Pre-processed Farm...", f"Fetching Sentinel-2 mosaic for {preset_id[:16]}...", "Reading files")
        folder = os.path.join(results_dir, preset_id)
        s2_tif = os.path.join(folder, "s2_ndvi_rgb.tif")
        if not os.path.exists(s2_tif):
            self.set_progress(0, "Error loading", f"Image not found: {s2_tif}", status="error")
            raise FileNotFoundError(f"Sentinel-2 image not found at: {s2_tif}")

        # Clear session
        self.clear_session()
        self.active_roi_id = preset_id
        self.image_path = s2_tif

        self.set_progress(30, "Reading spatial metadata...", "Extracting CRS and bounding box in WGS84...", "Projections")
        with rasterio.open(s2_tif) as src:
            self.image_crs = str(src.crs)
            self.image_bounds_wgs84 = list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))

        # Generate temporary overlay PNG
        self.set_progress(50, "Generating spectral overlay...", "Normalizing RGB contrast for map rendering...", "Visualization")
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.overlay_png_path = os.path.join(cache_dir, f"overlay_{preset_id}.png")
        self._generate_overlay_png(s2_tif, self.overlay_png_path)

        # Retrieve CAR geometry
        self.car_geojson = None
        if os.path.exists(gpkg_path) and gpd is not None:
            try:
                gdf = gpd.read_file(gpkg_path)
                match = gdf[gdf['cod_imovel'] == preset_id]
                if not match.empty:
                    self.active_roi_name = str(match.iloc[0].get('municipio', 'Goiás'))
                    self.car_geojson = json.loads(match.to_crs("EPSG:4326").to_json())['features'][0]
            except Exception as e:
                print(f"Notice searching CAR in GPKG: {e}")

        # Load image into HQ-SAM image encoder
        if self.sam is not None:
            self.set_progress(75, "Allocating GPU tensors...", "Running ViT-H image encoder on NVIDIA GPU...", "Neural Network")
            print(f"Loading GeoTIFF into HQ-SAM image encoder: {s2_tif}...")
            self.sam.set_image(s2_tif)
            print("Image embeddings computed on GPU!")

        center_lon = (self.image_bounds_wgs84[0] + self.image_bounds_wgs84[2]) / 2
        center_lat = (self.image_bounds_wgs84[1] + self.image_bounds_wgs84[3]) / 2

        self.set_progress(100, "Completed!", "Property loaded and ready for segmentation!", "Ready", status="done")

        return {
            "roi_id": self.active_roi_id,
            "roi_name": self.active_roi_name,
            "bounds_wgs84": self.image_bounds_wgs84,
            "center": [center_lat, center_lon],
            "car_geojson": self.car_geojson,
            "has_overlay": True
        }

    def _find_preset_for_point(self, lat, lon):
        """Checks if a geographic coordinate falls inside an already processed property."""
        presets = self.get_presets()
        for p in presets:
            tif = p.get('tif_path')
            if tif and os.path.exists(tif):
                try:
                    with rasterio.open(tif) as src:
                        b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                        if b[0] <= lon <= b[2] and b[1] <= lat <= b[3]:
                            return p['id']
                except Exception:
                    pass
        return None

    def load_coordinates(self, lat, lon, buffer_m=2000, **kwargs):
        """
        Loads area of interest from geographic coordinates.
        The HQ-SAM model always processes Sentinel-2 imagery (official standard).
        """
        # If coordinates lie within a pre-processed farm, load directly
        matched_preset = self._find_preset_for_point(lat, lon)
        if matched_preset:
            self.set_progress(15, "Coordinates matched pre-processed property!", f"Reusing Sentinel-2 data from {matched_preset}...")
            print(f"Coordinates match pre-processed farm: {matched_preset}")
            return self.load_preset(matched_preset)

        # Calculate bounding box in WGS84 for fallback
        delta_lat = buffer_m / 111320.0
        delta_lon = buffer_m / (111320.0 * np.cos(np.radians(lat)))
        min_lon = lon - delta_lon
        max_lon = lon + delta_lon
        min_lat = lat - delta_lat
        max_lat = lat + delta_lat

        bbox = [min_lon, min_lat, max_lon, max_lat]

        self.clear_session()
        roi_id = f"custom_s2_{uuid.uuid4().hex[:8]}"
        self.active_roi_id = roi_id
        self.active_roi_name = f"Sentinel-2 ({lat:.4f}, {lon:.4f})"

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        tif_path = os.path.join(cache_dir, f"{roi_id}.tif")

        # Acquire via s2_mosaic
        self.acquire_sentinel2_image(
            roi_geometry_or_point=(lat, lon),
            output_tif_path=tif_path,
            buffer_m=buffer_m,
            fallback_bbox=bbox,
            year=2024
        )

        self.image_path = tif_path
        with rasterio.open(tif_path) as src:
            self.image_crs = str(src.crs)
            self.image_bounds_wgs84 = list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))

        self.set_progress(88, "Generating spectral overlay...", "Normalizing RGB contrast for map rendering...")
        self.overlay_png_path = os.path.join(cache_dir, f"overlay_{roi_id}.png")
        self._generate_overlay_png(tif_path, self.overlay_png_path)

        if self.sam is not None:
            self.set_progress(92, "Computing embeddings on GPU...", "Processing tensors on ViT-H (RTX 2080 Ti)...")
            print("Embedding Sentinel-2 in HQ-SAM encoder...")
            self.sam.set_image(tif_path)
            print("Sentinel-2 ready for prompt-based interactive segmentation!")

        self.set_progress(100, "Completed!", "Area loaded and ready for segmentation!", status="done")

        return {
            "roi_id": self.active_roi_id,
            "roi_name": self.active_roi_name,
            "bounds_wgs84": self.image_bounds_wgs84,
            "center": [lat, lon],
            "car_geojson": None,
            "has_overlay": True
        }

    def load_car_id(self, car_id, gpkg_path="car_goias_area_imovel_epsg4674_hottest.gpkg"):
        """Searches property by CAR code and loads the corresponding Sentinel-2 imagery."""
        self.set_progress(10, "Searching property in CAR...", f"Searching code {car_id[:16]}... in vector database")
        if os.path.exists(gpkg_path) and gpd is not None:
            gdf = gpd.read_file(gpkg_path)
            match = gdf[gdf['cod_imovel'] == car_id]
            if not match.empty:
                match_4326 = match.to_crs("EPSG:4326")
                bounds = match_4326.total_bounds  # minx, miny, maxx, maxy
                c_lon = (bounds[0] + bounds[2]) / 2
                c_lat = (bounds[1] + bounds[3]) / 2

                # If already processed with multi-temporal Sentinel-2, load directly
                results_s2 = os.path.join("results", car_id, "s2_ndvi_rgb.tif")
                if os.path.exists(results_s2):
                    self.set_progress(20, "Property located in pre-processed database!", "Loading multi-temporal Sentinel-2...")
                    return self.load_preset(car_id, gpkg_path=gpkg_path)

                geom_shp = match_4326.geometry.iloc[0]

                # 15% buffer for external fence context
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                bbox = [
                    bounds[0] - width * 0.15,
                    bounds[1] - height * 0.15,
                    bounds[2] + width * 0.15,
                    bounds[3] + height * 0.15
                ]

                self.clear_session()
                roi_id = f"car_s2_{car_id[:16]}"
                self.active_roi_id = car_id
                self.active_roi_name = f"{match.iloc[0].get('municipio', 'CAR')} ({car_id[:18]}...)"
                self.car_geojson = json.loads(match_4326.to_json())['features'][0]

                cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
                os.makedirs(cache_dir, exist_ok=True)
                tif_path = os.path.join(cache_dir, f"{roi_id}.tif")

                # Acquire image via s2_mosaic
                self.acquire_sentinel2_image(
                    roi_geometry_or_point=geom_shp,
                    output_tif_path=tif_path,
                    buffer_m=500,
                    fallback_bbox=bbox,
                    year=2024
                )

                self.image_path = tif_path
                with rasterio.open(tif_path) as src:
                    self.image_crs = str(src.crs)
                    self.image_bounds_wgs84 = list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))

                self.set_progress(88, "Generating spectral overlay...", "Normalizing RGB contrast for map rendering...")
                self.overlay_png_path = os.path.join(cache_dir, f"overlay_{roi_id}.png")
                self._generate_overlay_png(tif_path, self.overlay_png_path)

                if self.sam is not None:
                    self.set_progress(92, "Computing embeddings on GPU...", "Processing tensors on ViT-H...")
                    print("Embedding Sentinel-2 in HQ-SAM encoder...")
                    self.sam.set_image(tif_path)
                    print("Sentinel-2 loaded into GPU memory!")

                self.set_progress(100, "Completed!", "CAR property ready for segmentation!", status="done")

                return {
                    "roi_id": self.active_roi_id,
                    "roi_name": self.active_roi_name,
                    "bounds_wgs84": self.image_bounds_wgs84,
                    "center": [c_lat, c_lon],
                    "car_geojson": self.car_geojson,
                    "has_overlay": True
                }

        self.set_progress(0, "Error locating CAR", f"CAR code {car_id} not found in local database.", status="error")
        raise ValueError(f"CAR code '{car_id}' not found in local database.")

    def predict_prompts(self, points, multimask=False):
        """
        Executes prompt-based inference with sampling points on Sentinel-2 imagery.
        Generates an individual isolated feature per prompt without grouping other parcels.
        """
        if self.sam is None or self.image_path is None:
            raise RuntimeError("No Sentinel-2 image loaded in HQ-SAM.")

        if not points:
            raise ValueError("No prompt points provided.")

        point_coords = [[p['lon'], p['lat']] for p in points]
        point_labels = [int(p.get('label', 1)) for p in points]

        # Save current session points
        self.current_prompt_points = points

        # Execute SAM HQ prediction over Sentinel-2
        masks, scores, logits = self.sam.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            point_crs="EPSG:4326",
            multimask_output=multimask,
            return_results=True
        )

        with rasterio.open(self.image_path) as src:
            src_crs = src.crs
            src_trans = src.transform

        project_to_wgs84 = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True).transform
        project_to_src = pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True).transform

        pos_points = [p for p in points if int(p.get('label', 1)) == 1]
        neg_points = [p for p in points if int(p.get('label', 1)) == 0]

        src_pos_pts = []
        for p in pos_points:
            rx, ry = project_to_src(p['lon'], p['lat'])
            src_pos_pts.append(Point(rx, ry))

        src_neg_pts = []
        for p in neg_points:
            rx, ry = project_to_src(p['lon'], p['lat'])
            src_neg_pts.append(Point(rx, ry))

        if src_pos_pts:
            target_pt = src_pos_pts[0]
        else:
            rx, ry = project_to_src(points[0]['lon'], points[0]['lat'])
            target_pt = Point(rx, ry)

        # Prepare union of already saved parcels in raster CRS for real-time overlap prevention
        saved_src_union = None
        if self.saved_paddocks:
            saved_src_geoms = []
            for sp in self.saved_paddocks:
                if sp.get('geojson'):
                    try:
                        g_wgs = shape(sp['geojson'])
                        g_src = shp_transform(project_to_src, g_wgs)
                        if g_src.is_valid and not g_src.is_empty:
                            saved_src_geoms.append(g_src)
                    except Exception:
                        pass
            if saved_src_geoms:
                saved_src_union = unary_union(saved_src_geoms)

        results = []
        for i in range(len(masks)):
            m = masks[i]
            score = float(scores[i])

            # Vectorize binary mask
            polys = []
            for geom, val in shapes(m.astype(np.uint8), transform=src_trans):
                if val == 1:
                    s = shape(geom)
                    if s.is_valid and s.area > 0:
                        polys.append(s)
                    elif not s.is_valid:
                        s_fixed = s.buffer(0)
                        if s_fixed.area > 0:
                            polys.append(s_fixed)

            if not polys:
                continue

            # Select polygon enclosing highest number of sampled positive points
            best_poly = None
            max_contained = -1
            min_dist = float('inf')

            for p in polys:
                contained = sum(1 for pt in src_pos_pts if p.contains(pt) or p.touches(pt))
                dist = p.distance(target_pt)

                if contained > max_contained:
                    max_contained = contained
                    min_dist = dist
                    best_poly = p
                elif contained == max_contained and dist < min_dist:
                    min_dist = dist
                    best_poly = p

            selected_poly = best_poly or polys[0]

            # If MultiPolygon, select component with most positive points or closest
            if isinstance(selected_poly, MultiPolygon):
                parts = list(selected_poly.geoms)
                best_part = None
                max_part_pts = -1
                min_part_dist = float('inf')
                for part in parts:
                    pts_in_part = sum(1 for pt in src_pos_pts if part.contains(pt) or part.touches(pt))
                    dist = part.distance(target_pt)
                    if pts_in_part > max_part_pts:
                        max_part_pts = pts_in_part
                        min_part_dist = dist
                        best_part = part
                    elif pts_in_part == max_part_pts and dist < min_part_dist:
                        min_part_dist = dist
                        best_part = part
                selected_poly = best_part or parts[0]

            # 1. Perpendicular Bisector Cut (Eq. 1 in scientific paper)
            # When negative point exists, trace normal hyperplane at midpoint and retain positive half-plane
            if src_neg_pts and src_pos_pts:
                for neg_pt in src_neg_pts:
                    if selected_poly.contains(neg_pt) or selected_poly.distance(neg_pt) < 15.0:
                        closest_pos = min(src_pos_pts, key=lambda pt: pt.distance(neg_pt))
                        selected_poly = self._clip_polygon_by_bisector(selected_poly, closest_pos, neg_pt)

            # 2. Real-Time Zero-Overlap Enforcement with Existing Parcels (0% Overlap)
            if saved_src_union is not None and selected_poly.intersects(saved_src_union):
                diff = selected_poly.difference(saved_src_union)
                if not diff.is_empty:
                    if isinstance(diff, MultiPolygon):
                        best_part = None
                        max_c = -1
                        min_d = float('inf')
                        for part in diff.geoms:
                            c = sum(1 for pt in src_pos_pts if part.contains(pt) or part.touches(pt))
                            d = part.distance(target_pt)
                            if c > max_c:
                                max_c = c
                                min_d = d
                                best_part = part
                            elif c == max_c and d < min_d:
                                min_d = d
                                best_part = part
                        diff = best_part or diff.geoms[0]
                    if isinstance(diff, Polygon) and diff.area > 0:
                        selected_poly = Polygon(diff.exterior)

            # Calculate area in hectares
            area_ha = 0.0
            if src_crs.is_projected:
                area_ha = round(selected_poly.area / 10000.0, 2)
            else:
                area_ha = self._calc_area_ha(selected_poly, is_wgs84=True)

            # Convert to WGS84 GeoJSON
            poly_wgs84 = shp_transform(project_to_wgs84, selected_poly)
            # Light simplification preserving topology
            poly_wgs84_simplified = poly_wgs84.simplify(0.00002, preserve_topology=True)

            results.append({
                "mask_index": i,
                "confidence_score": round(score * 100, 2),
                "area_ha": area_ha,
                "geojson": mapping(poly_wgs84_simplified)
            })

        # Sort descending by confidence score
        results.sort(key=lambda x: x['confidence_score'], reverse=True)
        self.last_candidate_masks = results
        return results

    @staticmethod
    def _clip_polygon_by_bisector(polygon, pos_pt, neg_pt):
        """
        Clips polygon by perpendicular bisector hyperplane between pos_pt and neg_pt (Eq. 1 in paper).
        Retains the half-plane containing pos_pt.
        """
        p_pos = np.array([pos_pt.x, pos_pt.y])
        p_neg = np.array([neg_pt.x, neg_pt.y])
        m = (p_pos + p_neg) / 2.0
        n = p_pos - p_neg
        dist = np.linalg.norm(n)
        if dist < 1e-6:
            return polygon

        u_perp = np.array([-n[1], n[0]]) / dist
        u_norm = n / dist

        minx, miny, maxx, maxy = polygon.bounds
        diag = np.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2) + 1000.0

        pt_left = m - u_perp * diag
        pt_right = m + u_perp * diag
        pt_top_right = pt_right + u_norm * diag
        pt_top_left = pt_left + u_norm * diag

        half_plane = Polygon([pt_left, pt_right, pt_top_right, pt_top_left])
        if not polygon.intersects(half_plane):
            return polygon

        clipped = polygon.intersection(half_plane)
        if clipped.is_empty:
            return polygon

        if isinstance(clipped, MultiPolygon):
            for comp in clipped.geoms:
                if comp.contains(pos_pt) or comp.touches(pos_pt):
                    return comp
            return min(clipped.geoms, key=lambda c: c.distance(pos_pt))
        elif isinstance(clipped, Polygon):
            return clipped
        return polygon

    @staticmethod
    def _calc_area_ha(geom, is_wgs84=True):
        """Calculates accurate surface area in hectares."""
        if not is_wgs84:
            return round(geom.area / 10000.0, 2)
        c_lat = (geom.bounds[1] + geom.bounds[3]) / 2.0
        cos_lat = math.cos(math.radians(c_lat))
        area_m2 = geom.area * (111320.0 * cos_lat) * 110540.0
        return round(area_m2 / 10000.0, 2)

    def save_paddock(self, paddock_data, prevent_overlap=True):
        """Saves a confirmed parcel ensuring watertight topology (0% overlap with existing parcels)."""
        pid = paddock_data.get('id') or str(uuid.uuid4())[:8]
        name = paddock_data.get('name') or f"Parcel {len(self.saved_paddocks) + 1}"

        poly_shp = shape(paddock_data.get('geojson'))
        if not poly_shp.is_valid:
            poly_shp = poly_shp.buffer(0)

        # Rigorous mutual overlap prevention
        if prevent_overlap and self.saved_paddocks:
            saved_geoms = []
            for sp in self.saved_paddocks:
                if sp.get('geojson'):
                    try:
                        g = shape(sp['geojson'])
                        if g.is_valid and not g.is_empty:
                            saved_geoms.append(g)
                    except Exception:
                        pass
            if saved_geoms:
                saved_union = unary_union(saved_geoms)
                if poly_shp.intersects(saved_union):
                    diff = poly_shp.difference(saved_union)
                    if not diff.is_empty:
                        if isinstance(diff, MultiPolygon):
                            diff = max(diff.geoms, key=lambda g: g.area)
                        if isinstance(diff, Polygon) and diff.area > 0:
                            poly_shp = Polygon(diff.exterior)

        poly_shp = poly_shp.simplify(0.00002, preserve_topology=True)
        area_ha = self._calc_area_ha(poly_shp, is_wgs84=True)

        record = {
            "id": pid,
            "name": name,
            "area_ha": area_ha,
            "confidence_score": paddock_data.get('confidence_score', 100.0),
            "geojson": mapping(poly_shp),
            "points": paddock_data.get('points', [])
        }
        self.saved_paddocks.append(record)
        self.current_prompt_points = []
        return record

    def update_paddock(self, paddock_id, paddock_data):
        """Updates geometry and metrics of an existing parcel ensuring clean topology."""
        for p in self.saved_paddocks:
            if p['id'] == paddock_id:
                if 'name' in paddock_data and paddock_data['name']:
                    p['name'] = paddock_data['name']
                if 'confidence_score' in paddock_data:
                    p['confidence_score'] = paddock_data['confidence_score']
                if 'points' in paddock_data:
                    p['points'] = paddock_data['points']
                if 'geojson' in paddock_data and paddock_data['geojson']:
                    poly_shp = shape(paddock_data['geojson'])
                    if not poly_shp.is_valid:
                        poly_shp = poly_shp.buffer(0)

                    # Subtract other parcels (excluding itself)
                    other_geoms = [shape(sp['geojson']) for sp in self.saved_paddocks if sp['id'] != paddock_id and sp.get('geojson')]
                    if other_geoms:
                        other_union = unary_union(other_geoms)
                        if poly_shp.intersects(other_union):
                            diff = poly_shp.difference(other_union)
                            if not diff.is_empty:
                                if isinstance(diff, MultiPolygon):
                                    diff = max(diff.geoms, key=lambda g: g.area)
                                if isinstance(diff, Polygon):
                                    poly_shp = Polygon(diff.exterior)

                    poly_shp = poly_shp.simplify(0.00002, preserve_topology=True)
                    p['geojson'] = mapping(poly_shp)
                    p['area_ha'] = self._calc_area_ha(poly_shp, is_wgs84=True)
                return p
        raise ValueError(f"Parcel {paddock_id} not found.")

    def regularize_paddocks_topology(self, straight_tol=18.0):
        """
        Reconstructs all session parcels into a Watertight Planar Graph.
        Rectifies shared fences via Hausdorff distance, eliminates gaps/slivers, and guarantees 0% mutual overlap.
        """
        if not self.saved_paddocks:
            return {"status": "no_paddocks", "count": 0, "paddocks": []}

        if len(self.saved_paddocks) == 1:
            p = shape(self.saved_paddocks[0]['geojson'])
            p_clean = p.buffer(0).simplify(0.00003, preserve_topology=True)
            self.saved_paddocks[0]['geojson'] = mapping(p_clean)
            self.saved_paddocks[0]['area_ha'] = self._calc_area_ha(p_clean, is_wgs84=True)
            return {"status": "success", "count": 1, "paddocks": self.saved_paddocks, "total_area_ha": self.saved_paddocks[0]['area_ha']}

        # 1. Load geometries into GeoDataFrame
        records = []
        for p in self.saved_paddocks:
            geom = shape(p['geojson'])
            if not geom.is_valid:
                geom = geom.buffer(0)
            records.append({'id': p['id'], 'name': p['name'], 'score': p.get('confidence_score', 100.0), 'geometry': geom})

        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)

        # 2. Planar Graph Decomposition
        boundaries = [poly.boundary for poly in gdf_utm.geometry if poly.is_valid and not poly.is_empty]
        noded = shapely.node(unary_union(boundaries))
        merged = linemerge(noded)
        lines = merged.geoms if isinstance(merged, MultiLineString) else [merged]

        # 3. Rectify Shared Fence Lines
        straight_lines = []
        for line in lines:
            if line.length < 5.0:
                continue
            coords = np.array(line.coords)
            chord = LineString([coords[0], coords[-1]])
            h_dist = line.hausdorff_distance(chord)
            if h_dist <= straight_tol:
                straight_lines.append(chord)
            else:
                straight_lines.append(line.simplify(straight_tol * 0.75, preserve_topology=True))

        # 4. Planar Reconstruction (Polygonize)
        noded_straight = shapely.node(unary_union(straight_lines))
        reconstructed = list(polygonize(noded_straight))

        # 5. Cell Assignment to Original Parcels
        assigned_cells = {p['id']: [] for p in self.saved_paddocks}
        for cell in reconstructed:
            if cell.area < 100.0:  # Discard micro-noise under 100m²
                continue
            best_id = None
            max_inter = 0.0
            for idx, row in gdf_utm.iterrows():
                inter_area = cell.intersection(row.geometry).area
                if inter_area > max_inter:
                    max_inter = inter_area
                    best_id = row['id']
            if best_id and max_inter > (cell.area * 0.2):
                assigned_cells[best_id].append(cell)

        # 6. Update Saved Parcels
        updated_paddocks = []
        for p in self.saved_paddocks:
            pid = p['id']
            cells = assigned_cells.get(pid, [])
            if cells:
                u_poly = unary_union(cells)
                if isinstance(u_poly, MultiPolygon):
                    u_poly = max(u_poly.geoms, key=lambda g: g.area)
                if isinstance(u_poly, Polygon):
                    u_poly = Polygon(u_poly.exterior)

                # Convert back to WGS84
                gdf_single = gpd.GeoDataFrame([{'geometry': u_poly}], crs=utm_crs).to_crs("EPSG:4326")
                geom_wgs84 = gdf_single.geometry.iloc[0]
                p['geojson'] = mapping(geom_wgs84)
                p['area_ha'] = round(u_poly.area / 10000.0, 2)
            updated_paddocks.append(p)

        self.saved_paddocks = updated_paddocks
        return {
            "status": "success",
            "count": len(self.saved_paddocks),
            "paddocks": self.saved_paddocks,
            "total_area_ha": round(sum(p['area_ha'] for p in self.saved_paddocks), 2)
        }

    def delete_paddock(self, paddock_id):
        """Removes a parcel by ID."""
        self.saved_paddocks = [p for p in self.saved_paddocks if p['id'] != paddock_id]
        return len(self.saved_paddocks)

    def clear_session(self):
        """Clears active session parcels and temporary scratch data."""
        self.saved_paddocks = []
        self.current_prompt_points = []
        self.last_candidate_masks = None
        self.image_path = None
        self.active_roi_id = None
        self.overlay_png_path = None
        self.car_geojson = None
