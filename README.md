# 🛰️ ParcelAnything: Cadastral Segmentation & Vectorization of Rural Land Parcels (LAPIG)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![SAMGeo](https://img.shields.io/badge/Segment%20Anything-ViT--H%20Deep%20Learning-0080FF.svg)](https://samgeo.gishat.org)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Sentinel--2%20CS%2B-2BA55D.svg?logo=googleearthengine&logoColor=white)](https://earthengine.google.com)
[![Topology](https://img.shields.io/badge/Topology-Watertight%20(Zero%20Overlaps)-success.svg)](#️-3-pipeline-architecture)
[![Performance](https://img.shields.io/badge/Performance-5.9%20min%2Ffarm%20(RTX%202080%20Ti)-orange.svg)](#-2-computational-performance--hardware-benchmarks)
[![Audited Benchmark](https://img.shields.io/badge/Benchmark-21%20Properties%20%7C%2049.8k%20ha-blue.svg)](#-2-computational-performance--hardware-benchmarks)

**Image Processing and Geoprocessing Laboratory (LAPIG / Federal University of Goiás - UFG)**  
*The **ParcelAnything** suite is a modular, high-performance geospatial framework offering two complementary modes for extracting rural land parcels and pasture boundaries with rectilinear fence lines (post-to-post), watertight planar topology (Zero Overlaps, Zero Gaps), zonal validation via MapBiomas (10m / 30m), and continuous hardware telemetry.*

---

## 🗺️ Overview: Two Consolidated Operating Modes

The project organizes cadastral parcel segmentation and vectorization into **two independent and complementary modes**:

```
Vectorization_Pro/
├── 🖱️ interactive_hq_sam/          # MODE 1: INTERACTIVE (Human-in-the-Loop)
│   ├── backend/                     # FastAPI API + PyTorch ViT-H on GPU (CUDA)
│   ├── frontend/                    # Leaflet Web UI with confirmation dock and (+/-) refinement
│   └── run_interactive_app.py       # Launcher: python run_interactive_app.py --port 8088
│
├── ⚙️ lapig_parcel_segmentation/    # MODE 2: AUTO (Batch Processing Pipeline)
│   ├── pipeline_parcel_segmentation # End-to-end master orchestrator
│   ├── roi_resolver.py              # Universal geometry resolver (CAR / Shapefile / Coordinates)
│   ├── s2_mosaic.py                 # Sentinel-2 Harmonized temporal composites (GEE CS+)
│   ├── sam_segmenter.py             # SAMGeo ViT-H gridded inference with strict VRAM governance
│   ├── pasture_classifier.py        # MapBiomas zonal pasture validation (10m Col 4 / 30m Col 11)
│   ├── cadastral_vectorizer.py      # Watertight Planar Graph (Zero Overlaps, rectilinear fences)
│   └── example_run.py               # CLI: python -m lapig_parcel_segmentation.example_run
│
├── 🌐 sentinel2parcel_test/         # WEB SHOWCASE: Public static demonstration on GitHub Pages
│   └── (Live at https://vieiramesquita.github.io/sentinel2parcel_test/)
│
├── 🧠 models/                       # Neural network weights (sam_vit_h_4b8939.pth)
├── 📊 results/                      # Generated vector layers (.shp / .gpkg), metrics & benchmark reports
└── 📓 run_in_jupyter.ipynb          # Step-by-step interactive demonstration notebook for Auto Mode
```

| Criterion | 🖱️ Interactive Mode (`interactive_hq_sam/`) | ⚙️ Auto Mode (`lapig_parcel_segmentation/`) |
| :--- | :--- | :--- |
| **Target User / Purpose** | Human operator, cadastral auditing, fine refinement of complex boundaries | Massive, autonomous processing of entire rural estates or municipalities |
| **Intervention Level** | **Human-in-the-Loop:** Click prompts with positive (`+`) and negative (`−`) seeds | **Zero intervention:** Fully autonomous end-to-end execution |
| **AI Model** | HQ-SAM ViT-H with interactive point sampling and intelligent partition cuts | SAMGeo ViT-H with automated grid-based inference |
| **Interface** | Web (Leaflet dark glassmorphism on port `8088` with neon dock) | Command-Line Interface (CLI) or Python script / Jupyter notebook |
| **Validation** | Visual operator validation with explicit "Confirm Parcel" action | Automated zonal dominance filtering via MapBiomas (pasture coverage) |
| **Input Formats** | Vector upload (`.kml`, `.kmz`, `.gpkg`, `.geojson`, `.zip`) or Lat/Lon | `.shp`, `.gpkg`, `.fgb`, `.parquet`, Earth Engine Asset, or Lat/Lon |
| **Outputs** | Browser-exportable GeoJSON with real-time cadastral metrics | Dual Shapefiles (`pure_segmentation` and `final_pasture`) |

---

## 📸 1. Results Gallery: From Satellite Imagery to Cadastral Vectors

**ParcelAnything** converts multitemporal Sentinel-2 imagery into GIS-ready, topologically sound cadastral polygons. Unlike naive vectorization techniques (such as standard raster polygonization or independent boundary smoothing), our method employs **Planar Graphs** that simplify shared boundaries, eliminate raster stairstepping (10m pixel artifacts), and enforce **Zero Overlaps and Zero Gaps (Watertight Topology)**.

### Case 1: Rural Estate in Caiapônia (GO, Brazil) — 749.52 ha
> **Highlights:** Extraction of 42 rectified parcels. Riparian forest buffers and central water bodies are strictly preserved, eliminating internal enclaves and Deep Learning noise artifacts.
![Caiapônia Results](images/fig_caiaponia_results.png)

---

### Case 2: Rural Estate in Paranaíba (MS, Brazil) — 1,085.22 ha
> **Highlights:** Large-scale property diagonally intersected by a railroad and a highway. The algorithm respected external linear corridors and generated 54 crisp, rectilinear parcels.
![Paranaíba Results](images/fig_paranaiba_results.png)

---

### Case 3: Rural Estate in Itapirapuã (GO, Brazil) — 193.59 ha
> **Highlights:** High agreement with manual ground-truth boundaries from a benchmark pilot farm. Post-to-post straight fence line rectification contouring natural valley drainage lines.
![Itapirapuã Results](images/fig_itapirapua_results.png)

---

## ⚡ 2. Computational Performance & Hardware Benchmarks

**ParcelAnything** was subjected to an audited continuous batch stress test covering **21 rural properties** (20 in Goiás and 1 in Mato Grosso do Sul, Brazil) across major agricultural frontiers, totaling **49,760.02 hectares** of analyzed cadastral area.

### Audited Performance Summary (Hardware: NVIDIA RTX 2080 Ti 11 GB)
* **Cadastral Volume:** **1,861 productive pasture parcels** delimited (**44,253.17 ha**) and **2,010 physical parcels** in pure segmentation (**47,479.02 ha**).
* **Total GPU Compute Time:** **122.81 minutes (~2.05 hours)** of continuous batch processing.
* **Average Time per Estate:** **~5.85 minutes** (~351 s), ranging from **2.5 min** on mid-sized uniform terrain to **8.0–9.5 min** on massive rural properties (>9,000 to 10,800 ha).
* **Allocated VRAM Peak:** Average of **6.41 GB**, operating comfortably at **6.47 GB** (58.8% of GPU capacity) with an absolute peak of **9.54 GB** (86.7%) on the largest estate (10,828.8 ha).
* **Watertight Topology:** **0 overlaps and 0 gaps** mathematically verified across 100% of the 1,861 generated polygons.
* **Thermal Stability:** Controlled GPU thermals throughout the batch (peak registered temperature of **70 °C** under sustained tensor core load).

### Computational Profiling by Pipeline Phase

| Pipeline Phase | Time Share (%) | Primary Hardware Device | Typical Runtime per Farm |
| :--- | :--- | :--- | :--- |
| **SAMGeo ViT-H Inference** | **86.2%** | Dedicated GPU (CUDA / Tensor Cores) | ~5.0 min |
| **Sentinel-2 Composite Download (GEE)** | **8.1%** | Network & Google Cloud API | ~28 s |
| **MapBiomas Zonal Classification (10m/30m)** | **1.6%** | Google Earth Engine / CPU | ~5.6 s |
| **Planar Graph Vectorization & Rectification** | **0.6%** | Multithreaded CPU | **~2.1 s** *(Ultrarunner)* |

### Detailed Audit of the 21 Processed Estates:

| Municipality | Property ID (CAR) | Estate Area | Validated Parcels | Pure Parcels | Total Time | VRAM Peak |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Goiás (GO)** | `GO-5208905-F49642881887...` | 10,828.80 ha | 345 | 354 | 8.2 min | 9.54 GB |
| **Nova Crixás (GO)** | `GO-5214838-32EB159D2A3C...` | 9,719.91 ha | 224 | 267 | 1.0 min | 0.01 GB |
| **Jataí (GO)** | `GO-5211909-AD13173067F8...` | 9,107.04 ha | 234 | 248 | 9.5 min | 8.53 GB |
| **Bom Jesus de Goiás (GO)** | `GO-5203500-366A293867A2...` | 4,476.31 ha | 138 | 178 | 9.5 min | 6.47 GB |
| **Britânia (GO)** | `GO-5203807-146172AFEDAF...` | 2,629.80 ha | 75 | 79 | 8.0 min | 6.47 GB |
| **Serranópolis (GO)** | `GO-5220504-AE59920E9ABB...` | 2,114.08 ha | 96 | 99 | 8.7 min | 6.47 GB |
| **Santa Fé de Goiás (GO)** | `GO-5219258-B3304BD38B7B...` | 1,889.02 ha | 119 | 121 | 4.6 min | 6.47 GB |
| **Ipameri (GO)** | `GO-5210109-9C728F04622C...` | 1,763.45 ha | 124 | 134 | 8.6 min | 6.47 GB |
| **Paranaíba (MS)** | `MS-5006309-8DC93A48C242...` | 1,085.22 ha | 54 | 55 | 3.7 min | 6.47 GB |
| **São Francisco de Goiás (GO)** | `GO-5219902-041335F6BCD1...` | 960.44 ha | 74 | 75 | 6.4 min | 6.47 GB |
| **Colinas do Sul (GO)** | `GO-5205521-DA104FC8E15A...` | 896.58 ha | 35 | 43 | 6.7 min | 6.47 GB |
| **Goiás (GO)** | `GO-5208905-08E150868E2A...` | 828.95 ha | 76 | 76 | 2.8 min | 6.47 GB |
| **Caiapônia (GO)** | `GO-5204409-01D1AD1F9586...` | 798.23 ha | 42 | 54 | 2.5 min | 6.47 GB |
| **Iaciara (GO)** | `GO-5209903-C8653C1E6389...` | 532.20 ha | 32 | 32 | 6.4 min | 6.47 GB |
| **Edéia (GO)** | `GO-5207402-1D40C8AAD650...` | 530.02 ha | 45 | 45 | 3.4 min | 6.47 GB |
| **Jussara (GO)** | `GO-5212204-38A49ED0A118...` | 514.25 ha | 29 | 29 | 2.5 min | 6.47 GB |
| **Jussara (GO)** | `GO-5212204-0B3CAD6C8F1D...` | 420.28 ha | 42 | 42 | 7.9 min | 6.47 GB |
| **Crixás (GO)** | `GO-5206404-499D0D75A139...` | 233.81 ha | 37 | 37 | 5.8 min | 6.47 GB |
| **Itapirapuã (GO)** | `GO-5211008-46C08C6235D7...` | 193.59 ha | 14 | 14 | 3.1 min | 6.47 GB |
| **Luziânia (GO)** | `GO-5212501-89730760C68B...` | 167.76 ha | 16 | 17 | 6.4 min | 6.47 GB |
| **Goiânia (GO)** | `GO-5208707-4045057E90C9...` | 70.28 ha | 10 | 11 | 7.1 min | 6.47 GB |

> **💡 Zero VRAM Leak Memory Management:**  
> The [`clear_gpu_memory()`](lapig_parcel_segmentation/sam_segmenter.py) routine is systematically invoked upon completing each estate. It explicitly unbinds `SamGeo` tensors, triggers Python garbage collection (`gc.collect()`), flushes PyTorch CUDA caching allocator (`torch.cuda.empty_cache()`), and releases inter-process CUDA memory (`torch.cuda.ipc_collect()`). This guarantees that each property begins execution with **0 MB of residual VRAM**.

---

## 🏗️ 3. Pipeline Architecture

```
                 Universal Area of Interest Input (Universal ROI)
       ┌────────────────────────┬────────────────────────┬────────────────────────┐
       │  Local Vector File     │  Earth Engine Asset    │  GPS Coordinates       │
       │  .shp, .gpkg, .fgb,    │  'projects/...' or     │  (lat, lon) ->         │
       │  .parquet, .geojson    │  ee.FeatureCollection  │  GEE CAR Registry      │
       │  (with id_column)      │  (with ID filter)      │  (auto-extracts CAR)   │
       └───────────┬────────────┴───────────┬────────────┴───────────┬────────────┘
                   └────────────────────────┼────────────────────────┘
                                            │
                                            ▼
                       [Dedicated Property Folder]
                          results/{PROPERTY_ID}/
                                            │
                                            ▼
                                  Google Earth Engine
                          ┌─────────────────────────────────┐
                          │  Sentinel-2 Harmonized (CS+)    │
                          │  Bands: NDVI max, median, min   │
                          └─────────────────┬───────────────┘
                                            │
                                            ▼
                                 [Module: s2_mosaic]
                                            │ (Multitemporal GeoTIFF composite)
                                            ▼
                             [Module: sam_segmenter]
                         (SAMGeo ViT-H on CUDA GPU 8GB/11GB+)
                                            │ (Raw binary raster masks)
                                            ▼
                          [Module: cadastral_vectorizer]
                          ├── Internal enclave and hole elimination
                          ├── Planar Graph and Fence Line Rectification
                          └── Watertight Reconstruction (Zero Overlaps)
                                            │
                                            ▼
                     [Strict Boundary Clipping to Rural Property CAR]
                                            │
         ┌──────────────────────────────────┴──────────────────────────────────┐
         ▼                                                                     ▼
 [LAYER 1: PURE PHYSICAL SEGMENTATION]                                 [LAYER 2: PASTURE DOMINANCE]
 All physical parcels within the estate                                Productive pasture parcels
 with comprehensive audit attributes:                                  validated by zonal coverage:
 (pasture_cov, ndvi_med, is_pasture, class)                           (MapBiomas 10m Col 4 / 30m Col 11)
         │                                                                     │
         ▼                                                                     ▼
 Shapefile: parcels_pure_segmentation_{ID}.shp      Shapefile: parcels_final_straight_fences_{ID}.shp
```

---

## 🚀 4. Installation & Environment Setup

### 4.1. System Requirements
- **Operating System:** Linux (Ubuntu 22.04+) or Windows 11 with WSL 2
- **Python:** Version 3.10 or higher
- **NVIDIA GPU:** Minimum 8 GB VRAM (11 GB+ recommended, e.g., RTX 2080 Ti, RTX 3060 12GB, RTX 4070+)
- **Google Earth Engine:** Active account authenticated via Earth Engine CLI

### 4.2. Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/vieiramesquita/sentinel2ParcelAnything.git
cd sentinel2ParcelAnything

# 2. Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install PyTorch with CUDA 11.8 or 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install geospatial and web application dependencies
pip install segment-geospatial earthengine-api geopandas shapely rasterio matplotlib pandas duckdb psutil plotly uvicorn fastapi pydantic pyproj pillow

# 5. Authenticate with Google Earth Engine
earthengine authenticate
```

### 4.3. Downloading SAM (ViT-H) Model Weights
The SAM ViT-H model weights checkpoint (`sam_vit_h_4b8939.pth` - ~2.5 GB) must be placed in the `models/` directory:
```bash
mkdir -p models
# Direct download from Meta AI official weights repository:
curl -o models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## ⚙️ 5. Auto Mode CLI & Python Usage (`lapig_parcel_segmentation`)

The automated batch pipeline is executed via the Command-Line Interface (CLI) through [`example_run.py`](lapig_parcel_segmentation/example_run.py) or imported directly as a Python module.

### 5.1. CLI Parameter Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | `None` | Path to local vector file (`.shp`, `.gpkg`, `.fgb`, `.parquet`), Earth Engine asset, or `"lat, lon"` coordinates. |
| `--coord` | `str` | `None` | Geographic coordinate `"lat, lon"` to query and extract the property boundary automatically from the CAR registry in GEE. |
| `--id-col` | `str` | `None` | Name of the identifier column in the vector file (e.g., `cod_imovel`, `gid`, `id`). |
| `--id-val` | `str` | `None` | Value of the specific property ID to filter and process from the vector layer. |
| `--all` | `flag` | `False` | Executes sequential batch processing for **all** rural properties present in the vector dataset. |
| `--limit` | `int` | `None` | Caps the number of properties to process in the batch (ideal for quick tests, e.g., `--limit 2`). |
| `--list-ids` | `flag` | `False` | Prints all available IDs in the input vector file and exits. |
| `--year` | `int` | `2025` | Reference year for the Sentinel-2 seasonal composite and MapBiomas land cover mask. |
| `--collection` | `str` | `10m` | MapBiomas collection: `10m` (Sentinel-2 Col 4) or `30m` (Landsat Col 11). |
| `--straight-tol` | `float` | `22.0` | Simplification tolerance in meters for fence line rectification (higher = straighter fence lines). |
| `--min_area` | `float` | `2.0` | Minimum parcel area in hectares (smaller slivers/islands are merged into their largest neighbor). |
| `--no-pasture-filter`| `flag`| `False` | Bypasses the MapBiomas pasture filter, outputting purely the physical cadastral boundaries. |
| `--output-dir` | `str` | `results` | Root directory where property subfolders and reports are saved. |
| `--project` | `str` | `None` | Google Earth Engine project ID (optional; defaults to the active authenticated session). |

---

### 5.2. Practical Execution Scenarios

#### Scenario A: Process a specific farm from a local CAR GeoPackage
```bash
python -m lapig_parcel_segmentation.example_run \
    --input "car_goias_area_imovel_epsg4674_hottest.gpkg" \
    --id-col "cod_imovel" \
    --id-val "GO-5204409-01D1AD1F9586415E9D46BF93115D27DF" \
    --year 2025 \
    --straight-tol 22.0
```

#### Scenario B: Process a property using only its GPS coordinates
The pipeline queries the Earth Engine CAR repository, locates the exact farm boundaries, downloads the Sentinel-2 composite, and runs full cadastral vectorization:
```bash
python -m lapig_parcel_segmentation.example_run \
    --coord "-16.1234, -50.5678" \
    --year 2025 \
    --min_area 2.0
```

#### Scenario C: List all property codes in your dataset before executing
```bash
python -m lapig_parcel_segmentation.example_run \
    --input "car_goias_area_imovel_epsg4674_hottest.gpkg" \
    --id-col "cod_imovel" \
    --list-ids
```

#### Scenario D: Full Batch Processing with Consolidated Output Shapefiles
```bash
python -m lapig_parcel_segmentation.example_run \
    --input "car_goias_area_imovel_epsg4674_hottest.gpkg" \
    --id-col "cod_imovel" \
    --all
```
*At the conclusion of the batch run, the pipeline automatically compiles:*
- `results/parcelas_todos_imoveis_consolidados.shp` (All pasture parcels across all properties unified into a single shapefile).
- `results/parcelas_todos_imoveis_segmentacao_pura.shp` (Pure physical segmentation of all properties).
- `results/relatorio_benchmark_hardware_lote.json` and `.txt` (Consolidated hardware profiling report).

#### Scenario E: Pure Physical Segmentation Mode (Without Pasture Masking)
Ideal for researchers and GIS analysts wanting to map all physical field partitions regardless of current land use:
```bash
python -m lapig_parcel_segmentation.example_run \
    --input "car_goias_area_imovel_epsg4674_hottest.gpkg" \
    --id-col "cod_imovel" \
    --id-val "GO-5204409-01D1AD1F9586415E9D46BF93115D27DF" \
    --no-pasture-filter
```

---

### 5.3. Programmatic Usage in Python

```python
from lapig_parcel_segmentation import run_analysis_lapig

# Execute end-to-end segmentation directly from Python
gdf_pasture, shp_pasture, gdf_pure, shp_pure = run_analysis_lapig(
    input_target="car_goias_area_imovel_epsg4674_hottest.gpkg",
    id_column="cod_imovel",
    id_value="GO-5204409-01D1AD1F9586415E9D46BF93115D27DF",
    year=2025,
    min_area_ha=2.0,
    straight_tol=22.0,
    filter_by_pasture=True,
    enable_benchmark=True
)

print(f"Validated pasture parcels: {len(gdf_pasture)}")
print(f"Generated shapefile: {shp_pasture}")
```

---

## 🖱️ 6. Interactive Human-in-the-Loop Mode (`interactive_hq_sam`)

The interactive mode is engineered for **visual-intelligence assisted operations**. It allows GIS operators to audit pasture boundaries, resolve contested fence lines, and segment intricate parcel layouts with instantaneous AI feedback.

### 6.1. Starting the Local Web Server

From the project root:
```bash
python interactive_hq_sam/run_interactive_app.py --port 8088
```
Open in your browser:
👉 **`http://localhost:8088/`**

---

### 6.2. Step-by-Step Operator Workflow

```
  1. Load ROI              2. Prompt Sampling      3. Validation         4. Watertight Topology
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌────────────────────────┐
│ - Upload KML/KMZ │    │ - Positive (+)   │    │ - Inspect Area  │    │ - Regularize Boundaries│
│ - Property CAR   │ -> │ - Negative (−)   │ -> │ - "Confirm      │ -> │   (Planar Graph)       │
│ - Lat / Lon      │    │   (Perpendicular │    │    Parcel"      │    │ - "Export GeoJSON"     │
│ - Demo Pre-sets  │    │    Bisector Cut) │    │                 │    │                        │
└──────────────────┘    └──────────────────┘    └─────────────────┘    └────────────────────────┘
```

1. **Load Region of Interest (ROI):**
   - **Vector File Upload:** Drag-and-drop or upload `.kml`, `.kmz`, `.gpkg`, `.geojson`, or `.zip` (containing Shapefile). 3D altitude coordinates are automatically sanitized to 2D for Earth Engine compatibility.
   - **CAR Property Code:** Enter the cadastral property code to stream the multitemporal Sentinel-2 mosaic on the fly.
   - **Lat/Lon Coordinates:** Provide the central geographic coordinates of the rural property.
   - **Demo Properties:** Select pre-cached demonstration estates from the sidebar.

2. **Point Sampling with Positive and Negative Seeds:**
   - **Positive Seed (`+` - Green):** Click inside the desired pasture parcel. Multiple positive points merge contiguous field regions.
   - **Negative Seed (`−` - Red):** Click on dams, farmsteads, forest fragments, or adjacent pastures. The engine calculates the **perpendicular bisector cut** between closest positive and negative seeds, instantly severing disputed boundaries.

3. **Confirming the Parcel:**
   - A dashed neon cyan contour displays the proposed polygon with real-time area in hectares (`ha`) and model confidence.
   - Click **`Confirm Parcel`** on the bottom dock to commit the polygon to the active session database.

4. **Refining and Editing Existing Parcels:**
   - To fine-tune an already confirmed parcel, click on it directly on the map or in the sidebar inventory. Adding new positive or negative points updates its geometry dynamically.

5. **Watertight Topological Regularization & Export:**
   - Click **`⚡ Regularize Topology (Watertight)`** to pass all session parcels through a planar graph engine, snapping shared edges and straightening fence lines post-to-post (0% overlap guaranteed).
   - Click **`Export GeoJSON`** to download the clean vector layer ready for QGIS or ArcGIS.

---

## 📂 7. Directory Structure & Attribute Dictionary

### Generated File Hierarchy per Property in `results/{ID}/`:
```
results/GO-5204409-01D1AD1F9586415E9D46BF93115D27DF/
├── parcelas_segmentacao_pura_GO-5204409...shp      # All physical parcel partitions in CAR
├── parcelas_finais_cercas_retas_GO-5204409...shp   # Validated pasture parcels only
├── nao_pastagem_auditoria.shp                      # Discarded areas (forest, crop, water)
├── s2_ndvi_rgb.tif                                 # Sentinel-2 CS+ composite GeoTIFF
├── relatorio_benchmark_hardware.json               # Structured hardware profiling data
└── relatorio_benchmark_hardware.txt                # Human-readable hardware benchmark report
```

### Vector Layer Attribute Dictionary:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id_parcela` | `Integer` | Unique sequential identifier of the parcel within the property. |
| `cod_car` | `String` | Official cadastral identification code of the rural property in the CAR. |
| `area_ha` | `Float` | Calculated parcel area in hectares (`ha`). |
| `perim_m` | `Float` | Parcel polygon perimeter in meters. |
| `pasto_cov` | `Float` | Proportional pasture coverage of the parcel according to MapBiomas (0.00 to 1.00). |
| `ndvi_med` | `Float` | Median NDVI spectral index within the parcel over the temporal observation window. |
| `is_pasto` | `Integer` | `1` if approved as pasture (dominance $\ge$ 35%), `0` if classified as non-pasture. |
| `classe` | `String` | Textual classification label: `'Pasto'` or `'Outros (Mata/APP/Lavoura)'`. |

---

## 👥 Credits & Team

* **Image Processing and Geoprocessing Laboratory (LAPIG)** — Federal University of Goiás (UFG, Brazil)
* **Lead Development & Architecture:** Vinícius Mesquita & LAPIG Team
* **AI & Engineering Collaboration:** Antigravity (Google DeepMind)
