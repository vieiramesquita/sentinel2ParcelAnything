# ⚙️ ParcelAnything: Auto Mode (Automated Batch Cadastral Vectorization)

**Image Processing and Geoprocessing Laboratory (LAPIG / Federal University of Goiás - UFG)**  
*Modular pipeline for automated extraction of rural land parcels with rectilinear fence lines (post-to-post), pure cadastral segmentation, thematic land cover classification via MapBiomas, CAR estate clipping, and automated hardware benchmarking.*

---

## 📌 1. Architecture Overview

This package modernizes and modularizes the automated workflow from `parcelSegmentation_SAMGeo_LAPIG.ipynb`, eliminating legacy pixel-mutilating operations and introducing a state-of-the-art cadastral architecture based on **Universal ROI Resolution**, **Automated Property ID Subdirectories**, **Pure Physical Segmentation**, **Planar Graphs**, **Enclave Elimination**, **MapBiomas Zonal Dominance**, and **Continuous Hardware Telemetry**.

```
                 Universal Area of Interest Input (Universal ROI)
       ┌────────────────────────┬────────────────────────┬────────────────────────┐
       │  Local Vector File     │  Earth Engine Asset    │  GPS Coordinates       │
       │  .shp, .gpkg, .fgb,    │  'projects/...' or     │  (lat, lon) ->         │
       │  .parquet, .geojson    │  ee.FeatureCollection  │  GEE CAR Query         │
       │  (with id_column)      │  (with ID filter)      │  (auto-extracts CAR)   │
       └───────────┬────────────┴───────────┬────────────┴───────────┬────────────┘
                   └────────────────────────┼────────────────────────┘
                                            │
                                            ▼
                       [Dedicated Property Directory]
                          results/{PROPERTY_ID}/
                                            │ (roi_ee + gdf_car)
                                            ▼
                                  Google Earth Engine
                          ┌─────────────────────────────────┐
                          │  Sentinel-2 Harmonized (CS+)    │
                          │  Bands: NDVI max, median, min   │
                          └─────────────────┬───────────────┘
                                            │
                                            ▼
                                 [Module 1: s2_mosaic]
                                            │ (Multitemporal GeoTIFF composite)
                                            ▼
                             [Module 2: sam_segmenter]
                         (SAMGeo ViT-H on CUDA GPU 8GB/11GB+)
                                            │ (Raw binary raster masks)
                                            ▼
                          [Module 3: cadastral_vectorizer]
                          ├── Internal enclave and hole elimination
                          ├── Planar Graph and Fence Line Rectification
                          └── Watertight Reconstruction (Zero Overlaps)
                                            │
                                            ▼
                     [Strict Boundary Clipping to Rural Property CAR]
                                            │
         ┌──────────────────────────────────┴──────────────────────────────────┐
         ▼                                                                     ▼
 [LAYER 1: PURE PHYSICAL SEGMENTATION]                                 [LAYER 2: THEMATIC FILTER]
 All physical parcels within CAR estate                                Productive pasture parcels
 with comprehensive audit attributes:                                  validated by zonal coverage:
 (pasture_cov, ndvi_med, is_pasture, class)                           (MapBiomas 10m Col 4 / 30m Col 11)
         │                                                                     │
         ▼                                                                     ▼
 Shapefile: parcels_pure_segmentation_{ID}.shp      Shapefile: parcels_final_straight_fences_{ID}.shp
```

---

## ⚡ 2. Automated Hardware Benchmark

For systems equipped with dedicated GPUs (e.g., **NVIDIA RTX 2080 Ti 11 GB**, **RTX 3060 Ti 8 GB**), processing Sentinel-2 imagery via SAMGeo (ViT-H) and planar graph vectorization demands substantial computational resources.

The package integrates an asynchronous **Background Hardware Benchmark Monitor (`HardwareBenchmarkMonitor`)** that operates with zero CPU overhead, sampling every 1 second:
- **GPU VRAM:** Allocated and reserved CUDA memory via PyTorch (Peak & Mean).
- **System RAM:** Percentage and resident set size (RSS) memory consumed by the Python process.
- **CPU:** Global and per-core processor utilization percentage.
- **GPU (Compute Load & Temperature):** CUDA core workload (%) and operational temperature (°C) via NVIDIA drivers.
- **Phase Breakdown:** Exact execution time logged across all pipeline stages:
  1. *ROI Resolution & CAR Boundary Extraction*
  2. *Sentinel-2 Composite Download*
  3. *SAMGeo Inference (ViT-H)*
  4. *Planar Graph Vectorization & Fence Line Rectification*
  5. *Pure Physical Segmentation Clipping*
  6. *MapBiomas Zonal Dominance Classification (10m / 30m)*
  7. *Cadastral Export & Topological Validation*

### Generated Benchmark Artifacts
Upon completing each estate, the following reports are written to the property results folder:
1. `relatorio_benchmark_hardware.txt`: Human-readable tabular report for technical audits.
2. `relatorio_benchmark_hardware.json`: Structured JSON containing raw telemetry timeseries for dashboards and charts.
3. `relatorio_benchmark_hardware_lote.txt`: Consolidated multi-farm benchmark summary for batch executions (`run_batch_analysis_lapig`).

### 🧹 Continuous Memory Governance (Zero VRAM Leak)
To prevent GPU out-of-memory errors (*CUDA Out of Memory*) during large-scale sequential batch processing:
- **Forced Model Teardown:** The `SamGeo` model instance and internal tensor references (`sam.sam`, `sam.mask_generator`, `sam.predictor`) are explicitly deleted inside a `try ... finally` block immediately after generating `masks.tif`.
- **Garbage Collection & Cache Purge:** The `clear_gpu_memory(verbose=True)` function executes Python garbage collection (`gc.collect()`), flushes PyTorch CUDA caching allocators (`torch.cuda.empty_cache()`), and releases inter-process memory blocks (`torch.cuda.ipc_collect()`).
- **Guaranteed Per-Iteration Cleanliness:** Batch runs (`run_batch_analysis_lapig`) purge memory at the beginning and end of every estate, ensuring the subsequent property starts with 0 MB of residual VRAM.
- **Programmatic Access:** Developers can invoke `from lapig_parcel_segmentation import clear_gpu_memory; clear_gpu_memory()` at any point in custom scripts and notebooks.

---

## 🌾 3. Pure Physical Segmentation vs. Pasture Filtering

At the property scale, national-scale land cover maps (such as MapBiomas) may exhibit occasional classification discrepancies (e.g., degraded pastures classified as secondary vegetation, or fallow land classified as temporary crops).

To address this limitation, the pipeline generates **two complementary vector layers**:

| Layer | Generated File | Description & Application |
| :--- | :--- | :--- |
| **Pure Segmentation** | `parcelas_segmentacao_pura_{ID}.shp` | **All physical parcels and management units** delimited by real fence lines within the property, with no thematic exclusions. Includes audit columns: `pasto_cov`, `ndvi_med`, `is_pasto` (1 or 0), and `classe`. Ideal for whole-farm cadastral management and rotational grazing planning. |
| **Productive Parcels** | `parcelas_finais_cercas_retas_{ID}.shp` | Validated subset filtered by MapBiomas zonal dominance (`pasture_cov >= 0.35`). |
| **Non-Pasture Audit** | `nao_pastagem_auditoria.shp` | Polygons representing riparian forest buffers, legal reserves, water bodies, and infrastructure excluded by the pasture filter. |

---

## 📁 4. Standardized Output Structure (Organization by ID)

Results are automatically structured into a dedicated subfolder named after the sanitized property identifier:
```text
results/{PROPERTY_ID}/
```
*Example:* `results/GO-5203500-366A293867A24EF5A446DA91E23F8548/`

**Files contained within each property folder:**
* `s2_ndvi_rgb.tif`: Multitemporal Sentinel-2 composite (NDVI max, median, min).
* `masks.tif`: Raw raster masks predicted by SAMGeo.
* `temp_mb_mask_10m.tif`: Binary MapBiomas Class 15 pasture mask exported from GEE.
* `parcelas_segmentacao_pura_{ID}.shp`: **Pure physical segmentation of all parcels within the estate.**
* `parcelas_finais_cercas_retas_{ID}.shp`: **Official validated pasture parcels shapefile.**
* `nao_pastagem_auditoria.shp`: Native vegetation, forest, and non-pasture areas for audit.
* `relatorio_benchmark_hardware.txt`: Human-readable hardware benchmark report.
* `relatorio_benchmark_hardware.json`: Raw structured hardware telemetry JSON.

---

## 🎯 5. Input Modalities & ID Column (`id_column`)

The `input_target` parameter accepts:
* **Local Vector Files:** `.shp`, `.gpkg`, `.fgb`, `.parquet`, `.geojson`.
* **Earth Engine Asset or Object:** `'projects/...'` or `ee.FeatureCollection`.
* **Geographic Coordinates:** Tuple `(-15.747, -50.894)` with automated spatial intersection against official state CAR registries.

---

## 🚀 6. Code Examples

### Example 1: Python / JupyterLab Execution (with Pure Segmentation & Hardware Telemetry)
```python
import ee
from lapig_parcel_segmentation import run_analysis_lapig

ee.Initialize(project='your-gee-project-id')

# Returns both pasture polygons and pure physical segmentation:
gdf_pasture, shp_pasture, gdf_pure, shp_pure = run_analysis_lapig(
    input_target="car_goias_area_imovel_epsg4674_hottest.gpkg",
    id_column="cod_imovel",
    id_value="GO-5203500-366A293867A24EF5A446DA91E23F8548",
    year=2025,
    mapbiomas_collection='10m',
    min_area_ha=2.0,
    straight_tol=22.0,
    return_pure=True,          # Returns 4-tuple: (gdf_pasture, shp_pasture, gdf_pure, shp_pure)
    enable_benchmark=True      # Activates background hardware monitoring
)

print(f"Validated Parcels: {shp_pasture} ({len(gdf_pasture)} parcels)")
print(f"Pure Segmentation: {shp_pure} ({len(gdf_pure)} parcels)")
```

### Example 2: Inspecting Benchmark Reports in Notebooks
```python
import os

report_file = os.path.join(os.path.dirname(shp_pasture), "relatorio_benchmark_hardware.txt")
if os.path.exists(report_file):
    with open(report_file, 'r', encoding='utf-8') as f:
        print(f.read())
```

### Example 3: Command-Line Interface (CLI)
```bash
# 1. Process a specific farm (automatically generates both layers and hardware reports):
python lapig_parcel_segmentation/example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --id-val "GO-5203500-366A293867A24EF5A446DA91E23F8548"

# 2. Run in Pure Physical Segmentation mode (bypasses MapBiomas pasture filtering):
python lapig_parcel_segmentation/example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --id-val "GO-5203500-366A293867A24EF5A446DA91E23F8548" --no-pasture-filter

# 3. Batch processing for ALL properties in the dataset:
python lapig_parcel_segmentation/example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --all

# 4. List all property codes present in the file:
python lapig_parcel_segmentation/example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --list-ids
```

### Example 4: Python Batch Processing (`run_batch_analysis_lapig`)
```python
import ee
from lapig_parcel_segmentation import run_batch_analysis_lapig

ee.Initialize(project='your-gee-project-id')

# Sequentially processes all farms in the dataset,
# saves each to its dedicated subfolder (results/{cod_imovel}/),
# compiles consolidated shapefiles, and generates a hardware report:
gdfs_success, shp_consolidated = run_batch_analysis_lapig(
    input_target="car_goias_area_imovel_epsg4674_hottest.gpkg",
    id_column="cod_imovel",
    year=2025,
    mapbiomas_collection='10m',
    consolidate_shp=True,
    enable_benchmark=True
)

# Consolidated shapefiles generated at root results directory:
# - results/parcelas_todos_imoveis_consolidados.shp (Validated Pasture)
# - results/parcelas_todos_imoveis_segmentacao_pura.shp (Pure Segmentation)
# - results/relatorio_benchmark_hardware_lote.txt (Batch Hardware Benchmark)
```

---

## 📊 7. `run_analysis_lapig` Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_target` / `roi` | `str`, `tuple`, `ee.FeatureCollection` | *Required* | Local file (`.shp`, `.fgb`, `.gpkg`, `.parquet`), EE asset, or `(lat, lon)` tuple. |
| `id_column` | `str` | `None` | Identifier column in the vector dataset (e.g., `'cod_imovel'`, `'id'`). |
| `id_value` | `str` / `int` | `None` | Specific property ID value. Use `'all'` to process the entire dataset. |
| `folder_name` | `str` | `None` | If `None` or `'auto'`, creates standardized directory: `results/{PROPERTY_ID}/`. |
| `output_base_dir` | `str` | `"results"` | Root directory where property subfolders are created. |
| `year` | `int` | `2025` | Reference year for Sentinel-2 composites and MapBiomas land cover. |
| `start_month` | `int` | `3` | Start month of the dry observation window (highest spectral contrast). |
| `end_month` | `int` | `10` | End month of the dry observation window. |
| `buffer_size` | `int` | `500` | Buffer in meters around property boundary to prevent edge clipping artifacts. |
| `min_area_ha` | `float` | `2.0` | Minimum parcel area in hectares. Smaller slivers are merged into their neighbor. |
| `straight_tol` | `float` | `22.0` | Hausdorff distance tolerance (m) for post-to-post straight fence line simplification. |
| `min_pasture_coverage` | `float` | `0.35` | Minimum pasture pixel fraction (0 to 1) required to validate a parcel. |
| `mapbiomas_collection` | `str` | `'10m'` | MapBiomas collection: `'10m'` (Col 4 Sentinel-2) or `'30m'` (Col 11 Landsat). |
| `skip_sam_if_exists` | `bool` | `True` | Bypasses SAMGeo inference if `masks.tif` already exists in the folder. |
| `return_pure` | `bool` | `False` | If `True`, returns a 4-tuple: `(gdf_pasture, shp_pasture, gdf_pure, shp_pure)`. |
| `enable_benchmark` | `bool` | `True` | Enables asynchronous hardware profiling and saves benchmark reports. |

---
*LAPIG - Image Processing and Geoprocessing Laboratory / Federal University of Goiás (UFG, Brazil)*
