# 🖱️ ParcelAnything: Interactive Mode (Human-in-the-Loop Segmentation)
**LAPIG - Image Processing and Geoprocessing Laboratory (Federal University of Goiás - UFG)**

**ParcelAnything Interactive** is engineered for **visual-intelligence assisted operations**, enabling a human GIS operator to define, audit, or refine pasture and parcel boundaries with prompt-based clicks on an interactive map.

---

## 🎯 Key Features

1. **Intelligent Point Sampling (HQ-SAM ViT-H)**:
   - **Positive Seed (`+`)**: Identifies the interior of the parcel to be segmented. Supports multiple positive points to merge contiguous field partitions.
   - **Negative Seed (`−`)**: Excludes non-target areas (dams, farm infrastructure, forest buffers), eliminates adjacent parcels, or splits continuous pastures via automated perpendicular bisector cuts.
2. **Mandatory Confirmation & Progressive Refinement**:
   - Clicks generate a dashed neon cyan visual preview along with estimated area (ha) and confidence score.
   - Polygons are only committed after explicit operator validation on the bottom dock (`Confirm Parcel`).
   - Allows selecting any previously saved parcel to edit or refine its boundaries with new prompt points.
3. **Flexible Input Modalities**:
   - **Geographic Coordinates (Lat/Lon)**: Fetches the harmonized Sentinel-2 composite in real-time from Google Earth Engine.
   - **Property Code (CAR)**: Locates the rural estate geometry and downloads the multitemporal spectral composite.
   - **Vector Upload**: Direct drag-and-drop support for `.kml`, `.kmz`, `.gpkg`, `.zip` (Shapefile), and `.geojson` (with automatic 3D-to-2D altitude sanitization).
   - **Pre-cached Demo Farms**: Instant demonstrations of rural estates with normalized spectral composites.
4. **Cadastral Metrics & Watertight Export**:
   - Automated calculation of parcel area in hectares (`ha`), perimeter, and segmentation confidence.
   - Watertight planar graph topological regularization (Zero Overlaps, Zero Gaps).
   - Consolidated cadastral export in standard **GeoJSON** format compatible with QGIS and ArcGIS.

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+ with PyTorch and CUDA support (recommended GPU with 8GB+ VRAM, e.g., RTX 2080 Ti, RTX 3060 12GB).
- SAM ViT-H model weights located at `models/sam_vit_h_4b8939.pth`.

### Starting the Local Server
From the project root:
```bash
python interactive_hq_sam/run_interactive_app.py --port 8088
```

Or specifying the host for local network access:
```bash
python interactive_hq_sam/run_interactive_app.py --host 0.0.0.0 --port 8088
```

Open in your browser:
👉 **`http://localhost:8088/`**

---

## 📂 Directory Structure

```
interactive_hq_sam/
├── backend/
│   ├── app.py              # FastAPI server (endpoints /api/predict, /api/roi, /api/paddock)
│   └── hq_sam_engine.py    # PyTorch engine with HQ-SAM ViT-H and GEE integration
├── frontend/
│   ├── index.html          # Responsive Leaflet web application with refinement dock
│   ├── style.css           # LAPIG Design System (Dark glassmorphism neon theme)
│   └── app.js              # Map controls, event handlers, and API client logic
├── cache/                  # Cached spectral overlays and GeoTIFF composites
└── run_interactive_app.py  # Standalone application entrypoint
```
