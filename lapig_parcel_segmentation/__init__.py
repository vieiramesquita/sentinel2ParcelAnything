"""
Pacote Modular de Segmentação e Vetorização Cadastral de Piquetes
LAPIG - Laboratório de Processamento de Imagens e Geoprocessamento (UFG)

Módulos disponíveis:
  - roi_resolver: Resolvedor universal de entrada com suporte a id_column e busca CAR
  - s2_mosaic: Download Sentinel-2 Harmonized + Cloud Score Plus via Google Earth Engine
  - sam_segmenter: Segmentação SAMGeo com controle de GPU (RTX 3060 Ti / RTX 2080 Ti)
  - cadastral_vectorizer: Vetorização cadastral, absorção de enclaves/ilhas e cercas retilíneas
  - pasture_classifier: Classificação zonal de pastagem (MapBiomas 10m Col 4 / 30m Col 11 até 2025)
  - pipeline_parcel_segmentation: Orquestrador principal de ponta a ponta (run_analysis_lapig)
"""

try:
    from .pipeline_parcel_segmentation import run_analysis_lapig, run_batch_analysis_lapig
except ImportError:
    run_analysis_lapig, run_batch_analysis_lapig = None, None

try:
    from .roi_resolver import resolve_input_roi, parse_lat_lon, gdf_to_ee_feature_collection, sanitize_id
except ImportError:
    resolve_input_roi, parse_lat_lon, gdf_to_ee_feature_collection, sanitize_id = None, None, None, None

try:
    from .s2_mosaic import get_sentinel2_ndvi_mosaic
except ImportError:
    get_sentinel2_ndvi_mosaic = None

try:
    from .sam_segmenter import run_samgeo_segmentation, get_gpu_info, clear_gpu_memory
except ImportError:
    run_samgeo_segmentation, get_gpu_info, clear_gpu_memory = None, None, None

try:
    from .cadastral_vectorizer import (
        vectorize_and_straighten_parcels,
        clip_parcels_by_car,
        generate_cadastral_stats
    )
except ImportError:
    vectorize_and_straighten_parcels = None
    clip_parcels_by_car = None
    generate_cadastral_stats = None

try:
    from .pasture_classifier import (
        get_mapbiomas_pasture_ee,
        export_pasture_mask_local,
        classify_and_filter_pasture_parcels
    )
except ImportError:
    get_mapbiomas_pasture_ee = None
    export_pasture_mask_local = None
    classify_and_filter_pasture_parcels = None

try:
    from .benchmark_monitor import HardwareBenchmarkMonitor
except ImportError:
    HardwareBenchmarkMonitor = None

from .benchmark_dashboard import collect_benchmark_dataset, generate_dashboard_html

__all__ = [
    "run_analysis_lapig",
    "run_batch_analysis_lapig",
    "HardwareBenchmarkMonitor",
    "resolve_input_roi",
    "sanitize_id",
    "parse_lat_lon",
    "gdf_to_ee_feature_collection",
    "get_sentinel2_ndvi_mosaic",
    "run_samgeo_segmentation",
    "get_gpu_info",
    "clear_gpu_memory",
    "vectorize_and_straighten_parcels",
    "clip_parcels_by_car",
    "generate_cadastral_stats",
    "get_mapbiomas_pasture_ee",
    "export_pasture_mask_local",
    "classify_and_filter_pasture_parcels",
    "collect_benchmark_dataset",
    "generate_dashboard_html",
]

__version__ = "2.2.0"
