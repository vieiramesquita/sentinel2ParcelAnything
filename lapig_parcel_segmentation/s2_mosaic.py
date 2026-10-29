"""
Module 1: Sentinel-2 Download and Multi-Temporal Processing via Google Earth Engine
Sentinel-2 Harmonized Collection + Cloud Score Plus (CS+)
LAPIG - Image Processing and Geoprocessing Laboratory
"""

import os
import ee
import geemap

def is_ee_initialized():
    """Checks whether Earth Engine has already been initialized in the current session."""
    try:
        return ee.data._credentials is not None
    except Exception:
        return False

def initialize_earth_engine(project=None):
    """
    Initializes Earth Engine honoring the user's active session.
    If already initialized and no specific project is forced, retains active session.
    """
    if is_ee_initialized() and (project is None or str(project).lower() in ['none', 'auto', 'default']):
        return

    # If explicit project provided
    if project and str(project).lower() not in ['none', 'auto', 'default']:
        try:
            ee.Initialize(project=project)
            return
        except Exception:
            pass

    # Attempt standard user environment initialization
    try:
        ee.Initialize()
        return
    except Exception:
        pass

    # Check EE_PROJECT environment variable
    env_p = os.environ.get('EE_PROJECT') or os.environ.get('EARTHENGINE_PROJECT')
    if env_p:
        try:
            ee.Initialize(project=env_p)
            return
        except Exception:
            pass

    # Fallback initialization
    try:
        ee.Initialize(project='ee-vieiramesquita')
    except Exception:
        pass

def get_sentinel2_ndvi_mosaic(roi_ee, year, start_month=3, end_month=10, buffer_meters=500, output_path=None, ee_project=None):
    """
    Generates multi-temporal NDVI statistical composite [max, median, min]
    with Cloud Score Plus cloud/shadow masking (cs_cdf >= 0.65).

    Returns:
      image_ndvis: ee.Image with bands ['NDVI_max', 'NDVI_median', 'NDVI_min']
      output_path: local filesystem path of exported GeoTIFF
      roi_buffered: expanded buffered geometry
    """
    initialize_earth_engine(project=ee_project)

    roi_buffered = roi_ee.geometry().buffer(buffer_meters).bounds()
    coords = roi_buffered.getInfo()['coordinates'][0]
    bbox = [coords[0][0], coords[0][1], coords[2][0], coords[2][1]]

    def mask_clouds(image):
        mask = image.select('cs_cdf').gte(0.65)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi).updateMask(mask)

    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

    s2_col = (s2
        .filterBounds(roi_buffered)
        .filterDate(f'{year}-{start_month:02d}-01', f'{year}-{end_month:02d}-30')
        .linkCollection(cs_plus, ['cs_cdf'])
        .map(mask_clouds))

    s2_col_ndvi = s2_col.select('NDVI')
    ndvi_max = s2_col_ndvi.reduce(ee.Reducer.max()).rename('NDVI_max')
    ndvi_median = s2_col_ndvi.reduce(ee.Reducer.median()).rename('NDVI_median')
    ndvi_min = s2_col_ndvi.reduce(ee.Reducer.min()).rename('NDVI_min')

    image_ndvis = ee.Image([ndvi_max, ndvi_median, ndvi_min]).clip(roi_buffered).unmask(0)

    if output_path is None:
        output_path = f"s2_ndvi_rgb_{year}.tif"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        print(f"Exporting Sentinel-2 NDVI mosaic to: {output_path}...")
        geemap.ee_to_geotiff(
            image_ndvis,
            output_path,
            bbox,
            zoom=14,
            vis_params={'min': [0.3, 0.15, 0.1], 'max': [0.9, 0.9, 0.7], 'bands': ['NDVI_max', 'NDVI_median', 'NDVI_min']}
        )
        print("Sentinel-2 mosaic download completed!")
    else:
        print(f"Sentinel-2 mosaic already exists locally: {output_path}")

    return image_ndvis, output_path, roi_buffered
