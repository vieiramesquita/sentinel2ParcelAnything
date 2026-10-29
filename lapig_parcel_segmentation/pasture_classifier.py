"""
Module 4: Thematic Pasture Filtering (MapBiomas + Zonal NDVI)
LAPIG - Image Processing and Geoprocessing Laboratory

Supported Official MapBiomas Assets (coverage up to 2025):
  - 10m (Sentinel-2, Collection 4):
    projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1
  - 30m (Landsat, Collection 11):
    projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3

Objective:
Filter and classify active pasture parcels versus other land use classes
(riparian forest/APP, legal reserves, water bodies, crop agriculture) WITHOUT degrading
the straight-fence regularized vector geometry with raster pixel serrations.
"""

import os
import ee
import geemap
import rasterio
import rasterio.mask
import numpy as np
import geopandas as gpd
import shapely

# Official MapBiomas Assets (both with temporal coverage up to 2025)
MAPBIOMAS_10M_COL4 = "projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1"
MAPBIOMAS_30M_COL11 = "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3"

def get_mapbiomas_pasture_ee(roi_ee, year=2025, collection='10m', ee_project=None):
    """
    Retrieves the MapBiomas pasture mask via Google Earth Engine.

    Parameters:
      - roi_ee: ee.FeatureCollection or ee.Geometry defining the area of interest
      - year: Target year for land use classification (up to 2025)
      - collection: '10m' (Collection 4 Sentinel-2) or '30m' (Collection 11 Landsat)

    Class 15 = Pasture in MapBiomas standard classification.
    """
    # Do not reinitialize if Earth Engine is already authenticated in the active session
    is_init = False
    try:
        is_init = ee.data._credentials is not None
    except Exception:
        pass

    if not is_init or (ee_project and str(ee_project).lower() not in ['none', 'auto', 'default']):
        if ee_project and str(ee_project).lower() not in ['none', 'auto', 'default']:
            try:
                ee.Initialize(project=ee_project)
            except Exception:
                try:
                    ee.Initialize()
                except Exception:
                    pass
        else:
            try:
                ee.Initialize()
            except Exception:
                try:
                    ee.Initialize(project='ee-vieiramesquita')
                except Exception:
                    pass

    year = min(int(year), 2025)

    if str(collection).lower() in ['30', '30m', 'col11', 'collection11']:
        asset_id = MAPBIOMAS_30M_COL11
        year = max(year, 1985)
        print(f"Querying MapBiomas 30m (Collection 11) - Year {year}...")
    else:
        asset_id = MAPBIOMAS_10M_COL4
        year = max(year, 2017)
        print(f"Querying MapBiomas 10m (Collection 4) - Year {year}...")

    mb = ee.Image(asset_id)
    band_name = f'classification_{year}'

    # Select target year band
    mb_band = mb.select(band_name)

    # Class 15 = Pasture. Binary mask: 1 where pasture, 0 otherwise
    pasture_mask = mb_band.clip(roi_ee).eq(15).rename('pasture_mask')
    return pasture_mask

def export_pasture_mask_local(pasture_mask_ee, roi_ee, reference_tif, output_tif, scale=10):
    """
    Exports the MapBiomas pasture raster perfectly aligned to CRS and bounding box of S2 mosaic.
    """
    if os.path.exists(output_tif) and os.path.getsize(output_tif) > 1000:
        print(f"MapBiomas mask already exists locally: {output_tif}")
        return output_tif

    with rasterio.open(reference_tif) as ref:
        ref_crs = ref.crs

    print(f"Exporting MapBiomas pasture mask ({scale}m) to: {output_tif}...")
    geemap.ee_export_image(
        pasture_mask_ee,
        output_tif,
        scale=scale,
        crs=ref_crs.to_string(),
        region=roi_ee.geometry(),
        file_per_band=False
    )
    print("MapBiomas mask export completed!")
    return output_tif

def classify_and_filter_pasture_parcels(gdf_parcels, pasture_mask_tif, min_pasture_coverage=0.35, s2_ndvi_tif=None):
    """
    Classifies each vector polygon for pasture dominance via zonal statistical analysis:

    Critical advantage over raster-multiplication approaches:
    - Avoids pixel multiplication (np.where) which destroys tree-lined fence corridors and punches Swiss-cheese holes.
    - Computes coverage percentage (% of MapBiomas Class 15 pixels) inside each parcel boundary.
      If a polygon is predominantly pasture (>= 35-40%), its post-to-post straight fence line is fully preserved.
      Non-pasture polygons (forest reserves, rivers, crop fields) are separated for auditing.
    """
    print(f"\nClassifying and filtering pasture parcels...")
    print(f"   Input vector polygons: {len(gdf_parcels)}")

    with rasterio.open(pasture_mask_tif) as src_mb:
        mb_crs = src_mb.crs

    # Align CRS
    gdf_aligned = gdf_parcels.to_crs(mb_crs).copy()

    pasture_coverages = []
    mean_ndvis = []

    src_s2 = rasterio.open(s2_ndvi_tif) if (s2_ndvi_tif and os.path.exists(s2_ndvi_tif)) else None

    with rasterio.open(pasture_mask_tif) as src_mb:
        for idx, row in gdf_aligned.iterrows():
            geom = [shapely.geometry.mapping(row.geometry)]
            try:
                # Zonal sampling in MapBiomas mask
                mb_crop, _ = rasterio.mask.mask(src_mb, geom, crop=True, nodata=255)
                mb_pixels = mb_crop[0]
                valid = mb_pixels[mb_pixels != 255]
                if len(valid) > 0:
                    cov = float((valid == 1).sum()) / float(len(valid))
                else:
                    cov = 0.0
            except Exception:
                cov = 0.0
            pasture_coverages.append(round(cov, 3))

            # Sample median NDVI if S2 mosaic provided
            if src_s2:
                try:
                    s2_crop, _ = rasterio.mask.mask(src_s2, geom, crop=True, nodata=0)
                    ndvi_pixels = s2_crop[1] # Band 2 = NDVI_median
                    valid_ndvi = ndvi_pixels[ndvi_pixels > 0]
                    m_ndvi = float(valid_ndvi.mean()) if len(valid_ndvi) > 0 else 0.0
                except Exception:
                    m_ndvi = 0.0
                mean_ndvis.append(round(m_ndvi, 3))

    if src_s2:
        src_s2.close()

    gdf_aligned['pasture_cov'] = pasture_coverages
    if mean_ndvis:
        gdf_aligned['ndvi_median'] = mean_ndvis

    # Classification
    gdf_aligned['is_pasture'] = gdf_aligned['pasture_cov'] >= min_pasture_coverage
    gdf_aligned['class_name'] = np.where(gdf_aligned['is_pasture'], 'Pasture', 'Others (Forest/APP/Crops)')

    # Partition into two layers
    gdf_pasture = gdf_aligned[gdf_aligned['is_pasture']].copy().reset_index(drop=True)
    gdf_others = gdf_aligned[~gdf_aligned['is_pasture']].copy().reset_index(drop=True)

    print(f"   Confirmed pasture parcels: {len(gdf_pasture)} ({gdf_pasture['area_ha'].sum():.2f} ha)")
    print(f"   Other land use polygons (forest/APP/infrastructure): {len(gdf_others)} ({gdf_others['area_ha'].sum():.2f} ha)")

    return gdf_pasture, gdf_others
