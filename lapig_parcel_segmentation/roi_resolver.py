"""
Module 0: Universal Region of Interest (ROI) Resolver
LAPIG - Image Processing and Geoprocessing Laboratory (UFG)

Allows users to define the area of interest through any of the supported modalities:
  1. Local vector file (.shp, .gpkg, .fgb, .parquet, .geojson):
     - Support for property ID column (e.g., id_column='cod_imovel')
     - Support for specific ID filtering (e.g., id_value='CAR-GO-001')
  2. Google Earth Engine Asset or Object:
     - 'projects/...' or ee.FeatureCollection / ee.Feature / ee.Geometry
     - Optional filter car_property_filter=('cod_imovel', '...')
  3. Geographic Coordinates:
     - (lat, lon) -> Automatic spatial intersection with CAR assets in GEE.
     - Automatically extracts the official CAR property code.
"""

import os
import re
import json
import shapely
import geopandas as gpd
import numpy as np
import ee

DEFAULT_CAR_ASSETS = [
    'projects/ee-vieiramesquita/assets/CARs/CAR_GO',
    'projects/ee-vieiramesquita/assets/CARs/CAR_MS',
    'projects/ee-vieiramesquita/assets/CARs/CAR_MT',
]

def sanitize_id(id_val):
    """Generates a filesystem-safe string for directory and file names."""
    if id_val is None:
        return "property"
    cleaned = re.sub(r'[\\/*?:"<>|]', '_', str(id_val)).strip()
    return cleaned[:120]

def parse_lat_lon(coord):
    """
    Intelligently identifies latitude and longitude within Brazilian territory.
    Accepts tuple, list, dict, or 'lat, lon' string.
    """
    if isinstance(coord, str):
        parts = [float(p.strip()) for p in coord.replace(';', ',').split(',') if p.strip()]
        v1, v2 = parts[0], parts[1]
    elif isinstance(coord, dict):
        v1 = coord.get('lat', coord.get('latitude'))
        v2 = coord.get('lon', coord.get('lng', coord.get('longitude')))
        return float(v1), float(v2)
    elif isinstance(coord, (list, tuple)):
        v1, v2 = float(coord[0]), float(coord[1])
    else:
        raise ValueError(f"Invalid coordinate format: {coord}")

    # Brazil geographic boundaries: Longitude is between -75 and -34. Latitude is between -35 and +6.
    if -75 <= v1 <= -34 and -35 <= v2 <= 6:
        lon, lat = v1, v2
    elif -75 <= v2 <= -34 and -35 <= v1 <= 6:
        lat, lon = v1, v2
    else:
        lat, lon = v1, v2

    return lat, lon

def gdf_to_ee_feature_collection(gdf):
    """Converts a local GeoDataFrame to ee.FeatureCollection."""
    gdf_4326 = gdf.to_crs(epsg=4326)
    features = []
    for idx, row in gdf_4326.iterrows():
        geom_json = shapely.geometry.mapping(row.geometry)
        props = {}
        for k, v in row.items():
            if k == 'geometry':
                continue
            if isinstance(v, (str, int, bool)):
                props[k] = v
            elif isinstance(v, float) and not np.isnan(v):
                props[k] = v
        features.append(ee.Feature(ee.Geometry(geom_json), props))
    return ee.FeatureCollection(features)

def resolve_input_roi(
    input_target,
    id_column=None,            # Column identifying rural property in the vector file (e.g., 'cod_imovel', 'id')
    id_value=None,             # Optional specific ID filter
    car_property_filter=None,  # Tuple (field, value) for GEE asset
    car_assets=None,
    buffer_radius_meters=1500,
    ee_project=None
):
    """
    Universally resolves input target into an Area of Interest (ROI).

    Returns:
      roi_ee: ee.FeatureCollection ready for GEE (S2 mosaic and MapBiomas)
      gdf_car: gpd.GeoDataFrame in EPSG:4326 for local vector clipping
      info: Dictionary containing 'id_imovel', 'cod_imovel', 'area_ha', etc.
    """
    # Do not reinitialize if Earth Engine is already active in session
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

    if car_assets is None:
        car_assets = DEFAULT_CAR_ASSETS

    # -------------------------------------------------------------
    # CASE 1: Earth Engine Object already instantiated
    # -------------------------------------------------------------
    if isinstance(input_target, ee.FeatureCollection):
        print("Input identified: ee.FeatureCollection")
        roi_ee = input_target
        import geemap
        gdf_car = geemap.ee_to_gdf(roi_ee).to_crs(epsg=4326)

        prop_id = None
        for col in ['cod_imovel', 'codigo', 'id', 'system:index']:
            if col in gdf_car.columns and len(gdf_car) > 0:
                prop_id = str(gdf_car[col].iloc[0])
                break
        prop_id = prop_id or "ee_collection"

        info = {
            "type": "ee_feature_collection",
            "features": len(gdf_car),
            "id_imovel": prop_id,
            "cod_imovel": prop_id
        }
        return roi_ee, gdf_car, info

    if isinstance(input_target, ee.Feature):
        print("Input identified: ee.Feature")
        roi_ee = ee.FeatureCollection([input_target])
        import geemap
        gdf_car = geemap.ee_to_gdf(roi_ee).to_crs(epsg=4326)
        prop_id = str(gdf_car.get('cod_imovel', ['ee_feature'])[0])
        info = {"type": "ee_feature", "features": 1, "id_imovel": prop_id, "cod_imovel": prop_id}
        return roi_ee, gdf_car, info

    if isinstance(input_target, ee.Geometry):
        print("Input identified: ee.Geometry")
        roi_ee = ee.FeatureCollection([ee.Feature(input_target, {})])
        import geemap
        gdf_car = geemap.ee_to_gdf(roi_ee).to_crs(epsg=4326)
        info = {"type": "ee_geometry", "features": 1, "id_imovel": "ee_geometry", "cod_imovel": "ee_geometry"}
        return roi_ee, gdf_car, info

    # -------------------------------------------------------------
    # CASE 2: Local Vector File or Path String
    # -------------------------------------------------------------
    if isinstance(input_target, str):
        target_str = input_target.strip()

        # Check if local file exists
        if os.path.exists(target_str):
            ext = os.path.splitext(target_str)[1].lower()
            print(f"Input identified: Local Vector File ({ext}) -> {os.path.basename(target_str)}")

            if ext in ['.parquet', '.geoparquet']:
                gdf = gpd.read_parquet(target_str)
            else:
                gdf = gpd.read_file(target_str)

            if gdf.crs is None:
                print("Warning: Vector file has no defined CRS. Assuming EPSG:4326 (WGS84).")
                gdf = gdf.set_crs(epsg=4326)
            else:
                gdf = gdf.to_crs(epsg=4326)

            # Property ID validation and filtering
            prop_id = None
            if id_column is not None:
                if id_column not in gdf.columns:
                    cols_str = ", ".join([f"'{c}'" for c in gdf.columns if c != 'geometry'])
                    raise ValueError(
                        f"[-] ID column '{id_column}' not found in file.\n"
                        f"    Available columns: [{cols_str}]"
                    )

                if id_value is not None:
                    gdf = gdf[gdf[id_column].astype(str) == str(id_value)].copy().reset_index(drop=True)
                    if len(gdf) == 0:
                        raise ValueError(f"[-] No record found with {id_column} == '{id_value}'.")
                    prop_id = str(id_value)
                    print(f"Property filtered by column '{id_column}' == '{prop_id}'")
                else:
                    prop_id = str(gdf[id_column].iloc[0])
                    if len(gdf) > 1:
                        print(f"File contains {len(gdf)} properties. Using 1st record ('{prop_id}'). Tip: use id_value='...' to select.")
                    else:
                        print(f"Property identified by column '{id_column}': {prop_id}")
            else:
                # Auto-detect common CAR / ID column names
                candidate_cols = ['cod_imovel', 'codigo_car', 'codigo', 'id', 'gid', 'recibo', 'num_car', 'cod_prop', 'name', 'nome']
                detected = [c for c in candidate_cols if c in gdf.columns]
                if detected:
                    auto_col = detected[0]
                    prop_id = str(gdf[auto_col].iloc[0])
                    print(f"Auto-detected ID column: '{auto_col}' -> Property ID: {prop_id}")
                else:
                    prop_id = os.path.splitext(os.path.basename(target_str))[0]
                    print(f"No ID column provided. Using filename as ID: '{prop_id}'")
                    cols_str = ", ".join([f"'{c}'" for c in gdf.columns if c != 'geometry'])
                    print(f"   (Tip: specify id_column='...' from available columns: [{cols_str}])")

            roi_ee = gdf_to_ee_feature_collection(gdf)
            area_ha = round(gdf.to_crs(gdf.estimate_utm_crs()).geometry.area.sum() / 10000.0, 2)
            info = {
                "type": f"local_file_{ext[1:]}",
                "path": target_str,
                "features": len(gdf),
                "id_imovel": prop_id,
                "cod_imovel": prop_id,
                "area_ha": area_ha
            }
            return roi_ee, gdf, info

        # Check if Earth Engine asset path (e.g., 'projects/...')
        if target_str.startswith('projects/') or target_str.startswith('users/'):
            print(f"Input identified: Earth Engine Asset -> {target_str}")
            fc = ee.FeatureCollection(target_str)
            prop_id = None
            if car_property_filter:
                field, val = car_property_filter
                fc = fc.filter(ee.Filter.eq(field, val))
                prop_id = str(val)
            import geemap
            gdf = geemap.ee_to_gdf(fc).to_crs(epsg=4326)
            if prop_id is None and 'cod_imovel' in gdf.columns and len(gdf) > 0:
                prop_id = str(gdf['cod_imovel'].iloc[0])
            prop_id = prop_id or os.path.basename(target_str)
            info = {
                "type": "ee_asset",
                "asset_id": target_str,
                "features": len(gdf),
                "id_imovel": prop_id,
                "cod_imovel": prop_id
            }
            return fc, gdf, info

        # If coordinate string: "-15.747, -50.894"
        if ',' in target_str or ';' in target_str:
            input_target = target_str

    # -------------------------------------------------------------
    # CASE 3: Geographic Coordinates (Tuple, List, Dict, or String)
    # -------------------------------------------------------------
    if isinstance(input_target, (list, tuple, dict)) or (isinstance(input_target, str) and (',' in input_target or ';' in input_target)):
        lat, lon = parse_lat_lon(input_target)
        print(f"Input identified: Geographic Coordinate [Lat: {lat:.5f}, Lon: {lon:.5f}]")
        print("Searching spatial intersection with CAR assets in Earth Engine...")

        pt_ee = ee.Geometry.Point([lon, lat])
        matched_fc = None
        matched_cod = None
        matched_asset = None

        for asset_path in car_assets:
            try:
                fc = ee.FeatureCollection(asset_path)
                found = fc.filterBounds(pt_ee)
                if found.size().getInfo() > 0:
                    matched_fc = found
                    matched_asset = asset_path
                    first_feat = found.first()
                    props = first_feat.propertyNames().getInfo()
                    if 'cod_imovel' in props:
                        matched_cod = first_feat.get('cod_imovel').getInfo()
                    break
            except Exception:
                pass

        if matched_fc is not None:
            import geemap
            gdf_car = geemap.ee_to_gdf(matched_fc).to_crs(epsg=4326)
            area_ha = round(gdf_car.to_crs(gdf_car.estimate_utm_crs()).geometry.area.sum() / 10000.0, 2)
            prop_id = str(matched_cod or "car_property")
            print(f"CAR Property Successfully Located!")
            print(f"   CAR Code:      {prop_id}")
            print(f"   CAR Asset:     {matched_asset}")
            print(f"   Property Area: {area_ha:.2f} ha")
            info = {
                "type": "intersected_car_coordinate",
                "coord": (lat, lon),
                "id_imovel": prop_id,
                "cod_imovel": prop_id,
                "asset_car": matched_asset,
                "area_ha": area_ha
            }
            return matched_fc, gdf_car, info
        else:
            print("Warning: No CAR property intersected the coordinate in the configured assets.")
            print(f"Generating Area of Interest (AOI) bounding box with {buffer_radius_meters}m buffer around point...")
            aoi_geom = pt_ee.buffer(buffer_radius_meters).bounds()
            roi_ee = ee.FeatureCollection([ee.Feature(aoi_geom, {'tipo': 'buffer_coord'})])
            import geemap
            gdf_car = geemap.ee_to_gdf(roi_ee).to_crs(epsg=4326)
            area_ha = round(gdf_car.to_crs(gdf_car.estimate_utm_crs()).geometry.area.sum() / 10000.0, 2)
            prop_id = f"coord_{lat:.4f}_{lon:.4f}"
            info = {
                "type": "coordinate_buffer_aoi",
                "coord": (lat, lon),
                "id_imovel": prop_id,
                "cod_imovel": prop_id,
                "buffer_radius_meters": buffer_radius_meters,
                "area_ha": area_ha
            }
            return roi_ee, gdf_car, info

    raise ValueError(f"Could not resolve input_target: {type(input_target)} -> {input_target}")
