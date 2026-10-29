"""
Módulo 4: Filtragem Temática de Pastagem (MapBiomas + NDVI Zonal)
LAPIG - Laboratório de Processamento de Imagens e Geoprocessamento

Assets Oficiais MapBiomas Suportados (até 2025):
  - 10m (Sentinel-2, Coleção 4):
    projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1
  - 30m (Landsat, Coleção 11):
    projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3

Objetivo:
Filtrar e classificar o que é parcela de pastagem e o que são outras classes
(mata ciliar/APP, reserva legal, corpos d'água, lavoura) SEM degradar a
geometria retificada dos polígonos com dentes de serra de raster.
"""

import os
import ee
import geemap
import rasterio
import rasterio.mask
import numpy as np
import geopandas as gpd
import shapely

# Assets oficiais MapBiomas (ambos com cobertura até 2025)
MAPBIOMAS_10M_COL4 = "projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1"
MAPBIOMAS_30M_COL11 = "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3"

def get_mapbiomas_pasture_ee(roi_ee, year=2025, collection='10m', ee_project=None):
    """
    Recupera a máscara de pastagem MapBiomas via Google Earth Engine.
    
    Parâmetros:
      - roi_ee: ee.FeatureCollection ou ee.Geometry da área de interesse
      - year: Ano da classificação (até 2025)
      - collection: '10m' (Coleção 4 Sentinel-2) ou '30m' (Coleção 11 Landsat)
      
    Classe 15 = Pastagem no MapBiomas.
    """
    # Não re-inicializa se o Earth Engine já estiver ativo na sessão
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
        print(f"🛰️ Consultando MapBiomas 30m (Coleção 11) - Ano {year}...")
    else:
        asset_id = MAPBIOMAS_10M_COL4
        year = max(year, 2017)
        print(f"🛰️ Consultando MapBiomas 10m (Coleção 4) - Ano {year}...")

    mb = ee.Image(asset_id)
    band_name = f'classification_{year}'

    # Seleciona a banda do ano
    mb_band = mb.select(band_name)

    # Classe 15 = Pastagem. Criamos máscara binária: 1 onde é pastagem, 0 caso contrário
    pasture_mask = mb_band.clip(roi_ee).eq(15).rename('pasture_mask')
    return pasture_mask

def export_pasture_mask_local(pasture_mask_ee, roi_ee, reference_tif, output_tif, scale=10):
    """
    Exporta o raster de pastagem do MapBiomas alinhado perfeitamente ao CRS e extensão do mosaico S2.
    """
    if os.path.exists(output_tif) and os.path.getsize(output_tif) > 1000:
        print(f"ℹ️ Máscara MapBiomas já existe localmente: {output_tif}")
        return output_tif

    with rasterio.open(reference_tif) as ref:
        ref_crs = ref.crs

    print(f"🛰️ Exportando máscara de pastagem MapBiomas ({scale}m) para: {output_tif}...")
    geemap.ee_export_image(
        pasture_mask_ee,
        output_tif,
        scale=scale,
        crs=ref_crs.to_string(),
        region=roi_ee.geometry(),
        file_per_band=False
    )
    print("✅ Exportação da máscara MapBiomas concluída!")
    return output_tif

def classify_and_filter_pasture_parcels(gdf_parcels, pasture_mask_tif, min_pasture_coverage=0.35, s2_ndvi_tif=None):
    """
    Classifica cada polígono vetorial quanto à dominância de pastagem através de análise zonal:
    
    Vantagem crítica sobre a abordagem antiga do notebook:
    - O notebook antigo multiplicava a máscara no raster (final_array = np.where(mb_data == 1, sam_filtered, 0)),
      o que destruía as bordas das cercas onde há árvores plantadas e gerava buracos internos tipo queijo suíço.
    - Esta abordagem calcula o percentual de cobertura de pasto (% de pixels do MapBiomas Classe 15) dentro de cada parcela.
      Se o polígono é majoritariamente pasto (>= 35-40%), sua cerca reta mourão-a-mourão é preservada integralmente.
      Polígonos que são reservas florestais, rios ou sedes são separados para auditoria ou descartados.
    """
    print(f"\n🌾 Classificando e filtrando parcelas de pastagem...")
    print(f"   Polígonos de entrada: {len(gdf_parcels)}")

    with rasterio.open(pasture_mask_tif) as src_mb:
        mb_crs = src_mb.crs

    # Alinhar CRS
    gdf_aligned = gdf_parcels.to_crs(mb_crs).copy()

    pasture_coverages = []
    mean_ndvis = []

    src_s2 = rasterio.open(s2_ndvi_tif) if (s2_ndvi_tif and os.path.exists(s2_ndvi_tif)) else None

    with rasterio.open(pasture_mask_tif) as src_mb:
        for idx, row in gdf_aligned.iterrows():
            geom = [shapely.geometry.mapping(row.geometry)]
            try:
                # Amostragem zonal na máscara MapBiomas
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

            # Amostragem do NDVI mediano se mosaico fornecido
            if src_s2:
                try:
                    s2_crop, _ = rasterio.mask.mask(src_s2, geom, crop=True, nodata=0)
                    ndvi_pixels = s2_crop[1] # Banda 2 = NDVI_median
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

    # Classificação
    gdf_aligned['is_pasture'] = gdf_aligned['pasture_cov'] >= min_pasture_coverage
    gdf_aligned['class_name'] = np.where(gdf_aligned['is_pasture'], 'Pasto', 'Outros (Mata/APP/Lavoura)')

    # Separar em duas camadas
    gdf_pasture = gdf_aligned[gdf_aligned['is_pasture']].copy().reset_index(drop=True)
    gdf_others = gdf_aligned[~gdf_aligned['is_pasture']].copy().reset_index(drop=True)

    print(f"   ✅ Parcelas de pastagem confirmadas: {len(gdf_pasture)} ({gdf_pasture['area_ha'].sum():.2f} ha)")
    print(f"   🌲 Polígonos de outras classes (mata/APP/sede): {len(gdf_others)} ({gdf_others['area_ha'].sum():.2f} ha)")

    return gdf_pasture, gdf_others
