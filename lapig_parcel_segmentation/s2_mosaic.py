"""
Módulo 1: Download e Processamento Sentinel-2 via Google Earth Engine
Coleção Sentinel-2 Harmonized + Cloud Score Plus (CS+)
LAPIG - Laboratório de Processamento de Imagens e Geoprocessamento
"""

import os
import ee
import geemap

def is_ee_initialized():
    """Verifica se o Earth Engine já foi inicializado na sessão atual."""
    try:
        return ee.data._credentials is not None
    except Exception:
        return False

def initialize_earth_engine(project=None):
    """
    Inicializa o Earth Engine respeitando a sessão ativa do usuário.
    Se já estiver inicializado e nenhum projeto diferente for forçado, mantém a sessão ativa.
    """
    if is_ee_initialized() and (project is None or str(project).lower() in ['none', 'auto', 'default']):
        return

    # Se um projeto foi explicitamente fornecido
    if project and str(project).lower() not in ['none', 'auto', 'default']:
        try:
            ee.Initialize(project=project)
            return
        except Exception:
            pass

    # Tenta inicialização padrão do ambiente do usuário
    try:
        ee.Initialize()
        return
    except Exception:
        pass

    # Variável de ambiente EE_PROJECT
    env_p = os.environ.get('EE_PROJECT') or os.environ.get('EARTHENGINE_PROJECT')
    if env_p:
        try:
            ee.Initialize(project=env_p)
            return
        except Exception:
            pass

    # Fallback
    try:
        ee.Initialize(project='ee-vieiramesquita')
    except Exception:
        pass

def get_sentinel2_ndvi_mosaic(roi_ee, year, start_month=3, end_month=10, buffer_meters=500, output_path=None, ee_project=None):
    """
    Gera o mosaico multi-temporal de estatísticas de NDVI [max, median, min]
    com máscara de nuvens Cloud Score Plus (cs_cdf >= 0.65).
    
    Retorna:
      image_ndvis: ee.Image com as bandas ['NDVI_max', 'NDVI_median', 'NDVI_min']
      outpath: caminho local do arquivo GeoTIFF gerado
      roi_buffered: geometria expandida com o buffer
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
        print(f"🛰️ Exportando mosaico NDVI Sentinel-2 para: {output_path}...")
        geemap.ee_to_geotiff(
            image_ndvis,
            output_path,
            bbox,
            zoom=14,
            vis_params={'min': [0.3, 0.15, 0.1], 'max': [0.9, 0.9, 0.7], 'bands': ['NDVI_max', 'NDVI_median', 'NDVI_min']}
        )
        print("✅ Download do mosaico S2 concluído!")
    else:
        print(f"ℹ️ Mosaico S2 já existe localmente: {output_path}")

    return image_ndvis, output_path, roi_buffered
