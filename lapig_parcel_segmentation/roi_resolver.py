"""
Módulo 0: Resolvedor Universal de Área de Interesse (ROI)
LAPIG - Laboratório de Processamento de Imagens e Geoprocessamento (UFG)

Permite ao usuário fornecer a área de interesse através de qualquer uma das modalidades:
  1. Arquivo vetorial local (.shp, .gpkg, .fgb, .parquet, .geojson):
     - Suporte a coluna de ID do imóvel (ex: id_column='cod_imovel')
     - Suporte a filtro por valor de ID (ex: id_value='CAR-GO-001')
  2. Asset ou Objeto do Google Earth Engine:
     - 'projects/...' ou ee.FeatureCollection / ee.Feature / ee.Geometry
     - Filtro opcional car_property_filter=('cod_imovel', '...')
  3. Coordenada Geográfica:
     - (lat, lon) -> Busca espacial automática do imóvel na base do CAR do GEE.
     - Extrai o código oficial do CAR automaticamente.
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
    """Gera uma string segura para nomes de pastas e arquivos."""
    if id_val is None:
        return "imovel"
    cleaned = re.sub(r'[\\/*?:"<>|]', '_', str(id_val)).strip()
    return cleaned[:120]

def parse_lat_lon(coord):
    """
    Identifica de forma inteligente latitude e longitude para o território brasileiro.
    Aceita tupla, lista, dict ou string 'lat, lon'.
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
        raise ValueError(f"Formato de coordenada inválido: {coord}")

    # No Brasil: Longitude está entre -74 e -34. Latitude está entre -35 e +6.
    if -75 <= v1 <= -34 and -35 <= v2 <= 6:
        lon, lat = v1, v2
    elif -75 <= v2 <= -34 and -35 <= v1 <= 6:
        lat, lon = v1, v2
    else:
        lat, lon = v1, v2

    return lat, lon

def gdf_to_ee_feature_collection(gdf):
    """Converte um GeoDataFrame local para ee.FeatureCollection."""
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
    id_column=None,            # Coluna que identifica o imóvel no arquivo (ex: 'cod_imovel', 'id')
    id_value=None,             # Valor opcional de ID caso o arquivo tenha múltiplos imóveis
    car_property_filter=None,  # Tupla (campo, valor) para asset do EE
    car_assets=None,
    buffer_radius_meters=1500,
    ee_project=None
):
    """
    Resolve de forma universal o input da Área de Interesse (ROI).
    
    Retorna:
      roi_ee: ee.FeatureCollection pronta para GEE (mosaico S2 e MapBiomas)
      gdf_car: gpd.GeoDataFrame em EPSG:4326 para recorte vetorial local
      info: Dicionário contendo 'id_imovel', 'cod_imovel', 'area_ha', etc.
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

    if car_assets is None:
        car_assets = DEFAULT_CAR_ASSETS

    # -------------------------------------------------------------
    # CASO 1: Objeto Earth Engine já instanciado
    # -------------------------------------------------------------
    if isinstance(input_target, ee.FeatureCollection):
        print("📥 Entrada identificada: ee.FeatureCollection")
        roi_ee = input_target
        import geemap
        gdf_car = geemap.ee_to_gdf(roi_ee).to_crs(epsg=4326)
        
        # Tentar extrair cod_imovel se existir nas colunas
        prop_id = None
        for col in ['cod_imovel', 'codigo', 'id', 'system:index']:
            if col in gdf_car.columns and len(gdf_car) > 0:
                prop_id = str(gdf_car[col].iloc[0])
                break
        prop_id = prop_id or "ee_collection"
        
        info = {
            "tipo": "ee_feature_collection",
            "features": len(gdf_car),
            "id_imovel": prop_id,
            "cod_imovel": prop_id
        }
        return roi_ee, gdf_car, info

    if isinstance(input_target, ee.Feature):
        print("📥 Entrada identificada: ee.Feature")
        roi_ee = ee.FeatureCollection([input_target])
        import geemap
        gdf_car = geemap.ee_to_gdf(roi_ee).to_crs(epsg=4326)
        prop_id = str(gdf_car.get('cod_imovel', ['ee_feature'])[0])
        info = {"tipo": "ee_feature", "features": 1, "id_imovel": prop_id, "cod_imovel": prop_id}
        return roi_ee, gdf_car, info

    if isinstance(input_target, ee.Geometry):
        print("📥 Entrada identificada: ee.Geometry")
        roi_ee = ee.FeatureCollection([ee.Feature(input_target, {})])
        import geemap
        gdf_car = geemap.ee_to_gdf(roi_ee).to_crs(epsg=4326)
        info = {"tipo": "ee_geometry", "features": 1, "id_imovel": "ee_geometry", "cod_imovel": "ee_geometry"}
        return roi_ee, gdf_car, info

    # -------------------------------------------------------------
    # CASO 2: Arquivo Vetorial Local ou String de Caminho
    # -------------------------------------------------------------
    if isinstance(input_target, str):
        target_str = input_target.strip()

        # Verifica se é um arquivo local existente
        if os.path.exists(target_str):
            ext = os.path.splitext(target_str)[1].lower()
            print(f"📥 Entrada identificada: Arquivo Vetorial Local ({ext}) -> {os.path.basename(target_str)}")
            
            if ext in ['.parquet', '.geoparquet']:
                gdf = gpd.read_parquet(target_str)
            else:
                gdf = gpd.read_file(target_str)

            if gdf.crs is None:
                print("⚠️ Vetor sem CRS definido. Assumindo EPSG:4326 (WGS84).")
                gdf = gdf.set_crs(epsg=4326)
            else:
                gdf = gdf.to_crs(epsg=4326)

            # Validação e seleção da coluna de ID
            prop_id = None
            if id_column is not None:
                if id_column not in gdf.columns:
                    cols_str = ", ".join([f"'{c}'" for c in gdf.columns if c != 'geometry'])
                    raise ValueError(
                        f"❌ Coluna de ID '{id_column}' não encontrada no arquivo.\n"
                        f"   Colunas disponíveis no arquivo: [{cols_str}]"
                    )

                if id_value is not None:
                    gdf = gdf[gdf[id_column].astype(str) == str(id_value)].copy().reset_index(drop=True)
                    if len(gdf) == 0:
                        raise ValueError(f"❌ Nenhum registro encontrado com {id_column} == '{id_value}'.")
                    prop_id = str(id_value)
                    print(f"🎯 Imóvel filtrado pela coluna '{id_column}' == '{prop_id}'")
                else:
                    prop_id = str(gdf[id_column].iloc[0])
                    if len(gdf) > 1:
                        print(f"ℹ️ O arquivo possui {len(gdf)} imóveis. Usando o 1º registro ('{prop_id}'). Dica: use id_value='...' para selecionar.")
                    else:
                        print(f"🎯 Imóvel identificado pela coluna '{id_column}': {prop_id}")
            else:
                # Tentativa de auto-detecção de colunas comuns de CAR ou ID
                candidate_cols = ['cod_imovel', 'codigo_car', 'codigo', 'id', 'gid', 'recibo', 'num_car', 'cod_prop', 'name', 'nome']
                detected = [c for c in candidate_cols if c in gdf.columns]
                if detected:
                    auto_col = detected[0]
                    prop_id = str(gdf[auto_col].iloc[0])
                    print(f"ℹ️ Coluna de ID auto-detectada: '{auto_col}' -> ID do imóvel: {prop_id}")
                else:
                    prop_id = os.path.splitext(os.path.basename(target_str))[0]
                    print(f"ℹ️ Nenhuma coluna de ID informada. Usando nome do arquivo como ID: '{prop_id}'")
                    cols_str = ", ".join([f"'{c}'" for c in gdf.columns if c != 'geometry'])
                    print(f"   (Dica: você pode informar id_column='...' dentre as colunas: [{cols_str}])")

            roi_ee = gdf_to_ee_feature_collection(gdf)
            area_ha = round(gdf.to_crs(gdf.estimate_utm_crs()).geometry.area.sum() / 10000.0, 2)
            info = {
                "tipo": f"arquivo_local_{ext[1:]}",
                "caminho": target_str,
                "features": len(gdf),
                "id_imovel": prop_id,
                "cod_imovel": prop_id,
                "area_ha": area_ha
            }
            return roi_ee, gdf, info

        # Verifica se é um asset path do Earth Engine (ex: 'projects/...')
        if target_str.startswith('projects/') or target_str.startswith('users/'):
            print(f"📥 Entrada identificada: Asset do Earth Engine -> {target_str}")
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
                "tipo": "ee_asset",
                "asset_id": target_str,
                "features": len(gdf),
                "id_imovel": prop_id,
                "cod_imovel": prop_id
            }
            return fc, gdf, info

        # Se for string de coordenada: "-15.747, -50.894"
        if ',' in target_str or ';' in target_str:
            input_target = target_str

    # -------------------------------------------------------------
    # CASO 3: Coordenada Geográfica (Tupla, Lista, Dict ou String)
    # -------------------------------------------------------------
    if isinstance(input_target, (list, tuple, dict)) or (isinstance(input_target, str) and (',' in input_target or ';' in input_target)):
        lat, lon = parse_lat_lon(input_target)
        print(f"📥 Entrada identificada: Coordenada [Lat: {lat:.5f}, Lon: {lon:.5f}]")
        print("🔍 Buscando sobreposição com a base do CAR no Earth Engine...")

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
            except Exception as e:
                pass

        if matched_fc is not None:
            import geemap
            gdf_car = geemap.ee_to_gdf(matched_fc).to_crs(epsg=4326)
            area_ha = round(gdf_car.to_crs(gdf_car.estimate_utm_crs()).geometry.area.sum() / 10000.0, 2)
            prop_id = str(matched_cod or "car_imovel")
            print(f"🎯 Imóvel CAR Localizado com Sucesso!")
            print(f"   Código CAR:  {prop_id}")
            print(f"   Asset CAR:   {matched_asset}")
            print(f"   Área Imóvel: {area_ha:.2f} ha")
            info = {
                "tipo": "coordenada_car_interceptado",
                "coord": (lat, lon),
                "id_imovel": prop_id,
                "cod_imovel": prop_id,
                "asset_car": matched_asset,
                "area_ha": area_ha
            }
            return matched_fc, gdf_car, info
        else:
            print(f"⚠️ Nenhum imóvel CAR interceptou a coordenada nos assets cadastrados.")
            print(f"🌐 Gerando Área de Interesse (AOI) com buffer de {buffer_radius_meters}m ao redor do ponto...")
            aoi_geom = pt_ee.buffer(buffer_radius_meters).bounds()
            roi_ee = ee.FeatureCollection([ee.Feature(aoi_geom, {'tipo': 'buffer_coord'})])
            import geemap
            gdf_car = geemap.ee_to_gdf(roi_ee).to_crs(epsg=4326)
            area_ha = round(gdf_car.to_crs(gdf_car.estimate_utm_crs()).geometry.area.sum() / 10000.0, 2)
            prop_id = f"coord_{lat:.4f}_{lon:.4f}"
            info = {
                "tipo": "coordenada_buffer_aoi",
                "coord": (lat, lon),
                "id_imovel": prop_id,
                "cod_imovel": prop_id,
                "raio_metros": buffer_radius_meters,
                "area_ha": area_ha
            }
            return roi_ee, gdf_car, info

    raise ValueError(f"Não foi possível resolver o input_target: {type(input_target)} -> {input_target}")
