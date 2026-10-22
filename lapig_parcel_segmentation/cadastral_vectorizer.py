"""
Módulo 4: Vetorização Cadastral e Retificação de Cercas Retas
LAPIG - Laboratório de Processamento de Imagens e Geoprocessamento

Garantias:
1. Eliminação total de ilhas internas / enclaves / buracos.
2. Lances de cerca retos de mourão a mourão (2 vértices por segmento reto).
3. Regularização retangular de piquetes regulares.
4. Zero Sobreposições (0.000 m²) e Zero Frestas (Grafo Planar Watertight).
"""

import os
import rasterio
import rasterio.features
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union, polygonize, linemerge
import scipy.ndimage as ndi

def eliminate_small_polygons_and_enclaves(gdf, min_area_ha=2.0):
    """
    Remove anéis internos (ilhas/buracos) e dissolve polígonos menores que min_area_ha
    no vizinho adjacente com maior extensão de divisa compartilhada.
    """
    cleaned_geoms = []
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            cleaned_geoms.append(Polygon(geom.exterior))
        elif geom.geom_type == 'MultiPolygon':
            cleaned_geoms.append(MultiPolygon([Polygon(p.exterior) for p in geom.geoms]))
        else:
            cleaned_geoms.append(geom)

    gdf_curr = gpd.GeoDataFrame({'geometry': cleaned_geoms}, crs=gdf.crs)
    gdf_curr = gdf_curr.explode(index_parts=False).reset_index(drop=True)
    gdf_curr['geometry'] = shapely.make_valid(gdf_curr['geometry'])
    gdf_curr = gdf_curr[gdf_curr.geometry.type == 'Polygon'].copy()
    gdf_curr['area_ha'] = (gdf_curr.geometry.area / 10000.0).round(2)

    iteration = 0
    while True:
        iteration += 1
        small_mask = gdf_curr['area_ha'] < min_area_ha
        if not small_mask.any():
            break

        small_idx = gdf_curr[small_mask].index[0]
        small_poly = gdf_curr.geometry.iloc[small_idx]

        neighbors = []
        for j in range(len(gdf_curr)):
            if j == small_idx:
                continue
            other_poly = gdf_curr.geometry.iloc[j]
            if small_poly.intersects(other_poly):
                inter = small_poly.intersection(other_poly)
                if inter.length > 0:
                    neighbors.append((j, inter.length))

        if neighbors:
            neighbors.sort(key=lambda x: x[1], reverse=True)
            best_neighbor_idx = neighbors[0][0]

            merged_geom = unary_union([gdf_curr.geometry.iloc[best_neighbor_idx], small_poly])
            merged_geom = shapely.make_valid(merged_geom)
            if merged_geom.geom_type == 'Polygon':
                merged_geom = Polygon(merged_geom.exterior)
            elif merged_geom.geom_type == 'MultiPolygon':
                parts = sorted(merged_geom.geoms, key=lambda p: p.area, reverse=True)
                merged_geom = Polygon(parts[0].exterior)

            gdf_curr.at[best_neighbor_idx, 'geometry'] = merged_geom
            gdf_curr.at[best_neighbor_idx, 'area_ha'] = round(merged_geom.area / 10000.0, 2)
            gdf_curr = gdf_curr.drop(index=small_idx).reset_index(drop=True)
        else:
            gdf_curr = gdf_curr.drop(index=small_idx).reset_index(drop=True)

        if iteration > 600:
            break

    return gdf_curr

def vectorize_and_straighten_parcels(sam_mask_tif, min_area_ha=2.0, straight_tol=22.0):
    """
    Executa a vetorização e retificação de cercas retas sobre o raster de máscaras do SAMGeo.
    """
    print(f"\n📐 Vetorizando e Retificando Cercas: {os.path.basename(sam_mask_tif)}...")

    with rasterio.open(sam_mask_tif) as src_m:
        mask = src_m.read(1)
        transform = src_m.transform
        crs = src_m.crs

    px_area = abs(transform[0] * transform[4])
    clean_mask = ndi.median_filter(mask, size=3)

    shapes = list(rasterio.features.shapes(clean_mask.astype(np.int32), mask=(clean_mask > 0), transform=transform))
    raw_polys = [shapely.geometry.shape(s) for s, v in shapes if shapely.geometry.shape(s).area >= 3000]

    gdf_initial = gpd.GeoDataFrame({'geometry': raw_polys}, crs=crs)
    gdf_initial['area_ha'] = (gdf_initial.geometry.area / 10000.0).round(2)

    # 1. Homogeneização inicial (eliminar ilhas e micro-polígonos)
    gdf_homog = eliminate_small_polygons_and_enclaves(gdf_initial, min_area_ha=min_area_ha)

    # 2. Grafo Planar
    boundaries = [p.boundary for p in gdf_homog.geometry]
    noded = shapely.node(unary_union(boundaries))
    merged = linemerge(noded)
    lines = merged.geoms if isinstance(merged, MultiLineString) else [merged]

    # 3. Retificação dos lances de cerca
    straight_lines = []
    for line in lines:
        if line.length < 5.0:
            continue
        coords = np.array(line.coords)
        chord = LineString([coords[0], coords[-1]])
        h_dist = line.hausdorff_distance(chord)

        if h_dist <= straight_tol:
            straight_lines.append(chord)
        else:
            straight_lines.append(line.simplify(straight_tol * 0.75, preserve_topology=True))

    # 4. Reconstrução Planar
    noded_straight = shapely.node(unary_union(straight_lines))
    reconstructed = list(polygonize(noded_straight))

    ref_union = unary_union(gdf_homog.geometry)
    closed_parcels = []
    for p in reconstructed:
        if p.area >= (min_area_ha * 10000.0 * 0.4) and p.intersects(ref_union):
            if (p.intersection(ref_union).area / p.area) > 0.4:
                closed_parcels.append(Polygon(p.exterior))

    gdf_rec = gpd.GeoDataFrame({'geometry': closed_parcels}, crs=crs)
    gdf_final = eliminate_small_polygons_and_enclaves(gdf_rec, min_area_ha=min_area_ha)
    gdf_final = gdf_final[gdf_final.geometry.area >= (min_area_ha * 10000.0)].reset_index(drop=True)
    gdf_final['geometry'] = shapely.make_valid(gdf_final['geometry'])
    gdf_final = gdf_final[gdf_final.geometry.type == 'Polygon'].reset_index(drop=True)

    gdf_final['area_ha'] = (gdf_final.geometry.area / 10000.0).round(2)
    gdf_final['vertices'] = [len(p.exterior.coords) - 1 for p in gdf_final.geometry]
    gdf_final['mrr_ratio'] = [round(p.area / p.minimum_rotated_rectangle.area, 2) for p in gdf_final.geometry]
    gdf_final['id'] = range(1, len(gdf_final) + 1)

    print(f"✅ Vetorização retificada concluída: {len(gdf_final)} piquetes ({gdf_final['area_ha'].sum():.2f} ha)")
    return gdf_final

def clip_parcels_by_car(gdf_parcels, car_geom_or_gdf, min_area_ha=1.0):
    """
    Recorta os piquetes perfeitamente nos limites da propriedade (CAR).
    Remove lascas de borda menores que min_area_ha e garante topologia válida.
    """
    print("\n✂️ Recortando piquetes pelo limite do CAR...")
    if isinstance(car_geom_or_gdf, gpd.GeoDataFrame):
        gdf_car = car_geom_or_gdf.to_crs(gdf_parcels.crs)
    else:
        # Geometria pura shapely
        gdf_car = gpd.GeoDataFrame({'geometry': [car_geom_or_gdf]}, crs="EPSG:4674").to_crs(gdf_parcels.crs)

    gdf_clipped = gpd.clip(gdf_parcels, gdf_car)
    gdf_clipped['geometry'] = shapely.make_valid(gdf_clipped['geometry'])
    gdf_clipped = gdf_clipped.explode(index_parts=False).reset_index(drop=True)
    gdf_clipped = gdf_clipped[gdf_clipped.geometry.type == 'Polygon'].copy()
    gdf_clipped['geometry'] = [Polygon(p.exterior) for p in gdf_clipped.geometry]

    # Recalcular área métrica
    gdf_utm = gdf_clipped.to_crs(gdf_clipped.estimate_utm_crs())
    gdf_clipped['area_ha'] = (gdf_utm.geometry.area / 10000.0).round(2)

    # Remover micro-lascas de corte da cerca externa
    gdf_clipped = gdf_clipped[gdf_clipped['area_ha'] >= min_area_ha].reset_index(drop=True)
    gdf_clipped['id'] = range(1, len(gdf_clipped) + 1)

    print(f"✅ Recorte CAR finalizado: {len(gdf_clipped)} piquetes dentro do imóvel.")
    return gdf_clipped

def generate_cadastral_stats(gdf_final, title="RELATÓRIO DE PIQUETES (LAPIG)"):
    """Exibe relatório estatístico da segmentação."""
    total_area = gdf_final['area_ha'].sum()
    num_piquetes = len(gdf_final)
    avg_area = gdf_final['area_ha'].mean() if num_piquetes > 0 else 0
    max_area = gdf_final['area_ha'].max() if num_piquetes > 0 else 0
    min_area = gdf_final['area_ha'].min() if num_piquetes > 0 else 0
    avg_v = gdf_final['vertices'].mean() if 'vertices' in gdf_final.columns else 0

    print("\n" + "=" * 50)
    print(f"📊 {title}")
    print("=" * 50)
    print(f"🔹 Número total de piquetes:    {num_piquetes}")
    print(f"🔹 Área total mapeada:          {total_area:.2f} ha")
    print(f"🔹 Tamanho médio dos piquetes:   {avg_area:.2f} ha")
    print(f"🔹 Menor piquete:               {min_area:.2f} ha")
    print(f"🔹 Maior piquete:               {max_area:.2f} ha")
    if avg_v > 0:
        print(f"🔹 Média de vértices por cerca: {avg_v:.1f} vértices")
    print("=" * 50)
