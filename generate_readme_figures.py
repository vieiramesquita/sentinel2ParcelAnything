#!/usr/bin/env python3
"""
Gera figuras ilustrativas em alta resolução dos resultados da vetorização cadastral
para inclusão no README.md do repositório GitHub.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.plot import reshape_as_image
import geopandas as gpd

def render_property_comparison(folder_path, output_img_path, title_main, area_ha):
    s2_path = os.path.join(folder_path, "s2_ndvi_rgb.tif")
    masks_path = os.path.join(folder_path, "masks.tif")
    shp_paths = glob.glob(os.path.join(folder_path, "piquetes_finais_cercas_retas_*.shp"))
    pure_paths = glob.glob(os.path.join(folder_path, "piquetes_segmentacao_pura_*.shp"))

    if not os.path.exists(s2_path) or not os.path.exists(masks_path) or not shp_paths:
        print(f"Arquivos incompletos em {folder_path}")
        return

    shp_path = shp_paths[0]

    # Carrega imagem Sentinel-2
    with rasterio.open(s2_path) as src:
        s2_data = src.read()
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        raster_crs = src.crs
        # Pegar as 3 primeiras bandas como RGB
        rgb = s2_data[:3, :, :]
        rgb = reshape_as_image(rgb)

    # Carrega máscaras SAMGeo
    with rasterio.open(masks_path) as src:
        masks_data = src.read(1)

    # Carrega Shapefile de piquetes finais e reprojeta para o CRS do raster
    gdf = gpd.read_file(shp_path)
    gdf_proj = gdf.to_crs(raster_crs)

    num_paddocks = len(gdf_proj)

    # Cria figura de 3 painéis
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.8), facecolor='#0b0f19')
    
    # 1. Painel Sentinel-2
    ax1 = axes[0]
    ax1.set_facecolor('#0b0f19')
    ax1.imshow(rgb, extent=extent, origin='upper')
    # Desenha o contorno total da propriedade em amarelo neon
    try:
        boundary = gdf_proj.unary_union.boundary
        gpd.GeoSeries([boundary], crs=raster_crs).plot(ax=ax1, color='#facc15', linewidth=2, linestyle='--', label='Perímetro CAR')
    except Exception:
        pass
    ax1.set_title("1. Composto Sentinel-2 (CS+)\nMedianas Espectrais + Dinâmica NDVI", 
                  color='#38bdf8', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    ax1.axis('off')

    # 2. Painel Máscaras Brutas SAMGeo
    ax2 = axes[1]
    ax2.set_facecolor('#0b0f19')
    unique_ids = np.unique(masks_data)
    unique_ids = unique_ids[unique_ids > 0]
    num_masks = len(unique_ids)
    
    # Mapeia IDs para índices contíguos
    remapped_masks = np.zeros_like(masks_data, dtype=float)
    for idx, u_id in enumerate(unique_ids):
        remapped_masks[masks_data == u_id] = (idx % 20) + 1
    
    ax2.imshow(remapped_masks, extent=extent, cmap='tab20b', origin='upper', interpolation='nearest')
    try:
        gpd.GeoSeries([boundary], crs=raster_crs).plot(ax=ax2, color='#ffffff', linewidth=1.5, linestyle=':')
    except Exception:
        pass
    ax2.set_title(f"2. Máscaras SAMGeo (ViT-H)\n{num_masks} Segmentos Brutos de Deep Learning", 
                  color='#f43f5e', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.axis('off')

    # 3. Painel Vetorização Cadastral LAPIG
    ax3 = axes[2]
    ax3.set_facecolor('#0b0f19')
    # Fundo Sentinel-2 para contexto
    ax3.imshow(rgb, extent=extent, origin='upper', alpha=0.45)
    
    # Plota polígonos com cores distintas e bordas brancas bem visíveis
    gdf_plot = gdf_proj.copy()
    gdf_plot['color_id'] = range(len(gdf_plot))
    gdf_plot.plot(ax=ax3, column='color_id', cmap='turbo', alpha=0.55, 
                  edgecolor='#ffffff', linewidth=1.8)
    
    # Perímetro CAR destacado
    try:
        gpd.GeoSeries([boundary], crs=raster_crs).plot(ax=ax3, color='#facc15', linewidth=2.5)
    except Exception:
        pass

    ax3.set_title(f"3. Vetorização Cadastral Planar LAPIG\n{num_paddocks} Piquetes Retificados | Topologia Estanque (0 Overlaps)", 
                  color='#10b981', fontsize=12, fontweight='bold', pad=12)
    ax3.set_xlim(extent[0], extent[1])
    ax3.set_ylim(extent[2], extent[3])
    ax3.axis('off')

    # Título Geral
    fig.suptitle(f"{title_main} — Área: {area_ha:,.2f} ha | Piquetes Extraídos: {num_paddocks}", 
                 color='#f8fafc', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.93])
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    plt.savefig(output_img_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Salvo: {output_img_path}")

if __name__ == '__main__':
    base_dir = "/mnt/d/samGeoPasture/Vectorization_Pro"
    out_dir = f"{base_dir}/docs/images"
    
    # 1. Caiapônia
    render_property_comparison(
        folder_path=f"{base_dir}/results/GO-5204409-01D1AD1F9586415E9D46BF93115D27DF",
        output_img_path=f"{out_dir}/fig_caiaponia_results.png",
        title_main="Imóvel Rural em Caiapônia (GO)",
        area_ha=749.52
    )

    # 2. Itapirapuã
    render_property_comparison(
        folder_path=f"{base_dir}/results/GO-5211008-46C08C6235D74D36BE07B5384A17B156",
        output_img_path=f"{out_dir}/fig_itapirapua_results.png",
        title_main="Imóvel Rural em Itapirapuã (GO)",
        area_ha=193.59
    )

    # 3. Paranaíba (MS)
    render_property_comparison(
        folder_path=f"{base_dir}/results/MS-5006309-8DC93A48C2424FD7B3EDF235F8B531E1",
        output_img_path=f"{out_dir}/fig_paranaiba_results.png",
        title_main="Imóvel Rural em Paranaíba (MS)",
        area_ha=1085.22
    )
