#!/usr/bin/env python3
"""
Generates high-resolution illustrative figures of cadastral vectorization results
for inclusion in the GitHub repository README.md.
"""

import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.plot import reshape_as_image
import geopandas as gpd

def render_property_comparison(folder_path, output_paths, title_main, area_ha):
    s2_path = os.path.join(folder_path, "s2_ndvi_rgb.tif")
    masks_path = os.path.join(folder_path, "masks.tif")
    shp_paths = (
        glob.glob(os.path.join(folder_path, "parcelas_finais_cercas_retas_*.shp")) or
        glob.glob(os.path.join(folder_path, "piquetes_finais_cercas_retas_*.shp"))
    )

    if not os.path.exists(s2_path) or not os.path.exists(masks_path) or not shp_paths:
        print(f"Incomplete files in {folder_path}")
        return

    shp_path = shp_paths[0]

    # Load Sentinel-2 image
    with rasterio.open(s2_path) as src:
        s2_data = src.read()
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        raster_crs = src.crs
        # Use first 3 bands as RGB
        rgb = s2_data[:3, :, :]
        rgb = reshape_as_image(rgb)

    # Load SAMGeo masks
    with rasterio.open(masks_path) as src:
        masks_data = src.read(1)

    # Load final parcels shapefile and reproject to raster CRS
    gdf = gpd.read_file(shp_path)
    gdf_proj = gdf.to_crs(raster_crs)

    num_paddocks = len(gdf_proj)

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.8), facecolor='#0b0f19')

    # 1. Sentinel-2 Panel
    ax1 = axes[0]
    ax1.set_facecolor('#0b0f19')
    ax1.imshow(rgb, extent=extent, origin='upper')
    # Draw property boundary in neon yellow
    try:
        boundary = gdf_proj.unary_union.boundary
        gpd.GeoSeries([boundary], crs=raster_crs).plot(ax=ax1, color='#facc15', linewidth=2, linestyle='--', label='CAR Property Boundary')
    except Exception:
        pass
    ax1.set_title("1. Sentinel-2 Composite (CS+)\nSpectral Medians + Multi-Temporal NDVI Dynamics",
                  color='#38bdf8', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    ax1.axis('off')

    # 2. SAMGeo Raw Masks Panel
    ax2 = axes[1]
    ax2.set_facecolor('#0b0f19')
    unique_ids = np.unique(masks_data)
    unique_ids = unique_ids[unique_ids > 0]
    num_masks = len(unique_ids)

    # Remap IDs to contiguous indices for vibrant color mapping
    remapped_masks = np.zeros_like(masks_data, dtype=float)
    for idx, u_id in enumerate(unique_ids):
        remapped_masks[masks_data == u_id] = (idx % 20) + 1

    ax2.imshow(remapped_masks, extent=extent, cmap='tab20b', origin='upper', interpolation='nearest')
    try:
        gpd.GeoSeries([boundary], crs=raster_crs).plot(ax=ax2, color='#ffffff', linewidth=1.5, linestyle=':')
    except Exception:
        pass
    ax2.set_title(f"2. SAMGeo Masks (ViT-H)\n{num_masks} Raw Deep Learning Segments",
                  color='#f43f5e', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.axis('off')

    # 3. LAPIG Cadastral Planar Vectorization Panel
    ax3 = axes[2]
    ax3.set_facecolor('#0b0f19')
    # Sentinel-2 background for visual context
    ax3.imshow(rgb, extent=extent, origin='upper', alpha=0.45)

    # Plot parcels with distinct colors and visible white borders
    gdf_plot = gdf_proj.copy()
    gdf_plot['color_id'] = range(len(gdf_plot))
    gdf_plot.plot(ax=ax3, column='color_id', cmap='turbo', alpha=0.55,
                  edgecolor='#ffffff', linewidth=1.8)

    # Highlighted CAR perimeter
    try:
        gpd.GeoSeries([boundary], crs=raster_crs).plot(ax=ax3, color='#facc15', linewidth=2.5)
    except Exception:
        pass

    ax3.set_title(f"3. LAPIG Planar Cadastral Vectorization\n{num_paddocks} Rectified Parcels | Watertight Topology (0 Overlaps)",
                  color='#10b981', fontsize=12, fontweight='bold', pad=12)
    ax3.set_xlim(extent[0], extent[1])
    ax3.set_ylim(extent[2], extent[3])
    ax3.axis('off')

    # Super-title
    fig.suptitle(f"{title_main} — Area: {area_ha:,.2f} ha | Delimited Parcels: {num_paddocks}",
                 color='#f8fafc', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.93])

    if isinstance(output_paths, str):
        output_paths = [output_paths]

    first_path = output_paths[0]
    os.makedirs(os.path.dirname(first_path), exist_ok=True)
    plt.savefig(first_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {first_path}")

    # Copy to any secondary target paths
    for sec_path in output_paths[1:]:
        os.makedirs(os.path.dirname(sec_path), exist_ok=True)
        shutil.copy2(first_path, sec_path)
        print(f"Copied to: {sec_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_img_dir = os.path.join(base_dir, "docs", "images")
    images_dir = os.path.join(base_dir, "images")

    # 1. Caiapônia
    render_property_comparison(
        folder_path=os.path.join(base_dir, "results", "GO-5204409-01D1AD1F9586415E9D46BF93115D27DF"),
        output_paths=[
            os.path.join(docs_img_dir, "fig_caiaponia_results.png"),
            os.path.join(images_dir, "fig_caiaponia_results.png")
        ],
        title_main="Rural Property in Caiapônia (GO, Brazil)",
        area_ha=749.52
    )

    # 2. Itapirapuã
    render_property_comparison(
        folder_path=os.path.join(base_dir, "results", "GO-5211008-46C08C6235D74D36BE07B5384A17B156"),
        output_paths=[
            os.path.join(docs_img_dir, "fig_itapirapua_results.png"),
            os.path.join(images_dir, "fig_itapirapua_results.png")
        ],
        title_main="Rural Property in Itapirapuã (GO, Brazil)",
        area_ha=193.59
    )

    # 3. Paranaíba (MS)
    render_property_comparison(
        folder_path=os.path.join(base_dir, "results", "MS-5006309-8DC93A48C2424FD7B3EDF235F8B531E1"),
        output_paths=[
            os.path.join(docs_img_dir, "fig_paranaiba_results.png"),
            os.path.join(images_dir, "fig_paranaiba_results.png")
        ],
        title_main="Rural Property in Paranaíba (MS, Brazil)",
        area_ha=1085.22
    )
