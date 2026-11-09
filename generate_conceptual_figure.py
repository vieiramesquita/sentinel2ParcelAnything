#!/usr/bin/env python3
"""
Generates high-resolution publication-quality conceptual infographic:
'From Meta SAM to SAMGeo and LAPIG ParcelAnything'
Saved directly to images/fig_conceptual_framework.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def create_conceptual_framework_figure(output_path="images/fig_conceptual_framework.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set up dark theme aesthetic matching LAPIG / Antigravity styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#334155'
    plt.rcParams['axes.linewidth'] = 0.8

    fig = plt.figure(figsize=(18, 10.5), facecolor='#0b0f19')

    # Main Grid: 2 rows, 2 columns
    # Top Left: 1. Meta AI SAM (The Vision Foundation Model)
    # Top Right: 2. SAMGeo (Geospatial Extension)
    # Bottom Left: 3. The Unresolved Challenge (Raw AI Limitations in Rural Cadastre)
    # Bottom Right: 4. LAPIG ParcelAnything (The Complete Solution)

    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18, left=0.05, right=0.95, top=0.91, bottom=0.06)

    # Supertitle
    fig.suptitle("THE EVOLUTIONARY PATHWAY: FROM FOUNDATION AI TO CADASTRAL INTELLIGENCE",
                 fontsize=18, fontweight='bold', color='#f8fafc', y=0.97)
    fig.text(0.5, 0.94, "How LAPIG ParcelAnything transforms Meta SAM and SAMGeo into Authoritative, Watertight Agricultural Cadastre",
             fontsize=12, color='#94a3b8', ha='center')

    # -------------------------------------------------------------
    # PANEL 1: Meta AI SAM (Segment Anything Model)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#111827')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    # Header Box
    ax1.add_patch(patches.FancyBboxPatch((0.2, 8.4), 9.6, 1.4, boxstyle="round,pad=0.2",
                                         facecolor='#1e293b', edgecolor='#3b82f6', linewidth=2))
    ax1.text(0.5, 9.2, "1. Meta AI: Segment Anything Model (SAM)", fontsize=13, fontweight='bold', color='#60a5fa')
    ax1.text(0.5, 8.6, "Kirillov et al., 2023 • Foundation Model for Computer Vision (SA-1B Dataset)", fontsize=10, color='#94a3b8')

    # Architectural diagram inside Panel 1
    # Image -> Heavy ViT-H Encoder -> Image Embedding -> Mask Decoder <- Prompt Tokens
    ax1.add_patch(patches.FancyBboxPatch((0.5, 5.0), 2.2, 2.6, boxstyle="round,pad=0.1", facecolor='#1e3a8a', edgecolor='#60a5fa', linewidth=1.5))
    ax1.text(1.6, 6.6, "Standard\nRGB Image", color='#ffffff', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(1.6, 5.5, "(8-bit, 3 bands\nNatural Photos)", color='#93c5fd', ha='center', va='center', fontsize=7.5)

    ax1.annotate("", xy=(3.3, 6.3), xytext=(2.7, 6.3), arrowprops=dict(arrowstyle="->", color='#38bdf8', lw=2))

    ax1.add_patch(patches.FancyBboxPatch((3.4, 4.6), 2.8, 3.4, boxstyle="round,pad=0.1", facecolor='#0f172a', edgecolor='#38bdf8', linewidth=2))
    ax1.text(4.8, 7.2, "ViT-H Encoder", color='#38bdf8', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(4.8, 6.3, "636M Parameters\n14×14 Window Attn\n16× Downscaled", color='#cbd5e1', ha='center', va='center', fontsize=8)
    ax1.text(4.8, 5.1, "1× Image Embedding", color='#facc15', ha='center', va='center', fontsize=8, fontweight='bold')

    ax1.annotate("", xy=(6.8, 6.3), xytext=(6.2, 6.3), arrowprops=dict(arrowstyle="->", color='#38bdf8', lw=2))

    ax1.add_patch(patches.FancyBboxPatch((6.9, 4.6), 2.7, 3.4, boxstyle="round,pad=0.1", facecolor='#1e293b', edgecolor='#4ade80', linewidth=2))
    ax1.text(8.25, 7.2, "Fast Mask Decoder", color='#4ade80', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(8.25, 6.2, "Lightweight MLP\nAmbiguity Aware\nLatency ~50ms", color='#cbd5e1', ha='center', va='center', fontsize=8)
    ax1.text(8.25, 5.1, "Raw Binary Mask", color='#f43f5e', ha='center', va='center', fontsize=8, fontweight='bold')

    # Key strengths & boundaries
    bullets_p1 = (
        "✓ Trained on >1 Billion masks across 11 Million images (SA-1B).\n"
        "✓ Promptable zero-shot general segmentation (points, boxes, masks).\n"
        "✗ Limitation: Built for non-geospatial RGB pixels; no CRS, no geospatial coordinates."
    )
    ax1.text(0.5, 2.2, bullets_p1, color='#e2e8f0', fontsize=8.5, linespacing=1.6)

    # -------------------------------------------------------------
    # PANEL 2: SAMGeo (Segment Geospatial)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#111827')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Header Box
    ax2.add_patch(patches.FancyBboxPatch((0.2, 8.4), 9.6, 1.4, boxstyle="round,pad=0.2",
                                         facecolor='#1e293b', edgecolor='#10b981', linewidth=2))
    ax2.text(0.5, 9.2, "2. SAMGeo: Segment Anything for Geospatial Data", fontsize=13, fontweight='bold', color='#34d399')
    ax2.text(0.5, 8.6, "Wu, 2023 • Bridging Computer Vision & GIS (segment-geospatial Python Package)", fontsize=10, color='#94a3b8')

    # Architectural diagram inside Panel 2
    ax2.add_patch(patches.FancyBboxPatch((0.5, 4.8), 2.6, 3.2, boxstyle="round,pad=0.1", facecolor='#064e3b', edgecolor='#10b981', linewidth=1.5))
    ax2.text(1.8, 7.1, "Geospatial Rasters", color='#ffffff', ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax2.text(1.8, 5.8, "• GeoTIFF I/O\n• Projected CRS\n• Affine GeoTransform\n• Spatial Extents", color='#a7f3d0', ha='center', va='center', fontsize=8)

    ax2.annotate("", xy=(3.8, 6.4), xytext=(3.1, 6.4), arrowprops=dict(arrowstyle="->", color='#34d399', lw=2))

    ax2.add_patch(patches.FancyBboxPatch((3.9, 4.8), 2.7, 3.2, boxstyle="round,pad=0.1", facecolor='#0f172a', edgecolor='#10b981', linewidth=2))
    ax2.text(5.25, 7.1, "Tiled Inference", color='#34d399', ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax2.text(5.25, 5.8, "• Automatic Grid Cut\n• Tile Size & Overlaps\n• ViT-H Inference\n• IoU & Stability Filter", color='#cbd5e1', ha='center', va='center', fontsize=8)

    ax2.annotate("", xy=(7.3, 6.4), xytext=(6.6, 6.4), arrowprops=dict(arrowstyle="->", color='#34d399', lw=2))

    ax2.add_patch(patches.FancyBboxPatch((7.4, 4.8), 2.2, 3.2, boxstyle="round,pad=0.1", facecolor='#1e293b', edgecolor='#f59e0b', linewidth=2))
    ax2.text(8.5, 7.1, "GIS Vectors", color='#f59e0b', ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax2.text(8.5, 5.8, "• Raster to Shapefile\n• GeoPandas output\n• Raw GeoPackage", color='#fde68a', ha='center', va='center', fontsize=8)

    bullets_p2 = (
        "✓ Enables SAM to directly consume georeferenced satellite rasters (GeoTIFF).\n"
        "✓ Solves GPU VRAM limits on large rasters via tiled window segmentation.\n"
        "✗ Limitation: Outputs raw pixel-level raster masks vectorized without cadastral topology."
    )
    ax2.text(0.5, 2.2, bullets_p2, color='#e2e8f0', fontsize=8.5, linespacing=1.6)

    # -------------------------------------------------------------
    # PANEL 3: The Gap: What Happens with Naive Raw SAM / SAMGeo
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#111827')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')

    # Header Box
    ax3.add_patch(patches.FancyBboxPatch((0.2, 8.4), 9.6, 1.4, boxstyle="round,pad=0.2",
                                         facecolor='#1e293b', edgecolor='#ef4444', linewidth=2))
    ax3.text(0.5, 9.2, "3. The Cadastral Bottleneck: Why Raw AI Fails in the Field", fontsize=13, fontweight='bold', color='#f87171')
    ax3.text(0.5, 8.6, "Physical Rural Land Parcels require Watertight Topology & Straight Post-to-Post Fences", fontsize=10, color='#94a3b8')

    # Visual drawings of the 4 flaws
    # Flaw 1: Pixel Stairstepping (10m pixel artifacts)
    step_x = [0.6, 1.0, 1.0, 1.4, 1.4, 1.8, 1.8, 2.2]
    step_y = [5.0, 5.0, 5.4, 5.4, 5.8, 5.8, 6.2, 6.2]
    ax3.step(step_x, step_y, where='post', color='#f43f5e', linewidth=3)
    ax3.text(1.4, 4.3, "10m Pixel\nStairstepping", color='#f43f5e', ha='center', fontsize=8, fontweight='bold')

    # Flaw 2: Overlaps & Gaps
    poly1 = patches.Circle((3.8, 5.6), 0.9, facecolor='#ef4444', alpha=0.4, edgecolor='#ef4444', linewidth=2)
    poly2 = patches.Circle((4.7, 5.6), 0.9, facecolor='#3b82f6', alpha=0.4, edgecolor='#3b82f6', linewidth=2)
    ax3.add_patch(poly1)
    ax3.add_patch(poly2)
    ax3.text(4.25, 4.3, "Topological Overlaps\n& Sliver Gaps", color='#f87171', ha='center', fontsize=8, fontweight='bold')

    # Flaw 3: Wavy/Wobbly Fences (Noise)
    wave_x = np.linspace(6.0, 7.8, 50)
    wave_y = 5.6 + 0.35 * np.sin(wave_x * 9) + 0.15 * np.cos(wave_x * 15)
    ax3.plot(wave_x, wave_y, color='#fbbf24', linewidth=2.5)
    ax3.text(6.9, 4.3, "Wavy / Erratic\nFence Lines", color='#fbbf24', ha='center', fontsize=8, fontweight='bold')

    # Flaw 4: Blindness to Land Use
    ax3.add_patch(patches.Rectangle((8.3, 4.8), 1.2, 1.6, facecolor='#475569', edgecolor='#94a3b8', linewidth=1.5, linestyle='--'))
    ax3.text(8.9, 5.6, "?", color='#ffffff', fontsize=18, fontweight='bold', ha='center', va='center')
    ax3.text(8.9, 4.3, "No Land Use\n(Pasture vs Crop vs Riparian)", color='#94a3b8', ha='center', fontsize=7.5, fontweight='bold')

    bullets_p3 = (
        "• Naive polygonization produces hundreds of vertices per fence (pixel staircase).\n"
        "• Overlapping polygons break GIS topology (double-counted surface area).\n"
        "• Real fences are straight post-to-post; raw DL creates wavy organic borders.\n"
        "• Segmenter cannot certify whether a parcel is pasture, crops, or native forest."
    )
    ax3.text(0.5, 1.2, bullets_p3, color='#e2e8f0', fontsize=8.2, linespacing=1.5)

    # -------------------------------------------------------------
    # PANEL 4: What WE Are Doing (LAPIG ParcelAnything)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#111827')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')

    # Header Box
    ax4.add_patch(patches.FancyBboxPatch((0.2, 8.4), 9.6, 1.4, boxstyle="round,pad=0.2",
                                         facecolor='#1e293b', edgecolor='#38bdf8', linewidth=2))
    ax4.text(0.5, 9.2, "4. LAPIG ParcelAnything: Cadastral AI Engineering", fontsize=13, fontweight='bold', color='#38bdf8')
    ax4.text(0.5, 8.6, "Planar Graphs • Watertight Rectification • Phenology CS+ • MapBiomas Zonal Audit", fontsize=10, color='#94a3b8')

    # Visual drawings of the 4 solutions
    # Solution 1: Straight fence lines
    ax4.plot([0.6, 2.2], [5.0, 6.2], color='#38bdf8', linewidth=3)
    ax4.plot(0.6, 5.0, marker='o', markersize=6, color='#facc15')
    ax4.plot(2.2, 6.2, marker='o', markersize=6, color='#facc15')
    ax4.text(1.4, 4.3, "Straight Fences\nPost-to-Post", color='#38bdf8', ha='center', fontsize=8, fontweight='bold')

    # Solution 2: Watertight Planar Graph (0 overlap, shared boundary)
    p_a = patches.Polygon([[3.4, 4.9], [4.4, 4.9], [4.4, 6.3], [3.4, 6.3]], closed=True, facecolor='#0284c7', alpha=0.5, edgecolor='#38bdf8', lw=2)
    p_b = patches.Polygon([[4.4, 4.9], [5.4, 4.9], [5.4, 6.3], [4.4, 6.3]], closed=True, facecolor='#10b981', alpha=0.5, edgecolor='#34d399', lw=2)
    ax4.add_patch(p_a)
    ax4.add_patch(p_b)
    ax4.text(4.4, 4.3, "Watertight Topology\n0 Overlaps • 0 Gaps", color='#34d399', ha='center', fontsize=8, fontweight='bold')

    # Solution 3: Multitemporal Phenology (CS+)
    ax4.add_patch(patches.FancyBboxPatch((6.0, 4.9), 1.6, 1.4, boxstyle="round,pad=0.1", facecolor='#065f46', edgecolor='#10b981', lw=1.5))
    ax4.text(6.8, 5.8, "S2 Composite", color='#ffffff', ha='center', fontsize=8, fontweight='bold')
    ax4.text(6.8, 5.2, "NDVI Max/Med/Min", color='#6ee7b7', ha='center', fontsize=7)
    ax4.text(6.8, 4.3, "Phenological Dynamics\n(Cloud Score+)", color='#10b981', ha='center', fontsize=8, fontweight='bold')

    # Solution 4: MapBiomas Zonal Audit
    ax4.add_patch(patches.FancyBboxPatch((8.1, 4.9), 1.6, 1.4, boxstyle="round,pad=0.1", facecolor='#78350f', edgecolor='#f59e0b', lw=1.5))
    ax4.text(8.9, 5.8, "MapBiomas", color='#ffffff', ha='center', fontsize=8, fontweight='bold')
    ax4.text(8.9, 5.2, "10m Col4 / 30m Col11", color='#fde68a', ha='center', fontsize=7)
    ax4.text(8.9, 4.3, "Zonal Certification\n(Pure vs Pasture)", color='#f59e0b', ha='center', fontsize=8, fontweight='bold')

    bullets_p4 = (
        "★ Planar Graph Engine: Merges shared edges & applies chordal regularization.\n"
        "★ Automated Batch Mode & Interactive Human-in-the-Loop with bisector cuts.\n"
        "★ Dual-Layer Output: Pure Cadastral Layout + Certified Productive Pasture.\n"
        "★ Proven Hardware Governance: Zero VRAM leaks over 50,000 ha on RTX 2080 Ti."
    )
    ax4.text(0.5, 1.2, bullets_p4, color='#e2e8f0', fontsize=8.2, linespacing=1.5)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Successfully generated conceptual framework figure: {output_path}")

if __name__ == '__main__':
    create_conceptual_framework_figure()
