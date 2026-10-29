#!/usr/bin/env python3
"""
Modular Cadastral Parcel Segmentation and Vectorization Pipeline (LAPIG)
Complete modularized replacement for the legacy notebook parcelSegmentation_SAMGeo_LAPIG.ipynb

Supports single-property execution and batch processing for all IDs in a vector file.
"""

import os
import sys
import re
import time
from datetime import timedelta
import ee
import geemap
import geopandas as gpd
import pandas as pd

# Support relative import inside package or direct import as standalone script
try:
    from .roi_resolver import resolve_input_roi, sanitize_id
    from .s2_mosaic import get_sentinel2_ndvi_mosaic
    from .sam_segmenter import run_samgeo_segmentation, get_gpu_info, clear_gpu_memory
    from .cadastral_vectorizer import (
        vectorize_and_straighten_parcels,
        clip_parcels_by_car,
        generate_cadastral_stats
    )
    from .pasture_classifier import (
        get_mapbiomas_pasture_ee,
        export_pasture_mask_local,
        classify_and_filter_pasture_parcels
    )
    from .benchmark_monitor import HardwareBenchmarkMonitor
except ImportError:
    from roi_resolver import resolve_input_roi, sanitize_id
    from s2_mosaic import get_sentinel2_ndvi_mosaic
    from sam_segmenter import run_samgeo_segmentation, get_gpu_info, clear_gpu_memory
    from cadastral_vectorizer import (
        vectorize_and_straighten_parcels,
        clip_parcels_by_car,
        generate_cadastral_stats
    )
    from pasture_classifier import (
        get_mapbiomas_pasture_ee,
        export_pasture_mask_local,
        classify_and_filter_pasture_parcels
    )
    try:
        from benchmark_monitor import HardwareBenchmarkMonitor
    except ImportError:
        HardwareBenchmarkMonitor = None

def run_analysis_lapig(
    roi=None,
    input_target=None,
    id_column=None,             # Property ID column name in the vector file (e.g., 'cod_imovel', 'id')
    id_value=None,              # Specific ID value or 'all' to run all properties
    year=2025,
    start_month=3,
    end_month=10,
    buffer_size=500,
    folder_name=None,           # If None or 'auto': creates predefined folder results/{ID_IMOVEL}
    output_base_dir="results",  # Base directory when folder_name is automatic
    model_type="auto",
    min_area_ha=2.0,
    straight_tol=22.0,
    min_pasture_coverage=0.35,
    mapbiomas_collection='10m',
    skip_sam_if_exists=True,
    filter_by_pasture=True,
    return_pure=False,
    enable_benchmark=True,
    car_property_filter=None,
    car_assets=None,
    coord_buffer_meters=1500,
    ee_project=None
):
    """
    Executes the segmentation pipeline for a single property or delegates to batch if id_value='all'.

    Key Features:
    1. Pure Physical Segmentation: Generates 'parcelas_segmentacao_pura_{ID}.shp' containing all
       physical agricultural parcels within the CAR boundary prior to MapBiomas filtering.
    2. Hardware Benchmarking: Continuously tracks peak/average VRAM, system RAM, CPU load,
       and GPU compute load across all pipeline phases, generating a full technical report.
    """
    # If id_value is 'all', delegate to batch processing
    if str(id_value).lower() in ['all', '*', 'todos']:
        return run_batch_analysis_lapig(
            input_target=input_target or roi,
            id_column=id_column or "cod_imovel",
            year=year,
            start_month=start_month,
            end_month=end_month,
            buffer_size=buffer_size,
            output_base_dir=output_base_dir,
            model_type=model_type,
            min_area_ha=min_area_ha,
            straight_tol=straight_tol,
            min_pasture_coverage=min_pasture_coverage,
            mapbiomas_collection=mapbiomas_collection,
            skip_sam_if_exists=skip_sam_if_exists,
            filter_by_pasture=filter_by_pasture,
            enable_benchmark=enable_benchmark,
            ee_project=ee_project
        )

    # Ensure clean VRAM before starting parcel processing
    clear_gpu_memory()

    # Initialize Hardware Benchmark Monitor
    monitor = HardwareBenchmarkMonitor(sample_interval=1.0) if (enable_benchmark and HardwareBenchmarkMonitor) else None
    if monitor:
        monitor.start()
        monitor.set_phase("1. Area of Interest (ROI) Resolution")

    t_start = time.time()

    # 0. Universal ROI Resolution
    target = input_target if input_target is not None else roi
    if target is None:
        raise ValueError("[-] No input_target or roi provided for analysis.")

    roi_ee, gdf_car, roi_info = resolve_input_roi(
        input_target=target,
        id_column=id_column,
        id_value=id_value,
        car_property_filter=car_property_filter,
        car_assets=car_assets,
        buffer_radius_meters=coord_buffer_meters,
        ee_project=ee_project
    )

    # 1. Output Folder Setup based on Property ID
    prop_id = roi_info.get("id_imovel") or roi_info.get("cod_imovel") or "segmented_property"
    clean_id = sanitize_id(prop_id)

    if folder_name is None or str(folder_name).lower() in ["none", "auto", "default"]:
        folder_name = os.path.join(output_base_dir, clean_id)
    elif folder_name == "results":
        folder_name = os.path.join("results", clean_id)

    os.makedirs(folder_name, exist_ok=True)

    print("\n" + "=" * 70)
    print("LAPIG CADASTRAL PARCEL SEGMENTATION PIPELINE")
    print(f"   Property ID:      {clean_id}")
    print(f"   Output Directory: {folder_name}")
    print(f"   Target Year:      {year} (Dry season window: months {start_month:02d} to {end_month:02d})")
    print(f"   MapBiomas:        Collection {mapbiomas_collection} (coverage up to 2025)")
    print("=" * 70)

    # 2. Multi-temporal Sentinel-2 Mosaic Download (NDVI max, median, min)
    if monitor:
        monitor.set_phase("2. Sentinel-2 Mosaic Download (GEE)")

    s2_tif_path = os.path.join(folder_name, "s2_ndvi_rgb.tif")
    image_ndvis, s2_path, roi_buffered = get_sentinel2_ndvi_mosaic(
        roi_ee=roi_ee,
        year=year,
        start_month=start_month,
        end_month=end_month,
        buffer_meters=buffer_size,
        output_path=s2_tif_path,
        ee_project=ee_project
    )

    # 3. SAMGeo GPU Segmentation
    if monitor:
        monitor.set_phase("3. SAMGeo Inference (ViT-H GPU)")

    masks_tif_path = os.path.join(folder_name, "masks.tif")
    run_samgeo_segmentation(
        image_path=s2_path,
        output_mask_path=masks_tif_path,
        model_type=model_type,
        checkpoint_dir="models",
        force_rerun=(not skip_sam_if_exists)
    )

    # 4. Cadastral Vectorization and Straight Fence Regularization
    if monitor:
        monitor.set_phase("4. Vectorization and Cadastral Regularization (CPU)")

    gdf_parcels = vectorize_and_straighten_parcels(
        sam_mask_tif=masks_tif_path,
        min_area_ha=min_area_ha,
        straight_tol=straight_tol
    )

    # 5. Pure Physical Segmentation Clipped to CAR Boundary
    if monitor:
        monitor.set_phase("5. Pure Segmentation Clipped to CAR Boundary")

    gdf_pure_car = clip_parcels_by_car(
        gdf_parcels=gdf_parcels,
        car_geom_or_gdf=gdf_car,
        min_area_ha=min_area_ha * 0.5
    )
    gdf_pure_car['cod_car'] = clean_id
    pure_shp_path = os.path.join(folder_name, f"parcelas_segmentacao_pura_{clean_id}.shp")

    # 6. Zonal Classification and Thematic Pasture Filtering (MapBiomas)
    if monitor:
        monitor.set_phase("6. MapBiomas Zonal Classification (GEE/CPU)")

    if filter_by_pasture:
        print(f"\nProcessing pasture zonal classification with MapBiomas ({mapbiomas_collection})...")
        try:
            scale_val = 30 if str(mapbiomas_collection).lower() in ['30', '30m', 'col11'] else 10
            pasture_ee = get_mapbiomas_pasture_ee(roi_ee, year=min(year, 2025), collection=mapbiomas_collection, ee_project=ee_project)
            mb_mask_tif = os.path.join(folder_name, f"temp_mb_mask_{mapbiomas_collection}.tif")
            export_pasture_mask_local(pasture_ee, roi_ee, s2_path, mb_mask_tif, scale=scale_val)

            # Classify pure CAR segmentation directly
            gdf_classified, gdf_others = classify_and_filter_pasture_parcels(
                gdf_parcels=gdf_pure_car,
                pasture_mask_tif=mb_mask_tif,
                min_pasture_coverage=min_pasture_coverage,
                s2_ndvi_tif=s2_path
            )

            # Combine pasture and other parcels to form the fully annotated physical layer
            gdf_all_annotated = pd.concat([gdf_classified, gdf_others], ignore_index=True)
            gdf_all_annotated = gpd.GeoDataFrame(gdf_all_annotated, crs=gdf_classified.crs)

            # Layer 1: Pure Physical Segmentation (with annotated MapBiomas attributes)
            gdf_pure_clean = gdf_all_annotated.rename(columns={
                'pasture_cov': 'pasto_cov',
                'ndvi_median': 'ndvi_med',
                'is_pasture': 'is_pasto',
                'class_name': 'classe'
            })
            gdf_pure_clean['is_pasto'] = gdf_pure_clean['is_pasto'].astype(int)
            if 'id_parcela' not in gdf_pure_clean.columns:
                gdf_pure_clean['id_parcela'] = range(1, len(gdf_pure_clean) + 1)

            gdf_pure_clean_4326 = gdf_pure_clean.to_crs(epsg=4326)
            gdf_pure_clean_4326.to_file(pure_shp_path)
            print(f"Pure Segmentation Shapefile Saved (All {len(gdf_pure_clean_4326)} parcels): {pure_shp_path}")

            # Layer 2: Validated Pasture Parcels Only
            gdf_pasture = gdf_pure_clean_4326[gdf_pure_clean_4326['is_pasto'] == 1].copy().reset_index(drop=True)
            gdf_pasture['id_parcela'] = range(1, len(gdf_pasture) + 1)
            final_shp_path = os.path.join(folder_name, f"parcelas_finais_cercas_retas_{clean_id}.shp")
            gdf_pasture.to_file(final_shp_path)
            print(f"Final Filtered Pasture Shapefile Saved ({len(gdf_pasture)} parcels): {final_shp_path}")

            # Layer 3: Non-Pasture Parcels (Audit layer: native vegetation, agriculture, infrastructure)
            gdf_others_clean = gdf_pure_clean_4326[gdf_pure_clean_4326['is_pasto'] == 0].copy().reset_index(drop=True)
            if len(gdf_others_clean) > 0:
                others_shp = os.path.join(folder_name, "nao_pastagem_auditoria.shp")
                gdf_others_clean.to_file(others_shp)
                print(f"Non-pasture audit shapefile saved: {others_shp}")

            gdf_final_clean_4326 = gdf_pasture

        except Exception as e:
            print(f"Warning during MapBiomas query: {e}. Retaining pure segmentation.")
            gdf_pure_clean_4326 = gdf_pure_car.to_crs(epsg=4326)
            if 'id_parcela' not in gdf_pure_clean_4326.columns:
                gdf_pure_clean_4326['id_parcela'] = range(1, len(gdf_pure_clean_4326) + 1)
            gdf_pure_clean_4326.to_file(pure_shp_path)
            gdf_final_clean_4326 = gdf_pure_clean_4326
            final_shp_path = pure_shp_path
    else:
        # Without pasture filter: final result is pure physical segmentation
        gdf_pure_clean_4326 = gdf_pure_car.to_crs(epsg=4326)
        if 'id_parcela' not in gdf_pure_clean_4326.columns:
            gdf_pure_clean_4326['id_parcela'] = range(1, len(gdf_pure_clean_4326) + 1)
        gdf_pure_clean_4326.to_file(pure_shp_path)
        gdf_final_clean_4326 = gdf_pure_clean_4326
        final_shp_path = pure_shp_path
        print(f"Pure Segmentation Shapefile Saved ({len(gdf_pure_clean_4326)} parcels): {pure_shp_path}")

    # 7. Cadastral Statistical Summary
    if monitor:
        monitor.set_phase("7. Cadastral Statistics & Shapefile Export")

    generate_cadastral_stats(gdf_final_clean_4326, title=f"FINAL PASTURE PARCEL REPORT ({clean_id})")

    # 8. Hardware Benchmark Report
    bench_summary = None
    if monitor:
        bench_summary = monitor.stop()
        txt_path, json_path = monitor.save_report(folder_name, filename_prefix="relatorio_benchmark_hardware")
        print("\n" + monitor.generate_report_text())
        print(f"Hardware Benchmark Report saved to: {txt_path}")

    total_time = str(timedelta(seconds=int(time.time() - t_start)))
    print(f"Property processing completed in: {total_time}")
    print("=" * 70 + "\n")

    # Metadata attached to returned GeoDataFrame
    gdf_final_clean_4326.attrs['shp_pasture'] = final_shp_path
    gdf_final_clean_4326.attrs['shp_pure'] = pure_shp_path
    gdf_final_clean_4326.attrs['gdf_pure'] = gdf_pure_clean_4326
    if bench_summary:
        gdf_final_clean_4326.attrs['benchmark'] = bench_summary

    # Final VRAM cleanup after completing property
    clear_gpu_memory()

    if return_pure:
        return gdf_final_clean_4326, final_shp_path, gdf_pure_clean_4326, pure_shp_path
    return gdf_final_clean_4326, final_shp_path

def run_batch_analysis_lapig(
    input_target,
    id_column="cod_imovel",
    id_list=None,
    id_value=None,              # Support single ID or list for full compatibility
    max_imoveis=None,
    year=2025,
    start_month=3,
    end_month=10,
    buffer_size=500,
    output_base_dir="results",
    model_type="auto",
    min_area_ha=2.0,
    straight_tol=22.0,
    min_pasture_coverage=0.35,
    mapbiomas_collection='10m',
    skip_sam_if_exists=True,
    filter_by_pasture=True,
    consolidate_shp=True,
    enable_benchmark=True,
    ee_project=None
):
    """
    Executes the batch pipeline for ALL rural properties present in a vector file.
    Creates individual property subfolders and compiles consolidated shapefiles:
    - parcelas_todos_imoveis_consolidados.shp (thematically filtered pasture parcels)
    - parcelas_todos_imoveis_segmentacao_pura.shp (pure physical segmentation of all parcels)
    - relatorio_benchmark_hardware_lote.txt (comprehensive hardware tracking)
    - Enforces continuous VRAM deallocation on every processed property.
    """
    print("\n" + "=" * 80)
    print("STARTING BATCH PROCESSING PIPELINE - LAPIG")
    print(f"   Input Vector File: {os.path.basename(str(input_target))}")
    print(f"   Property ID Column: {id_column}")
    print("=" * 80)

    # Support id_value in batch mode if provided
    if id_value and not id_list:
        id_list = [id_value] if not isinstance(id_value, (list, tuple, set)) else list(id_value)

    # Read ID list
    ext = os.path.splitext(str(input_target))[1].lower()
    if ext in ['.parquet', '.geoparquet']:
        gdf_all = gpd.read_parquet(input_target)
    elif os.path.exists(str(input_target)):
        gdf_all = gpd.read_file(input_target)
    else:
        raise FileNotFoundError(f"Vector file not found: {input_target}")

    if id_column not in gdf_all.columns:
        raise ValueError(f"Column '{id_column}' not found. Available columns: {list(gdf_all.columns)}")

    # Extract unique IDs
    unique_ids = gdf_all[id_column].dropna().unique().tolist()
    if id_list:
        unique_ids = [i for i in unique_ids if str(i) in [str(x) for x in id_list]]
    if max_imoveis:
        unique_ids = unique_ids[:int(max_imoveis)]

    total = len(unique_ids)
    print(f"Total rural properties to process: {total}")

    success_gdfs_pasture = []
    success_gdfs_pure = []
    success_paths = []
    failed_ids = []

    # Batch hardware monitor
    batch_monitor = HardwareBenchmarkMonitor(sample_interval=2.0) if (enable_benchmark and HardwareBenchmarkMonitor) else None
    if batch_monitor:
        batch_monitor.start()

    t_batch_start = time.time()

    for idx, prop_id in enumerate(unique_ids):
        print("\n" + "#" * 80)
        print(f"[{idx+1}/{total}] PROCESSING PROPERTY: {prop_id}")
        print("#" * 80)

        # Deallocate VRAM at the start of each property
        clear_gpu_memory(verbose=True)

        if batch_monitor:
            batch_monitor.set_phase(f"Property [{idx+1}/{total}] {prop_id}")

        try:
            gdf_res, shp_res, gdf_pure_res, shp_pure_res = run_analysis_lapig(
                input_target=input_target,
                id_column=id_column,
                id_value=prop_id,
                year=year,
                start_month=start_month,
                end_month=end_month,
                buffer_size=buffer_size,
                folder_name=None, # Automatically uses subfolder results/{prop_id}
                output_base_dir=output_base_dir,
                model_type=model_type,
                min_area_ha=min_area_ha,
                straight_tol=straight_tol,
                min_pasture_coverage=min_pasture_coverage,
                mapbiomas_collection=mapbiomas_collection,
                skip_sam_if_exists=skip_sam_if_exists,
                filter_by_pasture=filter_by_pasture,
                return_pure=True,
                enable_benchmark=enable_benchmark,
                ee_project=ee_project
            )
            success_gdfs_pasture.append(gdf_res)
            success_gdfs_pure.append(gdf_pure_res)
            success_paths.append(shp_res)
        except Exception as e:
            print(f"[-] Error processing property {prop_id}: {e}")
            failed_ids.append((prop_id, str(e)))
        finally:
            # Force VRAM deallocation at the end of each property
            clear_gpu_memory(verbose=True)

    # Compile Consolidated Shapefiles if requested
    consolidated_pasture_path = None
    consolidated_pure_path = None

    if consolidate_shp:
        # 1. Consolidated Pasture Shapefile
        if success_gdfs_pasture:
            try:
                gdf_consolidated_pasto = pd.concat(success_gdfs_pasture, ignore_index=True)
                gdf_consolidated_pasto = gpd.GeoDataFrame(gdf_consolidated_pasto, geometry='geometry', crs=success_gdfs_pasture[0].crs)
                consolidated_pasture_path = os.path.join(output_base_dir, "parcelas_todos_imoveis_consolidados.shp")
                gdf_consolidated_pasto.to_file(consolidated_pasture_path)
                print(f"\nConsolidated Pasture Shapefile Saved: {consolidated_pasture_path}")
            except Exception as e:
                print(f"Warning: Could not consolidate pasture shapefile: {e}")

        # 2. Consolidated Pure Physical Segmentation Shapefile (All agricultural parcels)
        if success_gdfs_pure:
            try:
                gdf_consolidated_pure = pd.concat(success_gdfs_pure, ignore_index=True)
                gdf_consolidated_pure = gpd.GeoDataFrame(gdf_consolidated_pure, geometry='geometry', crs=success_gdfs_pure[0].crs)
                consolidated_pure_path = os.path.join(output_base_dir, "parcelas_todos_imoveis_segmentacao_pura.shp")
                gdf_consolidated_pure.to_file(consolidated_pure_path)
                print(f"Consolidated Pure Segmentation Shapefile Saved: {consolidated_pure_path}")
            except Exception as e:
                print(f"Warning: Could not consolidate pure segmentation shapefile: {e}")

    # Final Batch & Benchmark Report
    if batch_monitor:
        batch_monitor.stop()
        batch_txt, batch_json = batch_monitor.save_report(output_base_dir, filename_prefix="relatorio_benchmark_hardware_lote")
        print("\n" + batch_monitor.generate_report_text())
        print(f"Batch Hardware Report saved to: {batch_txt}")

    elapsed = str(timedelta(seconds=int(time.time() - t_batch_start)))
    print("\n" + "=" * 80)
    print("BATCH PROCESSING FINAL REPORT")
    print("=" * 80)
    print(f"  Total properties in batch:        {total}")
    print(f"  Successfully processed:           {len(success_gdfs_pasture)}")
    print(f"  Failed / Errors:                  {len(failed_ids)}")
    print(f"  Total processing time:            {elapsed}")
    if consolidated_pasture_path:
        total_pasto_ha = sum([g['area_ha'].sum() for g in success_gdfs_pasture if 'area_ha' in g.columns])
        total_parcelas = sum([len(g) for g in success_gdfs_pasture])
        print(f"  Total validated pasture parcels:  {total_parcelas} ({total_pasto_ha:.2f} ha)")
        print(f"  Consolidated Pasture Shapefile:   {consolidated_pasture_path}")
    if consolidated_pure_path:
        total_puro_ha = sum([g['area_ha'].sum() for g in success_gdfs_pure if 'area_ha' in g.columns])
        total_talhoes = sum([len(g) for g in success_gdfs_pure])
        print(f"  Total physical parcels (pure):    {total_talhoes} ({total_puro_ha:.2f} ha)")
        print(f"  Consolidated Pure Shapefile:      {consolidated_pure_path}")
    print("=" * 80 + "\n")

    return success_gdfs_pasture, consolidated_pasture_path

if __name__ == '__main__':
    print("Cadastral parcel segmentation pipeline ready for execution.")
