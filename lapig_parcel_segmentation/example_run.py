#!/usr/bin/env python3
"""
Example Execution Script for the LAPIG Cadastral Parcel Segmentation Pipeline
Supports execution with local vector files (.shp, .fgb, .gpkg, .parquet), Earth Engine assets, or coordinates.

Usage Examples with the Goiás CAR GeoPackage:
  # 1. List all rural property codes (cod_imovel) available in the vector file:
  python example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --list-ids

  # 2. Run the pipeline for a specific property:
  python example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --id-val "GO-5203500-366A293867A24EF5A446DA91E23F8548" --year 2025

  # 3. Run in batch mode for ALL properties in the file:
  python example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --all

  # 4. Run a batch test on the first 2 properties:
  python example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --limit 2
"""

import os
import sys
import argparse
import ee
import geopandas as gpd

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from lapig_parcel_segmentation import run_analysis_lapig, run_batch_analysis_lapig
except ImportError:
    from pipeline_parcel_segmentation import run_analysis_lapig, run_batch_analysis_lapig

def main():
    parser = argparse.ArgumentParser(description="Cadastral Parcel Segmentation - LAPIG Pipeline")
    parser.add_argument("--input", type=str, default=None, help="Local vector file (.shp, .gpkg, .fgb, .parquet), EE asset, or 'lat, lon' coordinates")
    parser.add_argument("--coord", type=str, default=None, help="Geographic coordinate 'lat, lon' for automatic CAR query")
    parser.add_argument("--id-col", type=str, default=None, dest="id_col", help="Property ID column name in the vector file (e.g., cod_imovel, id)")
    parser.add_argument("--id-val", type=str, default=None, dest="id_val", help="Specific property ID value to process individually")
    parser.add_argument("--all", action="store_true", help="Execute the batch pipeline for ALL properties in the vector file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of properties to process in batch mode (e.g., --limit 3)")
    parser.add_argument("--list-ids", action="store_true", dest="list_ids", help="List all property IDs available in the vector file and exit")
    parser.add_argument("--farm", type=str, default=None, choices=["GO", "MS"], help="Quick test preset: GO or MS")
    parser.add_argument("--filter", nargs=2, metavar=('FIELD', 'VALUE'), default=None, help="Optional filter for EE asset (e.g., --filter cod_imovel GO-...)")
    parser.add_argument("--year", type=int, default=2025, help="Analysis year (Sentinel-2 & MapBiomas, default: 2025)")
    parser.add_argument("--collection", type=str, default="10m", choices=["10m", "30m"], help="MapBiomas collection (10m Col 4 or 30m Col 11)")
    parser.add_argument("--output-dir", type=str, default="results", dest="output_dir", help="Base directory for output folders")
    parser.add_argument("--min_area", type=float, default=2.0, help="Minimum parcel area threshold in hectares")
    parser.add_argument("--straight-tol", type=float, default=22.0, dest="straight_tol", help="Straight fence line simplification/rectification tolerance (meters)")
    parser.add_argument("--no-pasture-filter", action="store_true", dest="no_pasture", help="Disable MapBiomas pasture filter and produce raw segmentation only")
    parser.add_argument("--project", type=str, default=None, help="Google Earth Engine project ID (optional, uses active session by default)")

    args = parser.parse_args()

    # Feature: List IDs in the vector file if requested
    if args.list_ids:
        target_file = args.input or "car_goias_area_imovel_epsg4674_hottest.gpkg"
        if not os.path.isabs(target_file) and not os.path.exists(target_file):
            for base in [os.getcwd(), parent_dir, script_dir]:
                c = os.path.join(base, target_file)
                if os.path.exists(c):
                    target_file = c
                    break

        if not os.path.exists(target_file):
            print(f"[-] File not found: {target_file}")
            sys.exit(1)

        id_col = args.id_col or "cod_imovel"
        ext = os.path.splitext(target_file)[1].lower()
        gdf = gpd.read_parquet(target_file) if ext in ['.parquet', '.geoparquet'] else gpd.read_file(target_file)

        if id_col not in gdf.columns:
            print(f"[-] Column '{id_col}' not found in file.")
            print(f"    Available columns: {list(gdf.columns)}")
            sys.exit(1)

        print("=" * 80)
        print(f"PROPERTIES AVAILABLE IN: {os.path.basename(target_file)} ({len(gdf)} records)")
        print("=" * 80)
        has_area = 'num_area' in gdf.columns
        has_mun = 'municipio' in gdf.columns

        for idx, row in gdf.iterrows():
            val = row[id_col]
            extra = []
            if has_mun:
                extra.append(f"Municipality: {row['municipio']}")
            if has_area:
                extra.append(f"Area: {row['num_area']:.1f} ha")
            extra_str = f" ({', '.join(extra)})" if extra else ""
            print(f"  [{idx+1:02d}] {val}{extra_str}")
        print("=" * 80)
        print("Tip: To process a specific property, run:")
        sample_id = gdf[id_col].iloc[0]
        print(f'   python example_run.py --input "{os.path.basename(target_file)}" --id-col "{id_col}" --id-val "{sample_id}"')
        sys.exit(0)

    # Determine input target
    if args.coord:
        target = args.coord
    elif args.input:
        target = args.input
        if not os.path.isabs(target) and not os.path.exists(target):
            for base in [os.getcwd(), parent_dir, script_dir]:
                c = os.path.join(base, target)
                if os.path.exists(c):
                    target = c
                    break
    elif args.farm == "GO":
        target = (-15.74728, -50.89405)
    elif args.farm == "MS":
        target = (-21.5794, -54.7645)
    else:
        target = (-15.74728, -50.89405)

    output_dir = os.path.join(parent_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    # Batch processing branch
    if args.all or (args.limit is not None and not args.id_val):
        print("=" * 80)
        print("EXECUTING BATCH PROCESSING PIPELINE - LAPIG")
        print(f"   Input Target:     {target}")
        print(f"   ID Column:        {args.id_col or 'cod_imovel'}")
        if args.limit:
            print(f"   Property Limit:   {args.limit}")
        print("=" * 80)

        success_gdfs, consolidated_shp = run_batch_analysis_lapig(
            input_target=target,
            id_column=args.id_col or "cod_imovel",
            max_imoveis=args.limit,
            year=args.year,
            start_month=3,
            end_month=10,
            buffer_size=500,
            output_base_dir=output_dir,
            model_type="auto",
            min_area_ha=args.min_area,
            straight_tol=args.straight_tol,
            min_pasture_coverage=0.35,
            mapbiomas_collection=args.collection,
            skip_sam_if_exists=True,
            filter_by_pasture=(not args.no_pasture),
            consolidate_shp=True,
            enable_benchmark=True,
            ee_project=args.project
        )

        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"Total properties processed: {len(success_gdfs)}")
        if consolidated_shp:
            print(f"Consolidated Pasture Shapefile: {consolidated_shp}")
        print(f"Individual result directories in: {output_dir}/")
        print("=" * 80)
        return

    # Individual property execution branch
    print("=" * 70)
    print("EXECUTING CADASTRAL PARCEL SEGMENTATION PIPELINE (LAPIG)")
    print(f"   Input Target:     {target}")
    if args.id_col:
        print(f"   ID Column:        {args.id_col}")
    if args.id_val:
        print(f"   ID Filter:        {args.id_val}")
    if args.no_pasture:
        print(f"   Pasture Filter:   DISABLED (Pure Segmentation Mode)")
    print("=" * 70)

    filter_tuple = tuple(args.filter) if args.filter else None

    gdf_pasto, shp_pasto, gdf_pure, shp_pure = run_analysis_lapig(
        input_target=target,
        id_column=args.id_col,
        id_value=args.id_val,
        year=args.year,
        start_month=3,
        end_month=10,
        buffer_size=500,
        folder_name=None, # Creates a dedicated subfolder with the property ID
        output_base_dir=output_dir,
        model_type="auto",
        min_area_ha=args.min_area,
        straight_tol=args.straight_tol,
        min_pasture_coverage=0.35,
        mapbiomas_collection=args.collection,
        skip_sam_if_exists=True,
        filter_by_pasture=(not args.no_pasture),
        return_pure=True,
        enable_benchmark=True,
        car_property_filter=filter_tuple,
        ee_project=args.project
    )

    print("\nIndividual processing completed successfully!")
    print(f"Pure Segmentation Shapefile (all parcels):    {shp_pure} ({len(gdf_pure)} parcels)")
    if not args.no_pasture:
        print(f"Final Filtered Pasture Shapefile:            {shp_pasto} ({len(gdf_pasto)} parcels)")

if __name__ == '__main__':
    main()
