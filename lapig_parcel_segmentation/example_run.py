#!/usr/bin/env python3
"""
Script de Exemplo de Execução do Pipeline Cadastral de Parcelas (LAPIG)
Suporta execução com arquivo local (shp, fgb, gpkg, parquet), asset EE ou coordenadas.

Exemplos de Uso com o arquivo GeoPackage do CAR de Goiás:
  # 1. Listar todos os códigos de imóveis (cod_imovel) disponíveis no arquivo:
  python example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --list-ids

  # 2. Rodar o pipeline para um imóvel específico:
  python example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --id-val "GO-5203500-366A293867A24EF5A446DA91E23F8548" --year 2025

  # 3. Rodar em lote para TODOS os imóveis do arquivo:
  python example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --all

  # 4. Rodar um lote de teste com os primeiros 2 imóveis:
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
    parser = argparse.ArgumentParser(description="Execução Cadastral de Parcelas - LAPIG")
    parser.add_argument("--input", type=str, default=None, help="Arquivo local (.shp, .gpkg, .fgb, .parquet), asset EE ou coordenada 'lat, lon'")
    parser.add_argument("--coord", type=str, default=None, help="Coordenada geográfica 'lat, lon' para busca automática no CAR")
    parser.add_argument("--id-col", type=str, default=None, dest="id_col", help="Nome da coluna de ID no arquivo vetorial (ex: cod_imovel, id)")
    parser.add_argument("--id-val", type=str, default=None, dest="id_val", help="Valor do ID caso queira rodar apenas para um imóvel específico")
    parser.add_argument("--all", action="store_true", help="Executa o pipeline em lote para TODOS os imóveis do arquivo vetorial")
    parser.add_argument("--limit", type=int, default=None, help="Limita o número de imóveis a processar no lote (ex: --limit 3)")
    parser.add_argument("--list-ids", action="store_true", dest="list_ids", help="Lista todos os IDs disponíveis no arquivo vetorial e encerra")
    parser.add_argument("--farm", type=str, default=None, choices=["GO", "MS"], help="Preset de teste rápido: GO ou MS")
    parser.add_argument("--filter", nargs=2, metavar=('CAMPO', 'VALOR'), default=None, help="Filtro opcional para asset do EE (ex: --filter cod_imovel GO-...)")
    parser.add_argument("--year", type=int, default=2025, help="Ano da análise (até 2025)")
    parser.add_argument("--collection", type=str, default="10m", choices=["10m", "30m"], help="Coleção MapBiomas (10m Col 4 ou 30m Col 11)")
    parser.add_argument("--output-dir", type=str, default="results", dest="output_dir", help="Diretório raiz para as pastas de resultados")
    parser.add_argument("--min_area", type=float, default=2.0, help="Área mínima da parcela em hectares")
    parser.add_argument("--straight-tol", type=float, default=22.0, dest="straight_tol", help="Tolerância de simplificação/retificação de cercas (metros)")
    parser.add_argument("--no-pasture-filter", action="store_true", dest="no_pasture", help="Desabilita o filtro do MapBiomas e gera apenas a segmentação pura")
    parser.add_argument("--project", type=str, default=None, help="Projeto Google Earth Engine (opcional, usa a sessão ativa por padrão)")

    args = parser.parse_args()

    # Recurso: Listar IDs do arquivo vetorial se solicitado
    if args.list_ids:
        target_file = args.input or "car_goias_area_imovel_epsg4674_hottest.gpkg"
        if not os.path.isabs(target_file) and not os.path.exists(target_file):
            for base in [os.getcwd(), parent_dir, script_dir]:
                c = os.path.join(base, target_file)
                if os.path.exists(c):
                    target_file = c
                    break

        if not os.path.exists(target_file):
            print(f"❌ Arquivo não encontrado: {target_file}")
            sys.exit(1)

        id_col = args.id_col or "cod_imovel"
        ext = os.path.splitext(target_file)[1].lower()
        gdf = gpd.read_parquet(target_file) if ext in ['.parquet', '.geoparquet'] else gpd.read_file(target_file)

        if id_col not in gdf.columns:
            print(f"❌ Coluna '{id_col}' não encontrada no arquivo.")
            print(f"   Colunas disponíveis: {list(gdf.columns)}")
            sys.exit(1)

        print("=" * 80)
        print(f"📋 IMÓVEIS DISPONÍVEIS EM: {os.path.basename(target_file)} ({len(gdf)} registros)")
        print("=" * 80)
        has_area = 'num_area' in gdf.columns
        has_mun = 'municipio' in gdf.columns

        for idx, row in gdf.iterrows():
            val = row[id_col]
            extra = []
            if has_mun:
                extra.append(f"Município: {row['municipio']}")
            if has_area:
                extra.append(f"Área: {row['num_area']:.1f} ha")
            extra_str = f" ({', '.join(extra)})" if extra else ""
            print(f"  [{idx+1:02d}] {val}{extra_str}")
        print("=" * 80)
        print(f"💡 Dica: Para processar um imóvel específico, use:")
        sample_id = gdf[id_col].iloc[0]
        print(f'   python example_run.py --input "{os.path.basename(target_file)}" --id-col "{id_col}" --id-val "{sample_id}"')
        sys.exit(0)

    # Determinar o input target
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

    # Caso o usuário queira rodar em lote para todos os imóveis (ou um limite)
    if args.all or (args.limit is not None and not args.id_val):
        print("=" * 80)
        print("🚜 EXECUTANDO PROCESSAMENTO EM LOTE (BATCH) - LAPIG")
        print(f"   Alvo de Entrada: {target}")
        print(f"   Coluna de ID:    {args.id_col or 'cod_imovel'}")
        if args.limit:
            print(f"   Limite de Imóveis: {args.limit}")
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
        print("🎉 PROCESSAMENTO EM LOTE CONCLUÍDO COM SUCESSO!")
        print(f"📦 Total de imóveis processados: {len(success_gdfs)}")
        if consolidated_shp:
            print(f"💾 Shapefile Geral Pasto:        {consolidated_shp}")
        print(f"📁 Pastas individuais geradas em: {output_dir}/")
        print("=" * 80)
        return

    # Execução individual para um único imóvel
    print("=" * 70)
    print(f"🌾 EXECUTANDO SEGMENTAÇÃO CADASTRAL DE PARCELAS (LAPIG)")
    print(f"   Alvo de Entrada: {target}")
    if args.id_col:
        print(f"   Coluna de ID:    {args.id_col}")
    if args.id_val:
        print(f"   Filtro ID:       {args.id_val}")
    if args.no_pasture:
        print(f"   Filtro Pasto:    DESATIVADO (Modo Segmentação Pura)")
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
        folder_name=None, # Cria a subpasta pré-definida com o ID do imóvel!
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

    print(f"\n🎉 Processo individual concluído com sucesso!")
    print(f"📦 Shapefile de Segmentação Pura (todos os talhões): {shp_pure} ({len(gdf_pure)} talhões)")
    if not args.no_pasture:
        print(f"💾 Shapefile Final de Pastagem Filtrada:             {shp_pasto} ({len(gdf_pasto)} parcelas)")

if __name__ == '__main__':
    main()
