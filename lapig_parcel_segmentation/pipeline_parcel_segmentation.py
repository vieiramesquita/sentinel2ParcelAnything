#!/usr/bin/env python3
"""
Pipeline Modular de Segmentação e Vetorização Cadastral de Parcelas (LAPIG)
Substituição definitiva e modularizada do notebook parcelSegmentation_SAMGeo_LAPIG.ipynb

Suporta execução individual e processamento em lote (Batch Processing para todos os IDs de um arquivo).
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

# Suporte a importação relativa dentro de pacote ou importação direta como script
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
    id_column=None,             # Nome da coluna de ID no arquivo vetorial (ex: 'cod_imovel', 'id')
    id_value=None,              # Valor de ID ou 'all' para rodar todos
    year=2025,
    start_month=3,
    end_month=10,
    buffer_size=500,
    folder_name=None,           # Se None ou 'auto': cria pasta pré-definida results/{ID_IMOVEL}
    output_base_dir="results",  # Diretório base quando folder_name for automático
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
    Executa a análise para um imóvel ou delega para lote caso id_value='all'.
    
    Novidades:
    1. Segmentação Pura: Gera 'parcelas_segmentacao_pura_{ID}.shp' contendo todos os talhões
       físicos no CAR antes de qualquer descarte pelo MapBiomas.
    2. Benchmark de Hardware: Monitora em segundo plano VRAM (pico/média), RAM do sistema,
       uso de CPU e GPU compute durante todas as etapas e gera relatório técnico.
    """
    # Se id_value for 'all', executa processamento em lote
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

    # Garante VRAM limpa antes de iniciar o processamento da parcela
    clear_gpu_memory()

    # Inicialização do Monitor de Benchmark de Hardware
    monitor = HardwareBenchmarkMonitor(sample_interval=1.0) if (enable_benchmark and HardwareBenchmarkMonitor) else None
    if monitor:
        monitor.start()
        monitor.set_phase("1. Resolução da Área de Interesse (ROI)")

    t_start = time.time()

    # 0. Resolução Universal da Área de Interesse (ROI)
    target = input_target if input_target is not None else roi
    if target is None:
        raise ValueError("❌ Nenhum input_target ou roi fornecido para análise.")

    roi_ee, gdf_car, roi_info = resolve_input_roi(
        input_target=target,
        id_column=id_column,
        id_value=id_value,
        car_property_filter=car_property_filter,
        car_assets=car_assets,
        buffer_radius_meters=coord_buffer_meters,
        ee_project=ee_project
    )

    # 1. Definição Pré-determinada da Pasta de Saída baseada no ID do Imóvel
    prop_id = roi_info.get("id_imovel") or roi_info.get("cod_imovel") or "imovel_segmentado"
    clean_id = sanitize_id(prop_id)

    if folder_name is None or str(folder_name).lower() in ["none", "auto", "default"]:
        folder_name = os.path.join(output_base_dir, clean_id)
    elif folder_name == "results":
        folder_name = os.path.join("results", clean_id)

    os.makedirs(folder_name, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"🚀 PIPELINE DE SEGMENTAÇÃO CADASTRAL LAPIG")
    print(f"   Imóvel / ID:      {clean_id}")
    print(f"   Pasta de Saída:   {folder_name}")
    print(f"   Ano Base:         {year} (Janela seca: meses {start_month:02d} a {end_month:02d})")
    print(f"   MapBiomas:        Coleção {mapbiomas_collection} (cobertura até 2025)")
    print("=" * 70)

    # 2. Download do Mosaico Sentinel-2 Multi-temporal (NDVI max, median, min)
    if monitor:
        monitor.set_phase("2. Download Mosaico Sentinel-2 (GEE)")

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

    # 3. Segmentação SAMGeo com GPU
    if monitor:
        monitor.set_phase("3. Inferência SAMGeo (ViT-H GPU)")

    masks_tif_path = os.path.join(folder_name, "masks.tif")
    run_samgeo_segmentation(
        image_path=s2_path,
        output_mask_path=masks_tif_path,
        model_type=model_type,
        checkpoint_dir="models",
        force_rerun=(not skip_sam_if_exists)
    )

    # 4. Vetorização Cadastral e Retificação de Cercas Retas
    if monitor:
        monitor.set_phase("4. Vetorização e Regularização Cadastral (CPU)")

    gdf_parcels = vectorize_and_straighten_parcels(
        sam_mask_tif=masks_tif_path,
        min_area_ha=min_area_ha,
        straight_tol=straight_tol
    )

    # 5. Segmentação Pura no Limite do CAR (Todos os talhões físicos dentro da fazenda)
    if monitor:
        monitor.set_phase("5. Segmentação Pura no Perímetro CAR")

    gdf_pure_car = clip_parcels_by_car(
        gdf_parcels=gdf_parcels,
        car_geom_or_gdf=gdf_car,
        min_area_ha=min_area_ha * 0.5
    )
    gdf_pure_car['cod_car'] = clean_id
    pure_shp_path = os.path.join(folder_name, f"parcelas_segmentacao_pura_{clean_id}.shp")

    # 6. Classificação Zonal e Filtragem Temática de Pastagem (MapBiomas)
    if monitor:
        monitor.set_phase("6. Classificação Zonal MapBiomas (GEE/CPU)")

    if filter_by_pasture:
        print(f"\n🌿 Processando classificação zonal de pastagem com MapBiomas ({mapbiomas_collection})...")
        try:
            scale_val = 30 if str(mapbiomas_collection).lower() in ['30', '30m', 'col11'] else 10
            pasture_ee = get_mapbiomas_pasture_ee(roi_ee, year=min(year, 2025), collection=mapbiomas_collection, ee_project=ee_project)
            mb_mask_tif = os.path.join(folder_name, f"temp_mb_mask_{mapbiomas_collection}.tif")
            export_pasture_mask_local(pasture_ee, roi_ee, s2_path, mb_mask_tif, scale=scale_val)

            # Classifica diretamente a segmentação pura do CAR
            gdf_classified, gdf_others = classify_and_filter_pasture_parcels(
                gdf_parcels=gdf_pure_car,
                pasture_mask_tif=mb_mask_tif,
                min_pasture_coverage=min_pasture_coverage,
                s2_ndvi_tif=s2_path
            )

            # Combina pasto e outros para formar a segmentação pura completa anotada (todos os talhões)
            import pandas as pd
            gdf_all_annotated = pd.concat([gdf_classified, gdf_others], ignore_index=True)
            gdf_all_annotated = gpd.GeoDataFrame(gdf_all_annotated, crs=gdf_classified.crs)

            # Camada 1: Segmentação Pura Completa (com atributos do MapBiomas anotados)
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
            print(f"📦 Shapefile de Segmentação Pura Salvo (Todos os {len(gdf_pure_clean_4326)} talhões): {pure_shp_path}")

            # Camada 2: Apenas Parcelas de Pastagem
            gdf_pasture = gdf_pure_clean_4326[gdf_pure_clean_4326['is_pasto'] == 1].copy().reset_index(drop=True)
            gdf_pasture['id_parcela'] = range(1, len(gdf_pasture) + 1)
            final_shp_path = os.path.join(folder_name, f"parcelas_finais_cercas_retas_{clean_id}.shp")
            gdf_pasture.to_file(final_shp_path)
            print(f"💾 Shapefile Final de Pastagem Salvo ({len(gdf_pasture)} parcelas): {final_shp_path}")

            # Camada 3: Não-Pastagem (Auditoria de áreas excluídas: mata, lavoura, etc.)
            gdf_others_clean = gdf_pure_clean_4326[gdf_pure_clean_4326['is_pasto'] == 0].copy().reset_index(drop=True)
            if len(gdf_others_clean) > 0:
                others_shp = os.path.join(folder_name, "nao_pastagem_auditoria.shp")
                gdf_others_clean.to_file(others_shp)
                print(f"📁 Camada de auditoria de não-pastagem salva: {others_shp}")

            gdf_final_clean_4326 = gdf_pasture

        except Exception as e:
            print(f"⚠️ Aviso na consulta ao MapBiomas: {e}. Mantendo segmentação pura.")
            gdf_pure_clean_4326 = gdf_pure_car.to_crs(epsg=4326)
            if 'id_parcela' not in gdf_pure_clean_4326.columns:
                gdf_pure_clean_4326['id_parcela'] = range(1, len(gdf_pure_clean_4326) + 1)
            gdf_pure_clean_4326.to_file(pure_shp_path)
            gdf_final_clean_4326 = gdf_pure_clean_4326
            final_shp_path = pure_shp_path
    else:
        # Sem filtro de pastagem: o resultado final é a própria segmentação pura
        gdf_pure_clean_4326 = gdf_pure_car.to_crs(epsg=4326)
        if 'id_parcela' not in gdf_pure_clean_4326.columns:
            gdf_pure_clean_4326['id_parcela'] = range(1, len(gdf_pure_clean_4326) + 1)
        gdf_pure_clean_4326.to_file(pure_shp_path)
        gdf_final_clean_4326 = gdf_pure_clean_4326
        final_shp_path = pure_shp_path
        print(f"📦 Shapefile de Segmentação Pura Salvo ({len(gdf_pure_clean_4326)} talhões): {pure_shp_path}")

    # 7. Relatório Estatístico Cadastral
    if monitor:
        monitor.set_phase("7. Exportação e Estatísticas Cadastrais")

    generate_cadastral_stats(gdf_final_clean_4326, title=f"RELATÓRIO FINAL DE PASTAGEM ({clean_id})")

    # 8. Relatório de Benchmark de Hardware
    bench_summary = None
    if monitor:
        bench_summary = monitor.stop()
        txt_path, json_path = monitor.save_report(folder_name, filename_prefix="relatorio_benchmark_hardware")
        print("\n" + monitor.generate_report_text())
        print(f"📊 Relatório de Hardware salvo em: {txt_path}")

    total_time = str(timedelta(seconds=int(time.time() - t_start)))
    print(f"✨ Processamento do imóvel concluído em: {total_time}")
    print("=" * 70 + "\n")

    # Metadados no GeoDataFrame retornado
    gdf_final_clean_4326.attrs['shp_pasture'] = final_shp_path
    gdf_final_clean_4326.attrs['shp_pure'] = pure_shp_path
    gdf_final_clean_4326.attrs['gdf_pure'] = gdf_pure_clean_4326
    if bench_summary:
        gdf_final_clean_4326.attrs['benchmark'] = bench_summary

    # Liberação final de VRAM ao concluir o imóvel
    clear_gpu_memory()

    if return_pure:
        return gdf_final_clean_4326, final_shp_path, gdf_pure_clean_4326, pure_shp_path
    return gdf_final_clean_4326, final_shp_path

def run_batch_analysis_lapig(
    input_target,
    id_column="cod_imovel",
    id_list=None,
    id_value=None,              # Suporte a ID único ou lista para compatibilidade total
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
    Executa o pipeline em lote (batch) para TODOS os imóveis presentes no arquivo vetorial.
    Gera as pastas individuais de cada imóvel e compila shapefiles consolidados:
    - parcelas_todos_imoveis_consolidados.shp (pastagem filtrada)
    - parcelas_todos_imoveis_segmentacao_pura.shp (segmentação física pura de todos os talhões)
    - relatorio_benchmark_hardware_lote.txt (monitoramento global de hardware)
    - Liberação contínua de VRAM garantida a cada imóvel processado.
    """
    print("\n" + "=" * 80)
    print(f"🚜 INICIANDO PROCESSAMENTO EM LOTE (BATCH) - LAPIG")
    print(f"   Arquivo de Entrada: {os.path.basename(str(input_target))}")
    print(f"   Coluna de ID:       {id_column}")
    print("=" * 80)

    # Suporte a id_value no batch se fornecido
    if id_value and not id_list:
        id_list = [id_value] if not isinstance(id_value, (list, tuple, set)) else list(id_value)

    # Ler lista de IDs
    ext = os.path.splitext(str(input_target))[1].lower()
    if ext in ['.parquet', '.geoparquet']:
        gdf_all = gpd.read_parquet(input_target)
    elif os.path.exists(str(input_target)):
        gdf_all = gpd.read_file(input_target)
    else:
        raise FileNotFoundError(f"Arquivo vetorial não encontrado: {input_target}")

    if id_column not in gdf_all.columns:
        raise ValueError(f"Coluna '{id_column}' não encontrada. Colunas disponíveis: {list(gdf_all.columns)}")

    # Obter IDs únicos
    unique_ids = gdf_all[id_column].dropna().unique().tolist()
    if id_list:
        unique_ids = [i for i in unique_ids if str(i) in [str(x) for x in id_list]]
    if max_imoveis:
        unique_ids = unique_ids[:int(max_imoveis)]

    total = len(unique_ids)
    print(f"📋 Total de imóveis a serem processados: {total}")

    success_gdfs_pasture = []
    success_gdfs_pure = []
    success_paths = []
    failed_ids = []

    # Monitor de hardware geral do lote
    batch_monitor = HardwareBenchmarkMonitor(sample_interval=2.0) if (enable_benchmark and HardwareBenchmarkMonitor) else None
    if batch_monitor:
        batch_monitor.start()

    t_batch_start = time.time()

    for idx, prop_id in enumerate(unique_ids):
        print("\n" + "#" * 80)
        print(f"🚜 [{idx+1}/{total}] PROCESSANDO IMÓVEL: {prop_id}")
        print("#" * 80)

        # 🧹 Esvazia a VRAM no início de cada parcela/iteração
        clear_gpu_memory(verbose=True)

        if batch_monitor:
            batch_monitor.set_phase(f"Imóvel [{idx+1}/{total}] {prop_id}")

        try:
            gdf_res, shp_res, gdf_pure_res, shp_pure_res = run_analysis_lapig(
                input_target=input_target,
                id_column=id_column,
                id_value=prop_id,
                year=year,
                start_month=start_month,
                end_month=end_month,
                buffer_size=buffer_size,
                folder_name=None, # Usa subpasta results/{prop_id}
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
            print(f"❌ Erro ao processar o imóvel {prop_id}: {e}")
            failed_ids.append((prop_id, str(e)))
        finally:
            # 🧹 Esvazia a VRAM compulsoriamente ao final de cada parcela/iteração
            clear_gpu_memory(verbose=True)

    # Compilar Shapefiles Consolidados se solicitado
    consolidated_pasture_path = None
    consolidated_pure_path = None

    if consolidate_shp:
        # 1. Shapefile Consolidado de Pastagem
        if success_gdfs_pasture:
            try:
                gdf_consolidated_pasto = pd.concat(success_gdfs_pasture, ignore_index=True)
                gdf_consolidated_pasto = gpd.GeoDataFrame(gdf_consolidated_pasto, geometry='geometry', crs=success_gdfs_pasture[0].crs)
                consolidated_pasture_path = os.path.join(output_base_dir, "parcelas_todos_imoveis_consolidados.shp")
                gdf_consolidated_pasto.to_file(consolidated_pasture_path)
                print(f"\n📦 Shapefile Consolidado de Pastagem Salvo: {consolidated_pasture_path}")
            except Exception as e:
                print(f"⚠️ Não foi possível consolidar shapefile de pasto: {e}")

        # 2. Shapefile Consolidado de Segmentação Pura (Todos os talhões físicos)
        if success_gdfs_pure:
            try:
                gdf_consolidated_pure = pd.concat(success_gdfs_pure, ignore_index=True)
                gdf_consolidated_pure = gpd.GeoDataFrame(gdf_consolidated_pure, geometry='geometry', crs=success_gdfs_pure[0].crs)
                consolidated_pure_path = os.path.join(output_base_dir, "parcelas_todos_imoveis_segmentacao_pura.shp")
                gdf_consolidated_pure.to_file(consolidated_pure_path)
                print(f"📦 Shapefile Consolidado de Segmentação Pura Salvo: {consolidated_pure_path}")
            except Exception as e:
                print(f"⚠️ Não foi possível consolidar shapefile de segmentação pura: {e}")

    # Relatório Final do Batch & Benchmark
    if batch_monitor:
        batch_monitor.stop()
        batch_txt, batch_json = batch_monitor.save_report(output_base_dir, filename_prefix="relatorio_benchmark_hardware_lote")
        print("\n" + batch_monitor.generate_report_text())
        print(f"📊 Relatório de Hardware do Lote salvo em: {batch_txt}")

    elapsed = str(timedelta(seconds=int(time.time() - t_batch_start)))
    print("\n" + "=" * 80)
    print("🏁 RELATÓRIO DO PROCESSAMENTO EM LOTE (BATCH)")
    print("=" * 80)
    print(f"🔹 Total de imóveis no lote:       {total}")
    print(f"🔹 Concluídos com sucesso:         {len(success_gdfs_pasture)}")
    print(f"🔹 Falhas / Erros:                 {len(failed_ids)}")
    print(f"⏱️ Tempo total de processamento:   {elapsed}")
    if consolidated_pasture_path:
        total_pasto_ha = sum([g['area_ha'].sum() for g in success_gdfs_pasture if 'area_ha' in g.columns])
        total_parcelas = sum([len(g) for g in success_gdfs_pasture])
        print(f"📊 Total de parcelas validadas:   {total_parcelas} ({total_pasto_ha:.2f} ha)")
        print(f"💾 Shapefile Consolidado:         {consolidated_pasture_path}")
    if consolidated_pure_path:
        total_puro_ha = sum([g['area_ha'].sum() for g in success_gdfs_pure if 'area_ha' in g.columns])
        total_talhoes = sum([len(g) for g in success_gdfs_pure])
        print(f"📊 Total talhões físicos (puro):   {total_talhoes} ({total_puro_ha:.2f} ha)")
        print(f"💾 Shapefile Puro Consolidado:     {consolidated_pure_path}")
    print("=" * 80 + "\n")

    return success_gdfs_pasture, consolidated_pasture_path

if __name__ == '__main__':
    print("Pipeline de segmentação cadastral pronto para execução.")
