#!/usr/bin/env python3
"""
Script CLI para geração instantânea do Dashboard de Performance e Hardware do LAPIG
Execução:
    python generate_dashboard.py
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Garante path local
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lapig_parcel_segmentation.benchmark_dashboard import collect_benchmark_dataset, generate_dashboard_html

def main():
    print("=" * 80)
    print("🚀 GERADOR DO DASHBOARD DE BENCHMARK & HARDWARE - LAPIG")
    print("=" * 80)
    
    results_dir = "results"
    gpkg = "car_goias_area_imovel_epsg4674_hottest.gpkg"
    
    print(f"🔍 Escaneando diretório de resultados: {results_dir}/...")
    records = collect_benchmark_dataset(results_dir=results_dir, gpkg_path=gpkg)
    print(f"📊 Total de propriedades compiladas: {len(records)}")
    
    if not records:
        print("⚠️ Nenhum registro com 'relatorio_benchmark_hardware.json' encontrado em results/.")
        sys.exit(1)
        
    out1 = generate_dashboard_html(records, output_html_path="results/dashboard_benchmark.html")
    out2 = generate_dashboard_html(records, output_html_path="dashboard_benchmark.html")
    
    print("\n" + "=" * 80)
    print("✨ DASHBOARD GERADO COM SUCESSO!")
    print(f"📁 Arquivo na pasta de resultados: {os.path.abspath(out1)}")
    print(f"📁 Arquivo na raiz do projeto:    {os.path.abspath(out2)}")
    print("💡 Você pode abrir o arquivo HTML diretamente no seu navegador (Chrome, Edge, Firefox)!")
    print("=" * 80)

if __name__ == '__main__':
    main()
