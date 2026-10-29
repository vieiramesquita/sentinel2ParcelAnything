#!/usr/bin/env python3
"""
Runner para o aplicativo de Segmentação Interativa por Amostragem com HQ-SAM
LAPIG - Laboratório de Processamento de Imagens e Geoprocessamento (UFG)

Uso:
  python run_interactive_app.py [--port 8088] [--host 0.0.0.0]
"""

import os
import sys
import argparse
import uvicorn

# Adiciona o diretório interativo e a raiz do projeto ao sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="LAPIG HQ-SAM Interactive Parcel Segmenter")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP para escuta (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8088, help="Porta TCP (default: 8088)")
    parser.add_argument("--reload", action="store_true", help="Habilita auto-reload")
    args = parser.parse_args()

    print("=" * 70)
    print("🌿 LAPIG HQ-SAM - Segmentador Interativo de Parcelas por Amostragem")
    print("=" * 70)
    print(f"📡 Servidor iniciando em: http://127.0.0.1:{args.port}")
    print(f"📡 Acesso de rede local:  http://localhost:{args.port}")
    print("=" * 70)

    # Inicia o servidor FastAPI apontando para o app
    os.chdir(PROJECT_ROOT)  # Permite acesso a results/ e gpkg da raiz
    uvicorn.run("interactive_hq_sam.backend.app:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
