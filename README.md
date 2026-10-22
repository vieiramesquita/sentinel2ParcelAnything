# 🛰️ sentinel2parcel: Segmentação & Vetorização Cadastral de Piquetes de Pastagem (LAPIG)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![SAMGeo](https://img.shields.io/badge/Segment%20Anything-ViT--H%20Deep%20Learning-0080FF.svg)](https://samgeo.gishat.org)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Sentinel--2%20CS%2B-2BA55D.svg?logo=googleearthengine&logoColor=white)](https://earthengine.google.com)
[![Topology](https://img.shields.io/badge/Topology-Watertight%20(Zero%20Overlaps)-success.svg)](#️-3-arquitetura-do-pipeline)
[![Performance](https://img.shields.io/badge/Performance-5.9%20min%2Ffarm%20(RTX%202080%20Ti)-orange.svg)](#-2-desempenho--recursos-computacionais)
[![Benchmark Auditado](https://img.shields.io/badge/Benchmark-21%20Im%C3%B3veis%20%7C%2049.8k%20ha-blue.svg)](#-2-desempenho--recursos-computacionais)

**Laboratório de Processamento de Imagens e Geoprocessamento (LAPIG / UFG)**  
*O **sentinel2parcel** é um pipeline modular de alto desempenho para extração automatizada de piquetes e divisórias de pastagem com cercas retilíneas (mourão a mourão), topologia planar estanque (Zero Overlaps), validação zonal pelo MapBiomas (10m / 30m) e telemetria contínua de hardware.*

---

## 🗺️ Visão Geral: As Duas Abordagens Consolidadas

O projeto organiza a segmentação e vetorização cadastral de pastagens em **duas abordagens complementares e independentes**:

```
Vectorization_Pro/
├── 🖱️ interactive_hq_sam/          # ABORDAGEM 1: Segmentação Interativa Visual (Human-in-the-Loop)
│   ├── backend/                     # API FastAPI + PyTorch ViT-H na GPU (CUDA)
│   ├── frontend/                    # Interface Web Leaflet com dock de confirmação e refinamento (+/-)
│   └── run_interactive_app.py       # Inicializador: python run_interactive_app.py --port 8088
│
├── ⚙️ lapig_parcel_segmentation/    # ABORDAGEM 2: Segmentação e Vetorização Totalmente Automática em Lote
│   ├── pipeline_parcel_segmentation # Orquestrador principal de ponta a ponta
│   ├── roi_resolver.py              # Resolução de geometrias (CAR / Shapefile / Coordenadas)
│   ├── s2_mosaic.py                 # Mosaicos temporais Sentinel-2 Harmonized (GEE CS+)
│   ├── sam_segmenter.py             # Inferência SAMGeo ViT-H em grade com controle estrito de VRAM
│   ├── pasture_classifier.py        # Validação zonal MapBiomas pastagem (10m Col 4 / 30m Col 11)
│   ├── cadastral_vectorizer.py      # Grafo Planar Watertight (Zero Overlaps, cercas retilíneas)
│   └── example_run.py               # CLI: python -m lapig_parcel_segmentation.example_run --all
│
├── 🌐 sentinel2parcel_test/         # SHOWCASE WEB: Demonstração estática pública no GitHub Pages
│   └── (Live em https://vieiramesquita.github.io/sentinel2parcel_test/)
│
├── 🧠 models/                       # Pesos da rede neural (sam_vit_h_4b8939.pth)
├── 📊 results/                      # Vetores gerados (.shp / .gpkg), estatísticas e relatórios de benchmark
└── 📓 run_in_jupyter.ipynb          # Notebook de demonstração e execução interativa da abordagem automática
```

| Critério | Abordagem 1: Interativa Visual (`interactive_hq_sam/`) | Abordagem 2: Automática em Lote (`lapig_parcel_segmentation/`) |
| :--- | :--- | :--- |
| **Público / Uso** | Operador humano, auditoria, refinamento fino de perímetros complexos | Processamento massivo de fazendas inteiras ou municípios |
| **Intervenção** | **Human-in-the-Loop:** Cliques com pontos positivos (`+`) e negativos (`−`) | **Zero intervenção:** Execução autônoma ponta a ponta |
| **Modelo de IA** | HQ-SAM ViT-H com amostragem pontual interativa e recortes inteligentes | SAMGeo ViT-H com segmentação automática em grade |
| **Interface** | Web (Leaflet dark glassmorphism na porta 8088 com dock neon) | Linha de comando (CLI) ou script Python / Jupyter |
| **Validação** | O operador valida visualmente e clica em "Confirmar Piquete" | Dominância zonal automática pelo MapBiomas (pastagem) |
| **Saídas** | GeoJSON exportável pelo navegador com métricas em tempo real | Shapefiles duplos (`segmentacao_pura` e `pastagem_final`) |

---

## 📸 1. Galeria de Resultados: Do Satélite ao Vetor Cadastral

O **sentinel2parcel** transforma imagens multiespectrais do Sentinel-2 em polígonos vetoriais prontos para SIG e cadastro rural. Ao contrário de abordagens ingênuas de vetorização (como `gdal_polygonize` ou simplificação independente de polígonos), o método utiliza **Grafos Planares** que simplificam arestas compartilhadas, eliminam dentes de serra (degraus de 10m) e garantem **0 sobreposições e 0 frestas (Topologia Watertight)**.

### Caso 1: Imóvel Rural em Caiapônia (GO) — 749,52 ha
> **Destaques:** Segmentação de 42 piquetes de pastagem retificados. Preservação precisa da mata ciliar (APP) e corpo d'água ao centro, eliminando completamente os enclaves e ilhas de ruído do Deep Learning.
![Resultados Caiapônia](images/fig_caiaponia_results.png)

---

### Caso 2: Imóvel Rural em Paranaíba (MS) — 1.085,22 ha
> **Destaques:** Imóvel de grande extensão cortado diagonalmente por ferrovia e rodovia. O algoritmo respeitou as divisórias estruturais externas e extraiu 54 piquetes retilíneos bem delimitados.
![Resultados Paranaíba](images/fig_paranaiba_results.png)

---

### Caso 3: Imóvel Rural em Itapirapuã (GO) — 193,59 ha
> **Destaques:** Alta paridade com gabarito manual de fazenda piloto. Retificação perfeita de mourão a mourão em divisórias retas de cerca, contornando a drenagem natural no vale.
![Resultados Itapirapuã](images/fig_itapirapua_results.png)

---

## ⚡ 2. Desempenho & Recursos Computacionais

O **sentinel2parcel** foi submetido a um teste de estresse e auditoria contínua em lote (*stress test batch*) com **21 imóveis rurais** (20 em Goiás e 1 no Mato Grosso do Sul) selecionados em polos de pecuária intensiva e grande extensão territorial, totalizando **49.760,02 hectares** de área perimetral cadastral analisada.

### Visão Geral de Performance Auditada (Hardware: RTX 2080 Ti 11 GB)
* **Volume Cadastral:** **1.861 piquetes de pastagem** retilíneos delimitados (**44.253,17 ha**) e **2.010 talhões físicos** na segmentação pura (**47.479,02 ha**).
* **Tempo Total de GPU:** **122,81 minutos (~2,05 horas)** de processamento contínuo acumulado.
* **Tempo Médio por Imóvel:** **~5,85 minutos** (~351 s), variando de **2,5 min** em fazendas médias de topografia uniforme a **8 a 9,5 min** em megaprojetos de grande porte (>9.000 a 10.800 ha).
* **Pico de VRAM Alocada:** Média de **6,41 GB**, operando confortavelmente em **6,47 GB** (58,8% da GPU) e com pico máximo de **9,54 GB** (86,7% da GPU) no imóvel de maior extensão territorial (10.828,8 ha em Goiás).
* **Topologia Watertight:** **0 sobreposições e 0 frestas** comprovadas em 100% dos 1.861 piquetes gerados.
* **Estabilidade Térmica:** Temperatura da GPU permaneceu controlada (pico registrado de **70 °C** sob carga total de inferência).

### Decomposição do Esforço Computacional (Profiling por Etapa)

| Etapa do Pipeline | Participação (%) | Dispositivo Principal | Tempo Típico por Fazenda |
| :--- | :--- | :--- | :--- |
| **Inferência SAMGeo ViT-H** | **86,2%** | GPU (CUDA / Tensor Cores) | ~5,0 min |
| **Download Mosaico Sentinel-2 (GEE)** | **8,1%** | Rede & Google Cloud API | ~28 s |
| **Classificação Zonal MapBiomas (10m/30m)** | **1,6%** | Google Earth Engine / CPU | ~5,6 s |
| **Vetorização Cadastral e Grafo Planar** | **0,6%** | CPU (Multithread) | **~2,1 s** *(Ultrarrápido)* |

### Auditoria Detalhada dos 21 Imóveis Processados:

| Município | ID Imóvel (CAR) | Área CAR | Piquetes Pasto | Talhões Puros | Tempo Total | Pico VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Goiás (GO)** | `GO-5208905-F49642881887...` | 10.828,80 ha | 345 | 354 | 8,2 min | 9,54 GB |
| **Nova Crixás (GO)** | `GO-5214838-32EB159D2A3C...` | 9.719,91 ha | 224 | 267 | 1,0 min | 0,01 GB |
| **Jataí (GO)** | `GO-5211909-AD13173067F8...` | 9.107,04 ha | 234 | 248 | 9,5 min | 8,53 GB |
| **Bom Jesus de Goiás (GO)** | `GO-5203500-366A293867A2...` | 4.476,31 ha | 138 | 178 | 9,5 min | 6,47 GB |
| **Britânia (GO)** | `GO-5203807-146172AFEDAF...` | 2.629,80 ha | 75 | 79 | 8,0 min | 6,47 GB |
| **Serranópolis (GO)** | `GO-5220504-AE59920E9ABB...` | 2.114,08 ha | 96 | 99 | 8,7 min | 6,47 GB |
| **Santa Fé de Goiás (GO)** | `GO-5219258-B3304BD38B7B...` | 1.889,02 ha | 119 | 121 | 4,6 min | 6,47 GB |
| **Ipameri (GO)** | `GO-5210109-9C728F04622C...` | 1.763,45 ha | 124 | 134 | 8,6 min | 6,47 GB |
| **Paranaíba (MS)** | `MS-5006309-8DC93A48C242...` | 1.085,22 ha | 54 | 55 | 3,7 min | 6,47 GB |
| **São Francisco de Goiás (GO)** | `GO-5219902-041335F6BCD1...` | 960,44 ha | 74 | 75 | 6,4 min | 6,47 GB |
| **Colinas do Sul (GO)** | `GO-5205521-DA104FC8E15A...` | 896,58 ha | 35 | 43 | 6,7 min | 6,47 GB |
| **Goiás (GO)** | `GO-5208905-08E150868E2A...` | 828,95 ha | 76 | 76 | 2,8 min | 6,47 GB |
| **Caiapônia (GO)** | `GO-5204409-01D1AD1F9586...` | 798,23 ha | 42 | 54 | 2,5 min | 6,47 GB |
| **Iaciara (GO)** | `GO-5209903-C8653C1E6389...` | 532,20 ha | 32 | 32 | 6,4 min | 6,47 GB |
| **Edéia (GO)** | `GO-5207402-1D40C8AAD650...` | 530,02 ha | 45 | 45 | 3,4 min | 6,47 GB |
| **Jussara (GO)** | `GO-5212204-38A49ED0A118...` | 514,25 ha | 29 | 29 | 2,5 min | 6,47 GB |
| **Jussara (GO)** | `GO-5212204-0B3CAD6C8F1D...` | 420,28 ha | 42 | 42 | 7,9 min | 6,47 GB |
| **Crixás (GO)** | `GO-5206404-499D0D75A139...` | 233,81 ha | 37 | 37 | 5,8 min | 6,47 GB |
| **Itapirapuã (GO)** | `GO-5211008-46C08C6235D7...` | 193,59 ha | 14 | 14 | 3,1 min | 6,47 GB |
| **Luziânia (GO)** | `GO-5212501-89730760C68B...` | 167,76 ha | 16 | 17 | 6,4 min | 6,47 GB |
| **Goiânia (GO)** | `GO-5208707-4045057E90C9...` | 70,28 ha | 10 | 11 | 7,1 min | 6,47 GB |
> **💡 Nota sobre Hardware & Escalabilidade:**  
> 1. **Vetorização Leve na CPU:** A vetorização em grafo planar consome apenas **0,6%** do tempo total, viabilizando simplificação rigorosa de cercas sem gargalo de processamento.
> 2. **Viabilidade em GPUs Comerciais:** Com consumo médio de **6,4 GB de VRAM**, o pipeline roda com folga em GPUs de 8 GB a 16 GB (RTX 3060 12GB, RTX 4060 Ti 16GB, RTX 2080 Ti 11GB).
> 3. **Fine-Tuning e Escala:** O gargalo dominante (86,2%) é a inferência ViT-H do SAMGeo. Para escala territorial (municípios e biomas), recomenda-se uso de GPUs corporativas (A100, H100) e pesos customizados para vegetação campestre tropical.

### Gerenciamento Contínuo de Memória de Vídeo (Zero VRAM Leak)
Para viabilizar execuções em lote sem risco de estouros de memória (*CUDA Out of Memory*):
* A rotina [`clear_gpu_memory()`](lapig_parcel_segmentation/sam_segmenter.py) é acionada compulsória e automaticamente ao término de cada imóvel, deletando explicitamente tensores do `SamGeo`, invocando `gc.collect()`, liberando o *caching allocator* do PyTorch (`torch.cuda.empty_cache()`) e limpando blocos IPC (`torch.cuda.ipc_collect()`).
* Isso garante que cada imóvel inicie seu ciclo com **0 MB de VRAM residual**, permitindo processar dezenas ou centenas de fazendas sequencialmente de forma ininterrupta.

> Um dashboard interativo local de telemetria completa com gráficos dinâmicos e filtros por município pode ser gerado a qualquer momento via `python generate_dashboard.py` (arquivo `dashboard_benchmark.html`).

---

## 🏗️ 3. Arquitetura do Pipeline

```
                   Entrada da Área de Interesse (Universal ROI)
       ┌────────────────────────┬────────────────────────┬────────────────────────┐
       │  Arquivo Local         │  Asset Earth Engine    │  Coordenada            │
       │  .shp, .gpkg, .fgb,    │  'projects/...' ou     │  (lat, lon) ->         │
       │  .parquet, .geojson    │  ee.FeatureCollection  │  Busca no CAR GEE      │
       │  (com id_column)       │  (com filtro de ID)    │  (auto-extrai CAR ID)  │
       └───────────┬────────────┴───────────┬────────────┴───────────┬────────────┘
                   └────────────────────────┼────────────────────────┘
                                            │
                                            ▼
                       [Pasta Pré-definida por Imóvel]
                          results/{ID_DO_IMOVEL}/
                                            │
                                            ▼
                                  Google Earth Engine
                          ┌─────────────────────────────────┐
                          │  Sentinel-2 Harmonized (CS+)    │
                          │  Bandas: NDVI max, median, min  │
                          └─────────────────┬───────────────┘
                                            │
                                            ▼
                                 [Módulo: s2_mosaic]
                                            │ (GeoTIFF multi-temporal)
                                            ▼
                             [Módulo: sam_segmenter]
                         (SAMGeo ViT-H com GPU CUDA 8GB/11GB)
                                            │ (Máscaras brutas)
                                            ▼
                          [Módulo: cadastral_vectorizer]
                          ├── Eliminação de buracos/ilhas internas
                          ├── Grafo Planar e Retificação de Cercas Retas
                          └── Reconstrução Estanque (Zero Overlaps)
                                            │
                                            ▼
                     [Recorte Estrito pelo Limite do CAR]
                                            │
         ┌──────────────────────────────────┴──────────────────────────────────┐
         ▼                                                                     ▼
 [CAMADA 1: SEGMENTAÇÃO PURA]                                         [CAMADA 2: FILTRO PASTAGEM]
 Todos os talhões físicos no CAR                                      Apenas talhões de pastagem
 com colunas de auditoria:                                            validados por dominância zonal:
 (pasto_cov, ndvi_med, is_pasto, classe)                              (MapBiomas 10m Col 4 / 30m Col 11)
         │                                                                     │
         ▼                                                                     ▼
 Shapefile: piquetes_segmentacao_pura_{ID}.shp      Shapefile: piquetes_finais_cercas_retas_{ID}.shp
```

### Camadas de Saída Geradas:
1. **`piquetes_segmentacao_pura_{ID}.shp`:** Delimitação de **todos** os compartimentos e divisórias físicas identificadas dentro da propriedade, contendo colunas com percentual de sobreposição de pastagem, NDVI médio e classe de uso.
2. **`piquetes_finais_cercas_retas_{ID}.shp`:** Camada estrita de pastagem, filtrada por dominância zonal do MapBiomas (pastagens com mais de 30% a 50% de área validada).
3. **`nao_pastagem_auditoria.shp`:** Geometrias descartadas (lavouras, matas nativas, represas, infraestruturas) para fins de conferência técnica.
4. **`relatorio_benchmark_hardware.json` & `.txt`:** Auditoria de hardware e consumo da GPU por propriedade.

---

## 🚀 4. Instalação e Execução

### Pré-requisitos
- Python 3.10 ou superior
- GPU NVIDIA com suporte a CUDA (mínimo 8 GB VRAM recomendado)
- Conta autenticada no Google Earth Engine (`earthengine authenticate`)

### Instalação das Dependências

```bash
# Clone o repositório privado
git clone https://github.com/vieiramesquita/sentinel2parcel.git
cd sentinel2parcel

# Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale os pacotes principais
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install segment-geospatial earthengine-api geopandas shapely rasterio matplotlib pandas duckdb psutil plotly
```

### Execução via Linha de Comando (CLI)

```bash
# Processar um imóvel específico a partir de um GeoPackage local do CAR
python -m lapig_parcel_segmentation.example_run \
    --roi car_goias_area_imovel_epsg4674_hottest.gpkg \
    --id-col cod_imovel \
    --id-val GO-5204409-01D1AD1F9586415E9D46BF93115D27DF \
    --year 2024 \
    --straight-tol 8.0 \
    --gee-project ee-vieiramesquita
```

### Execução Interativa via Jupyter Notebook

Abra o arquivo [`run_in_jupyter.ipynb`](run_in_jupyter.ipynb) para visualizar o fluxo passo a passo com mapas interativos do Folium/Geemap e gráficos de telemetria integrados.

### Visualização do Dashboard

Abra o arquivo `dashboard_benchmark.html` diretamente em qualquer navegador moderno (Chrome, Edge, Firefox) para explorar todas as propriedades processadas de forma interativa.

---

## 👥 Créditos & Equipe

* **Laboratório de Processamento de Imagens e Geoprocessamento (LAPIG)** — Universidade Federal de Goiás (UFG)
* **Desenvolvimento e Otimização:** Vinícius Mesquita & Equipe LAPIG
* **Suporte de IA:** Antigravity (Google DeepMind)
