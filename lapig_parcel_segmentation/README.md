# 🛰️ Segmentação e Vetorização Cadastral de Piquetes de Pastagem (LAPIG)

**Laboratório de Processamento de Imagens e Geoprocessamento (LAPIG / UFG)**  
*Pipeline modular para extração automatizada de piquetes rurais com cercas retilíneas (mourão a mourão), segmentação cadastral pura, classificação temática de pastagem via MapBiomas (até 2025), recorte pelo CAR e benchmark automático de hardware.*

---

## 📌 1. Visão Geral da Arquitetura

Este pacote substitui e modulariza os procedimentos do notebook `parcelSegmentation_SAMGeo_LAPIG.ipynb`, eliminando as etapas que degradavam a geometria das cercas e introduzindo uma abordagem cadastral baseada em **Entrada Universal de ROI**, **Organização Automática por ID de Imóvel**, **Segmentação Pura**, **Grafos Planares**, **Absorção de Enclaves**, **Dominância Zonal do MapBiomas** e **Benchmark Contínuo de Hardware**.

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
                                           │ (roi_ee + gdf_car)
                                           ▼
                                 Google Earth Engine
                         ┌─────────────────────────────────┐
                         │  Sentinel-2 Harmonized (CS+)    │
                         │  Bandas: NDVI max, median, min  │
                         └─────────────────┬───────────────┘
                                           │
                                           ▼
                                [Módulo 1: s2_mosaic]
                                           │ (GeoTIFF multi-temporal)
                                           ▼
                            [Módulo 2: sam_segmenter]
                        (SAMGeo ViT-H com GPU CUDA 8GB/11GB)
                                           │ (Máscaras brutas)
                                           ▼
                         [Módulo 3: cadastral_vectorizer]
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

---

## ⚡ 2. Benchmark de Hardware Automático

Para computadores com GPUs dedicadas (ex: **RTX 2080 Ti de 11 GB**, **RTX 3060 Ti de 8 GB**), o processamento de imagens Sentinel-2 via SAMGeo (ViT-H) e vetorização planar exige considerável consumo computacional.

O pacote inclui um **Monitor de Benchmark em Segundo Plano (`HardwareBenchmarkMonitor`)** que roda de forma assíncrona sem onerar a CPU e coleta a cada 1 segundo:
- **VRAM de Vídeo:** Memória alocada e reservada pelo PyTorch CUDA (Pico e Média).
- **RAM do Sistema:** Uso percentual e memória física ocupada pelo processo Python (RSS).
- **CPU:** Percentual de uso global e por core da CPU.
- **GPU (Compute Load & Temp):** Carga dos núcleos CUDA (%) e temperatura operacional (°C) consultadas via drivers NVIDIA.
- **Duração por Fase:** Tempo exato gasto em cada etapa do pipeline:
  1. *Resolução da ROI e Perímetro CAR*
  2. *Download e Composição Sentinel-2*
  3. *Inferência SAMGeo (ViT-H)*
  4. *Vetorização Planar e Cercas Retilíneas*
  5. *Segmentação Pura no Limite do CAR*
  6. *Classificação Zonal MapBiomas (10m/30m)*
  7. *Exportação e Validação Topológica*

### Arquivos de Relatório Gerados
Ao fim do processamento de cada fazenda, são gravados automaticamente na pasta do imóvel:
1. `relatorio_benchmark_hardware.txt`: Relatório tabular legível e formatado para auditoria técnica.
2. `relatorio_benchmark_hardware.json`: JSON estruturado com métricas brutas para gráficos ou dashboards.
3. `relatorio_benchmark_hardware_lote.txt`: Relatório consolidado em execuções de lote (`run_batch_analysis_lapig`).

### 🧹 Gerenciamento e Esvaziamento Contínuo de VRAM (Zero VRAM Leak)
Para evitar acúmulo de memória de vídeo e estouros (*CUDA Out of Memory*) durante o processamento de múltiplos imóveis/parcelas (lote ou execuções sequenciais no Jupyter):
- **Descarregamento Forçado do Modelo:** O objeto `SamGeo` e seus tensores internos (`sam.sam`, `sam.mask_generator`, `sam.predictor`) são explicitamente desvinculados e deletados via bloco `try ... finally` logo após a geração da máscara raster `masks.tif`.
- **Coleta de Lixo & Purga de Cache CUDA:** A função `clear_gpu_memory(verbose=True)` executa a coleta de lixo Python (`gc.collect()`), o esvaziamento do cache reservado pelo PyTorch (`torch.cuda.empty_cache()`) e coleta blocos IPC compartilhados (`torch.cuda.ipc_collect()`).
- **Garantia por Iteração:** O processamento em lote (`run_batch_analysis_lapig`) executa o esvaziamento no início e ao término compulsório de cada parcela, garantindo que o próximo imóvel inicie com 0 MB de VRAM residual.
- **Uso Manual Disponível:** O desenvolvedor pode invocar `from lapig_parcel_segmentation import clear_gpu_memory; clear_gpu_memory()` a qualquer momento em seus próprios scripts e notebooks.

> **💡 Dica sobre o `nvtop`:**  
> O utilitário `nvtop` é uma interface de terminal interativa (TUI baseada em *ncurses*). Por exigir controle direto de terminal (TTY), ele não pode ser renderizado dentro de células do Jupyter Notebook. Para visualizá-lo em tempo real, abra um terminal separado do sistema (ex: PowerShell: `wsl nvtop` ou no terminal do WSL: `nvtop`). Já o relatório emitido pelo nosso pipeline funciona diretamente em qualquer ambiente (Jupyter, scripts ou servidores headless).

---

## 🌾 3. Segmentação Pura vs. Filtro de Pastagem

Em nível de propriedade rural, a classificação de pastagem de mapas em escala nacional (como o MapBiomas) pode apresentar discordâncias pontuais (ex: pastagens degradadas classificadas como vegetação secundária, ou áreas de pousio classificadas como lavoura temporária).

Para resolver essa limitação, o pipeline gera **duas camadas vetoriais complementares**:

| Camada | Arquivo Gerado | Descrição e Aplicação |
| :--- | :--- | :--- |
| **Segmentação Pura** | `piquetes_segmentacao_pura_{ID}.shp` | **Todos os piquetes e talhões físicos** delimitados pelas cercas e divisas reais da fazenda dentro do CAR, sem nenhum descarte temático. Inclui colunas de auditoria: `pasto_cov`, `ndvi_med`, `is_pasto` (1 ou 0) e `classe` ('Pastagem' ou 'Não Pastagem'). Ideal para planejamento agronômico e gestão de pastoreio rotacionado. |
| **Piquetes de Pasto** | `piquetes_finais_cercas_retas_{ID}.shp` | Subconjunto validado pela dominância zonal do MapBiomas (`pasto_cov >= 0.35`). |
| **Auditoria Não-Pasto** | `nao_pastagem_auditoria.shp` | Polígonos de matas ciliares, APPs, reservas legais e infraestruturas excluídos do filtro de pastagem. |

---

## 📁 4. Pasta de Saída Pré-definida (Organização por ID)

O pipeline organiza os resultados criando uma pasta dedicada por identificador sanitizado do imóvel:
```text
results/{ID_DO_IMOVEL}/
```
*Exemplo:* `results/GO-5203500-366A293867A24EF5A446DA91E23F8548/`

**Arquivos contidos na pasta de cada propriedade:**
* `s2_ndvi_rgb.tif`: Mosaico multi-temporal Sentinel-2 (NDVI max, median, min).
* `masks.tif`: Raster com as máscaras segmentadas pelo SAMGeo.
* `temp_mb_mask_10m.tif`: Máscara binária da Classe 15 do MapBiomas exportada do GEE.
* `piquetes_segmentacao_pura_{ID}.shp`: **Segmentação pura de todos os talhões no CAR.**
* `piquetes_finais_cercas_retas_{ID}.shp`: **Shapefile oficial dos piquetes de pastagem.**
* `nao_pastagem_auditoria.shp`: Áreas de vegetação nativa e não-pastagem para auditoria.
* `relatorio_benchmark_hardware.txt`: Relatório técnico de desempenho e uso de componentes.
* `relatorio_benchmark_hardware.json`: Dados brutos do benchmark de hardware.

---

## 🎯 5. Modalidades de Entrada e Coluna de ID (`id_column`)

Você pode passar como `input_target`:
* **Arquivo Vetorial Local:** `.shp`, `.gpkg`, `.fgb`, `.parquet`, `.geojson`.
* **Asset ou Objeto Earth Engine:** `'projects/...'` ou `ee.FeatureCollection`.
* **Coordenada Geográfica:** Tupla `(-15.747, -50.894)` com busca automática nas bases estaduais do CAR (`CAR_GO`, `CAR_MS`, `CAR_MT`).

---

## 🚀 6. Exemplos de Uso

### Exemplo 1: Execução em Python / JupyterLab (com Segmentação Pura e Benchmark)
```python
import ee
from lapig_parcel_segmentation import run_analysis_lapig

ee.Initialize(project='ee-vieiramesquita')

# Retorna tanto a camada de pastagem quanto a segmentação pura:
gdf_pasto, shp_pasto, gdf_puro, shp_puro = run_analysis_lapig(
    input_target="car_goias_area_imovel_epsg4674_hottest.gpkg",
    id_column="cod_imovel",
    id_value="GO-5203500-366A293867A24EF5A446DA91E23F8548",
    year=2025,
    mapbiomas_collection='10m',
    min_area_ha=2.0,
    straight_tol=22.0,
    return_pure=True,          # Retorna tupla de 4 valores (pasto + pura)
    enable_benchmark=True      # Ativa monitoramento de hardware
)

print(f"Piquetes de Pastagem: {shp_pasto} ({len(gdf_pasto)} piquetes)")
print(f"Segmentação Pura:     {shp_puro} ({len(gdf_puro)} talhões)")
```

### Exemplo 2: Visualizar Relatório de Benchmark no Notebook
```python
import os

report_file = os.path.join(os.path.dirname(shp_pasto), "relatorio_benchmark_hardware.txt")
if os.path.exists(report_file):
    with open(report_file, 'r', encoding='utf-8') as f:
        print(f.read())
```

### Exemplo 3: Linha de Comando (CLI)
```bash
# 1. Executar para um imóvel específico (gera automaticamente ambas as camadas e benchmark):
python lapig_parcel_segmentation/example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --id-val "GO-5203500-366A293867A24EF5A446DA91E23F8548"

# 2. Executar desativando o filtro de pastagem (apenas segmentação pura):
python lapig_parcel_segmentation/example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --id-val "GO-5203500-366A293867A24EF5A446DA91E23F8548" --no-pasture-filter

# 3. Executar em LOTE para todos os imóveis da base:
python lapig_parcel_segmentation/example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --all

# 4. Listar os códigos de imóveis disponíveis no arquivo:
python lapig_parcel_segmentation/example_run.py --input "car_goias_area_imovel_epsg4674_hottest.gpkg" --id-col "cod_imovel" --list-ids
```

### Exemplo 4: Processamento em Lote via Python (`run_batch_analysis_lapig`)
```python
import ee
from lapig_parcel_segmentation import run_batch_analysis_lapig

ee.Initialize(project='ee-vieiramesquita')

# Executa sequencialmente para todos os imóveis da base,
# salva cada um na sua pasta isolada (results/{cod_imovel}/),
# compila os dois shapefiles consolidados e emite o relatório de hardware:
gdfs_sucesso, shp_consolidado = run_batch_analysis_lapig(
    input_target="car_goias_area_imovel_epsg4674_hottest.gpkg",
    id_column="cod_imovel",
    year=2025,
    mapbiomas_collection='10m',
    consolidate_shp=True,
    enable_benchmark=True
)

# Shapefiles gerados na raiz:
# - results/piquetes_todos_imoveis_consolidados.shp (Pastagem)
# - results/piquetes_todos_imoveis_segmentacao_pura.shp (Segmentação Pura)
# - results/relatorio_benchmark_hardware_lote.txt (Benchmark do Lote)
```

---

## 📊 7. Parâmetros da Função `run_analysis_lapig`

| Parâmetro | Tipo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `input_target` / `roi` | `str`, `tuple`, `ee.FeatureCollection` | *Obrigatório* | Arquivo local (.shp, .fgb, .gpkg, .parquet), asset EE ou tupla (lat, lon). |
| `id_column` | `str` | `None` | Nome da coluna de ID no arquivo vetorial (ex: `'cod_imovel'`, `'id'`). |
| `id_value` | `str` / `int` | `None` | Valor específico de ID. Use `'all'` ou `'todos'` para rodar lote completo. |
| `folder_name` | `str` | `None` | Se `None` ou `'auto'`, cria pasta pré-definida: `results/{ID_IMOVEL}/`. |
| `output_base_dir` | `str` | `"results"` | Diretório raiz para as pastas de resultados. |
| `year` | `int` | `2025` | Ano base para S2 e MapBiomas (até 2025). |
| `start_month` | `int` | `3` | Mês inicial do período seco (maior contraste espectral). |
| `end_month` | `int` | `10` | Mês final do período seco. |
| `buffer_size` | `int` | `500` | Buffer em metros ao redor do imóvel para evitar cortes de borda. |
| `min_area_ha` | `float` | `2.0` | Área mínima em hectares. Polígonos menores são dissolvidos no vizinho. |
| `straight_tol` | `float` | `22.0` | Tolerância Hausdorff (m) para considerar uma divisa como cerca reta. |
| `min_pasture_coverage` | `float` | `0.35` | Fração mínima de pixels de pasto (0 a 1) para validar o piquete. |
| `mapbiomas_collection` | `str` | `'10m'` | Coleção do MapBiomas: `'10m'` (Col 4) ou `'30m'` (Col 11). |
| `skip_sam_if_exists` | `bool` | `True` | Pula a inferência do SAMGeo se `masks.tif` já existir na pasta. |
| `return_pure` | `bool` | `False` | Se `True`, retorna tupla com `(gdf_pasto, shp_pasto, gdf_puro, shp_puro)`. |
| `enable_benchmark` | `bool` | `True` | Ativa o monitoramento assíncrono de hardware e emissão de relatório. |

---
*LAPIG - Laboratório de Processamento de Imagens e Geoprocessamento / UFG*
