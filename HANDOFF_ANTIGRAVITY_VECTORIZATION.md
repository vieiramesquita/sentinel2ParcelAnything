# GUIA DE TRANSIÇÃO E ESPECIFICAÇÃO TÉCNICA: VETORIZAÇÃO TOPOLÓGICA DE TALHÕES E PIQUETES (LAPIG)
> **Destinatário:** Próxima instância do Antigravity / Desenvolvedor  
> **Objetivo:** Reproduzir, executar e escalar o pipeline de delimitação e vetorização de piquetes de pastagem e talhões agrícolas anuais com **Zero Sobreposições (Topologia Watertight)** e **Geometria Retilínea Cadastral**.

---

## 1. Visão Geral do Problema e Resultados Já Validados

### O Desafio
* Delimitar piquetes de pastagem e talhões agrícolas a partir de imagens de satélite (Sentinel-2) e máscaras de Deep Learning.
* **Problema das abordagens convencionais (GDAL / SAMGeo puro):**
  1. O SAMGeo hiper-segmenta os pastos (166 a 324 máscaras para áreas que têm 40 a 109 piquetes reais).
  2. O GDAL polygonize gera serrilhado de pixel (degraus de 10m).
  3. A simplificação independente de polígonos destrói a topologia (cria centenas de sobreposições e frestas).

### A Solução Validada
Implementamos um pipeline em **Grafo Planar com Fusão Espectral e Simplificação de Arestas Compartilhadas** que alcançou paridade com o desenho manual:

| Caso de Teste | Gabarito Manual | SAMGeo Bruto | Vetorização Final (`OUTPUT/`) | Sobreposições Topológicas | Média Vértices/Polígono |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Goiás (`GO`)** | 40 / 41 polígonos | 166 máscaras | **40 polígonos** | **0 (Zero)** | 13.8 (retas de cerca) |
| **Mato Grosso do Sul (`MS`)** | 109 / 115 polígonos | 324 máscaras | **113 polígonos** | **0 (Zero)** | 20.3 (retas de cerca) |

---

## 2. Dados de Referência e Imóveis do CAR

### Acesso Local ao CAR (Muito mais rápido que requisições ao GEE)
* **Arquivo Parquet Local:** `E:\TrabalhosLAPIG\PastoLegal\Time2Graze\CAR_BRASIL_AREA_IMOVEL_EPSG4674.parquet`
* **Consulta Instantânea com DuckDB:**
```python
import duckdb
con = duckdb.connect()
path = r'E:\TrabalhosLAPIG\PastoLegal\Time2Graze\CAR_BRASIL_AREA_IMOVEL_EPSG4674.parquet'

query = f"""
SELECT cod_imovel, municipio, cod_estado, num_area, geometry 
FROM read_parquet('{path}') 
WHERE cod_imovel IN (
    'GO-5211008-46C08C6235D74D36BE07B5384A17B156', 
    'MS-5006309-8DC93A48C2424FD7B3EDF235F8B531E1'
)
"""
imoveis = con.execute(query).fetchall()
```

### Propriedades Testadas
1. **Goiás (GO):**
   * `cod_imovel`: `GO-5211008-46C08C6235D74D36BE07B5384A17B156`
   * Município: Itapirapuã - GO
   * Área: **193.59 ha**
   * Arquivos na pasta: `INPUT/masks_GO.tif`, `INPUT/s2_ndvi_rgb_GO.tif`, `HAND_VECT/vect_goias.shp`
2. **Mato Grosso do Sul (MS):**
   * `cod_imovel`: `MS-5006309-8DC93A48C2424FD7B3EDF235F8B531E1`
   * Município: Paranaíba - MS
   * Área: **1.085,22 ha**
   * Arquivos na pasta: `INPUT/masks_MS.tif`, `INPUT/s2_ndvi_rgb_MS.tif`, `HAND_VECT/vect_ms.shp`

---

## 3. O Composto Anual Sentinel-2 Otimizado (Google Earth Engine)

### Por que a abordagem anterior (`[NDVI_max, NDVI_median, NDVI_min]`) falhou?
O notebook anterior (`parcelSegmentation_SAMGeo_LAPIG.ipynb`) usava apenas 3 estatísticas de NDVI. Isso causava dois problemas:
1. O NDVI não enxerga solo exposto, cercas de arame e carreadores (que refletem no SWIR e Vermelho).
2. O SAMGeo segmentava pequenas variações de umidade/vigor do capim dentro do mesmo pasto.

### A Fórmula do Composto Anual Recomendada
Para mapeamento anual de pastagens e talhões agrícolas:
1. **Banda 1 - SWIR1 (B11) Mediana Anual:** Altíssimo contraste para cercas, carreadores e aceiros.
2. **Banda 2 - Vermelho (B4) Mediana Anual:** Contraste visual e solo.
3. **Banda 3 - NIR (B8) Mediana Anual:** Biomassa média do período.
4. **Banda 4 - Amplitude do NDVI ($\text{NDVI}_{\max} - \text{NDVI}_{\min}$):** O segredo do pastejo rotacionado. No piquete, a amplitude é alta (o capim é comido e cresce). Na cerca e na estrada, a amplitude é estável.
5. **Máscara MapBiomas 10m Coleção 2/3 (Classe 15 = Pastagem):** Exclui rios, APPs e matas ciliares antes da inferência.

### Código GEE para Extração Anual
```python
import ee
ee.Initialize(project='ee-vieiramesquita')

def get_annual_paddock_composite(roi_geometry, year):
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    
    def mask_clouds(img):
        # cs_cdf >= 0.65 filtra nuvens e sombras com precisão
        mask = img.select('cs_cdf').gte(0.65)
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return img.addBands(ndvi).updateMask(mask)
        
    col = (s2.filterBounds(roi_geometry)
             .filterDate(f'{year}-01-01', f'{year}-12-31')
             .linkCollection(csPlus, ['cs_cdf'])
             .map(mask_clouds))
             
    # Medianas espectrais
    b11_med = col.select('B11').median()
    b8_med  = col.select('B8').median()
    b4_med  = col.select('B4').median()
    b3_med  = col.select('B3').median()
    b2_med  = col.select('B2').median()
    
    # Dinâmica anual de pastejo (Amplitude NDVI)
    ndvi_col = col.select('NDVI')
    ndvi_amp = ndvi_col.reduce(ee.Reducer.max()).subtract(ndvi_col.reduce(ee.Reducer.min())).rename('NDVI_AMP')
    
    # Máscara MapBiomas 10m (Pastagem = 15)
    mb = (ee.Image("projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1")
          .select(f'classification_{year}'))
    pasture_mask = mb.eq(15)
    
    composite = ee.Image([b4_med, b3_med, b2_med, b8_med, b11_med, ndvi_amp]).clip(roi_geometry).updateMask(pasture_mask)
    return composite
```

---

## 4. Modelos Abertos de Deep Learning (ResUNet / FTW / Trazo)

As principais iniciativas internacionais (Varda/DigiFarm, AI4Boundaries, OneSoil) utilizam modelos que predizem **Extent (Miolo)** e **Boundary (Cercas)**.

### Modelo Aberto de Referência: `torchgeo/fields-of-the-world` (FTW)
* **Repositório Hugging Face:** [`torchgeo/fields-of-the-world`](https://huggingface.co/torchgeo/fields-of-the-world)
* **Pesos Pré-treinados:** `ftw-3class-full_unet-efficientnetb3_rgbnir_1ba4e1bd.pth`
* **Entrada:** Sentinel-2 4 bandas (`B4`, `B3`, `B2`, `B8` - RGB + NIR).
* **Classes de Saída:**
  * `0`: Fundo (Não-agrícola)
  * `1`: **Interior do Piquete (*Extent*)**
  * `2`: **Cercas / Divisórias (*Boundary*)**
* **Como carregar no PyTorch:**
```python
import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    encoder_name="efficientnet-b3",
    encoder_weights=None,
    in_channels=4,
    classes=3
)
weights = torch.load("ftw-3class-full_unet-efficientnetb3_rgbnir_1ba4e1bd.pth", map_location="cuda")
model.load_state_dict(weights)
model.eval()
```

---

## 5. Algoritmo de Poligonização Topológica (Watertight Planar Graph)

Este é o algoritmo já implementado e funcional no arquivo [`vectorize_parcels.py`](file:///e:/TrabalhosLAPIG/Teste_Vectorization/vectorize_parcels.py).

### As Etapas Matemáticas
1. **Fusão Espectral (Adjacency Graph):**
   * Mede o gradiente do Sentinel-2 ao longo de cada borda compartilhada entre duas máscaras vizinhas.
   * Se o gradiente for baixo (sem cerca/estrada real), une as máscaras via *Union-Find*.
2. **Suavização Morfológica:**
   * Aplica filtro mediano 5x5 para eliminar o serrilhado de pixel (resolução 10m).
3. **Extração de Grafo Planar:**
   * Extrai o contorno das regiões e divide nos nós de interseção: `shapely.node(unary_union(boundaries))`.
   * Une os segmentos lineares contínuos entre cruzamentos: `linemerge`.
4. **Simplificação Ramer-Douglas-Peucker em Linhas Compartilhadas:**
   * Simplifica cada aresta da cerca com tolerância de 20 a 25 metros.
   * **Por que garante Zero Sobreposição?** Como a simplificação ocorre na aresta compartilhada antes de reconstituir as faces (`shapely.ops.polygonize`), dois piquetes vizinhos compartilham **exatamente os mesmos vértices e arestas**.
5. **Filtragem e Exportação:**
   * Remove lascas menores que a área mínima (ex: 1.0 ha).
   * Reprojeta para `EPSG:4326` (WGS84).
   * Salva os shapefiles com campos `id` e `area_ha`.

---

## 6. Como Executar na Nova Máquina

### Dependências Necessárias
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch rasterio geopandas shapely scipy duckdb earthengine-api geemap opencv-python
```

### Executar o Pipeline End-to-End Automatizado
O script `pipeline_end_to_end.py` executa o ciclo completo em ~30 a 60 segundos por fazenda:
```bash
# Executar para Goiás (ano 2024):
python3 pipeline_end_to_end.py --cod_imovel "GO-5211008-46C08C6235D74D36BE07B5384A17B156" --year 2024

# Executar para Mato Grosso do Sul (ano 2024):
python3 pipeline_end_to_end.py --cod_imovel "MS-5006309-8DC93A48C2424FD7B3EDF235F8B531E1" --year 2024 --min_area_ha 1.5
```

### Arquivos Chave no Repositório
* [`pipeline_end_to_end.py`](file:///e:/TrabalhosLAPIG/Teste_Vectorization/pipeline_end_to_end.py): Pipeline completo (DuckDB CAR + GEE Download + Gradientes + Grafo Planar).
* [`vectorize_parcels.py`](file:///e:/TrabalhosLAPIG/Teste_Vectorization/vectorize_parcels.py): Código de produção para vetorização de rasters pré-existentes.
* [`OUTPUT/parcels_goias_auto.shp`](file:///e:/TrabalhosLAPIG/Teste_Vectorization/OUTPUT/parcels_goias_auto.shp): Resultado Goiás calibrado com SAMGeo (40 polígonos, 0 overlaps).
* [`OUTPUT/parcels_ms_auto.shp`](file:///e:/TrabalhosLAPIG/Teste_Vectorization/OUTPUT/parcels_ms_auto.shp): Resultado MS calibrado com SAMGeo (113 polígonos, 0 overlaps).
* [`OUTPUT/parcels_GO_5211008_46C08C6235D74D36BE07B5384A17B156_2024.shp`](file:///e:/TrabalhosLAPIG/Teste_Vectorization/OUTPUT/parcels_GO_5211008_46C08C6235D74D36BE07B5384A17B156_2024.shp): Resultado anual 2024 gerado pelo pipeline end-to-end.
* [`OUTPUT/parcels_MS_5006309_8DC93A48C2424FD7B3EDF235F8B531E1_2024.shp`](file:///e:/TrabalhosLAPIG/Teste_Vectorization/OUTPUT/parcels_MS_5006309_8DC93A48C2424FD7B3EDF235F8B531E1_2024.shp): Resultado anual 2024 gerado pelo pipeline end-to-end.
* [`scratch/final_eval_go.png`](file:///e:/TrabalhosLAPIG/Teste_Vectorization/scratch/final_eval_go.png): Imagem comparativa Goiás (Manual vs Automático).
* [`scratch/final_eval_ms.png`](file:///e:/TrabalhosLAPIG/Teste_Vectorization/scratch/final_eval_ms.png): Imagem comparativa MS (Manual vs Automático).
* [`walkthrough.md`](file:///C:/Users/vinic/.gemini/antigravity-ide/brain/28456e77-c5b2-41a8-8b9d-4025b88a7414/walkthrough.md): Relatório de validação técnica.
