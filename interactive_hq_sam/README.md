# 🖱️ ParcelAnything: Interactive Mode (Segmentação Interativa Human-in-the-Loop)
**LAPIG - Laboratório de Processamento de Imagens e Geoprocessamento (UFG)**

O **ParcelAnything Interactive** é projetado para **operações assistidas por inteligência visual**, onde um operador humano define ou refina limites de parcelas e divisórias de pastagem através de cliques no mapa interativo.

---

## 🎯 Principais Funcionalidades

1. **Amostragem Inteligente por Pontos (HQ-SAM ViT-H)**:
   - **Ponto Positivo (`+`)**: Define o interior da pastagem a ser segmentada. Suporta múltiplos pontos positivos para unir áreas contíguas.
   - **Ponto Negativo (`−`)**: Recorta áreas indesejadas (represas, sedes, matas), descarta parcelas adjacentes ou divide pastos contíguos por bissetriz perpendicular.
2. **Confirmação Obrigatória & Refinamento**:
   - Cliques geram uma proposta visual em contorno tracejado neon (ciano).
   - O polígono só é salvo após validação explícita do usuário no dock inferior (`Confirmar Parcela`).
   - Permite selecionar qualquer parcela já salva para editar ou refinar seus limites com novos pontos.
3. **Entradas Flexíveis**:
   - **Coordenadas Geográficas (Lat/Lon)**: Busca em tempo real o mosaico Sentinel-2 harmonizado no Google Earth Engine.
   - **Código CAR**: Localiza a geometria do imóvel rural e faz o download automático da cobertura espectral.
   - **Upload de Vetor**: Suporte direto para arquivos `.gpkg`, `.zip` (Shapefile), `.kml`, `.kmz` ou `.geojson`.
   - **Fazendas Pré-carregadas**: Demonstrações instantâneas de imóveis em Goiás com mosaicos normalizados.
4. **Métricas Cadastrais**:
   - Cálculo automático de área em hectares (`ha`), perímetro e pontuação de confiança da segmentação.
   - Exportação cadastral consolidada em formato **GeoJSON** compatível com QGIS e ArcGIS.

---

## 🚀 Como Executar

### Pré-requisitos
- Ambiente Python com PyTorch e suporte CUDA (recomendado GPU com 8GB+ VRAM, ex: RTX 2080 Ti ou RTX 3060 Ti).
- Pesos do modelo SAM ViT-H presentes em `models/sam_vit_h_4b8939.pth`.

### Inicialização do Servidor Local
No terminal (ou via WSL):
```bash
python run_interactive_app.py --port 8088
```

Ou especificando o host para acesso na rede local:
```bash
python run_interactive_app.py --host 0.0.0.0 --port 8088
```

Acesse no navegador:
👉 **`http://localhost:8088/`**

---

## 📂 Estrutura de Arquivos

```
interactive_hq_sam/
├── backend/
│   ├── app.py              # API FastAPI (endpoints /api/predict, /api/roi, /api/paddock)
│   └── hq_sam_engine.py    # Motor PyTorch com HQ-SAM ViT-H e integração GEE
├── frontend/
│   ├── index.html          # Interface Leaflet responsiva com dock de refinamento
│   ├── style.css           # Design system LAPIG (Dark glassmorphism neon)
│   └── app.js              # Lógica de controle de mapa, eventos e chamadas à API
├── cache/                  # Overlays espectrais e rasters pré-processados
└── run_interactive_app.py  # Script principal de execução
```
