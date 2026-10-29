# 📋 Registro de Decisões do Projeto & Regras de Engenharia (PROJECT_DECISIONS.md)

Este documento registra de forma persistente e cumulativa todas as decisões de projeto, diretrizes de publicação, convenções terminológicas, restrições arquiteturais e regras de negócio estabelecidas ao longo do desenvolvimento do **ParcelAnything** e do artigo SBSR.

---

## 1. 🔤 Padrões de Terminologia & Vocabulário Obrigatório

| Termo Proibido / Evitar | Termo Obrigatório / Adotado | Contexto & Justificativa |
| :--- | :--- | :--- |
| **Composto** / **Compostos** | **Composite** / **Composites** | Convenção padrão em sensoriamento remoto e literatura internacional (ex.: *Composite Temporal Sentinel-2 (CS+)*, *composites anuais Sentinel-2*). Aplicar no artigo e na documentação. |
| **Hermética** / **Hermeticidade** | **Estanque** / **Estanqueidade topológica** | Termo técnico geoespacial rigoroso e consagrado em português para traduzir *watertight planar graph* (ausência total de lacunas/gaps e sobreposições/overlaps). |
| **Piquete** / **Piquetes** (como polígono) | **Parcela** / **Parcelas** | O polígono ou unidade espacial delineada deve ser denominado estritamente **"parcela"** / **"parcelas"** (*parcels* em inglês), unificando a terminologia do projeto e eliminando a restrição exclusiva a pecuária. |
| **Apenas "Pastagem"** | **Parcelas rurais / Field boundaries** | O escopo do artigo SBSR e do método abrange o delineamento de limites cadastrais gerais de propriedades rurais (parcelas agrícolas, talhões, áreas de pousio e reservas), e não apenas pastagens. |
| *Omissão de interação* | **Human-in-the-loop** | Manter explicitamente no Resumo e Abstract como diferencial do modo interativo frente à segmentação puramente automática. |

---

## 2. ⚖️ Restrições Estritas & Limites de Publicação / Negócio

*Diretrizes e limites que **NUNCA** podem ser violados.*

- **Limite de Páginas (SBSR):** O artigo deve ter **estritamente 4 páginas**. Artigos com 5 páginas são desqualificados sumariamente pelas normas do Simpósio Brasileiro de Sensoriamento Remoto.
- **Tamanho de Resumo e Abstract:** Exatamente entre **100 e 150 palavras** cada.
- **Topologia de Parcelas:** **0% de sobreposição** entre polígonos adjacentes. Nenhuma parcela/talhão novo pode invadir a área de uma parcela já salva no banco/sessão.
- **Formatação do Artigo:** Formato A4, 2 colunas, Times New Roman 10pt, sem numeração de páginas (`\pagestyle{empty}`), referências no padrão numérico IEEE/unsrt (`[1]`, `[2]`).

---

## 3. 🏛️ Decisões de Arquitetura & Implementação Técnica

### 3.1. Pipeline e Modelos (ParcelAnything Auto & Interactive)
- **Modelo de Segmentação:** HQ-SAM com backbone ViT-H (`vit_h`), executado com aceleração de GPU dedicada (NVIDIA RTX 2080 Ti).
- **Composição Espectral:** Composição temporal RGB formada por NDVI máximo (R), NDVI mediano (G) e NDVI mínimo (B) com máscara *Google Cloud Score Plus* (CS+ cdf $\ge$ 0.65).
- **Bisseção Perpendicular (Negative Seeds):** O refinamento interativo por sementes negativas calcula a reta bissetriz perpendicular ao segmento entre a semente positiva e a negativa, garantindo o corte limpo da partição em disputa (Equação 1 do artigo).
- **Regularização Topológica (Grafo Planar Estanque):**
  - Implementada em `regularize_paddocks_topology(straight_tol=18.0)`.
  - Executa intersecção de arestas via `shapely.node`, fusão de linhas (`linemerge`), simplificação chordal por distância de Hausdorff e reconstrução de polígonos limpos via `polygonize`.
  - Disponível como botão na interface web e acionada automaticamente na exportação de GeoJSON.

### 3.2. Formatos de Entrada & Tratamento de Coordenadas (KML / KMZ)
- **Formatos Suportados:** `.kml`, `.kmz`, `.gpkg`, `.geojson`, `.zip` (shapefile) e `.shp`.
- **Tratamento de KMZ:** Descompactação dinâmica do arquivo comprimido em memória/tempdir, com localização automática do `doc.kml` interno.
- **Sanitização de Coordenadas 3D (Altitude):**
  - KMLs/KMZs de GPS e Google Earth trazem coordenadas 3D `(x, y, z)`.
  - **Regra obrigatória:** Toda geometria importada deve ser convertida para 2D via `shapely.ops.transform(lambda x, y, *args: (x, y), geom)` antes de ser enviada ao Google Earth Engine (`s2_mosaic`), evitando a falha `Invalid GeoJSON geometry`.

---

## 4. 🌐 Parâmetros de Ambiente & Infraestrutura

| Parâmetro | Valor Padrão | Descrição |
| :--- | :--- | :--- |
| **Porta da UI Interativa** | `8088` | Porta padrão para inicialização via `run_interactive_app.py --port 8088` |
| **Host de Acesso** | `http://localhost:8088` ou `http://127.0.0.1:8088` | Endpoint da interface gráfica interativa do HQ-SAM |
| **Ambiente de Execução** | WSL 2 (Ubuntu / Python 3.13) | Ambiente onde estão instalados PyTorch, PyOgrio, GeoPandas e Rasterio |
| **Projeto Google Earth Engine** | `ee-vieiramesquita` | Conta de serviço habilitada para aquisição do catálogo Sentinel-2 CS+ |

---

## 5. 📜 Registro Cronológico de Decisões (Audit Log)

| Data / Timestamp | Categoria | Decisão / Instrução | Impacto no Repositório |
| :--- | :--- | :--- | :--- |
| **2026-10-28** | *Terminologia* | Substituição do termo "hermética" por "estanque" / "estanqueidade topológica" | Ajustes no texto do artigo SBSR (`01_Resumo.tex`, `03_Material_e_metodos.tex`). |
| **2026-10-28** | *Terminologia* | Proibição do termo "composto", adotando "composite" / "composites" | Atualização em todas as seções do LaTeX e legendas de figuras. |
| **2026-10-28** | *Escopo* | Enfatizar "field boundaries" e parcelas rurais gerais, e não apenas pastagens | Revisão de texto em Resumo, Abstract e Introdução do SBSR. |
| **2026-10-28** | *Topologia* | Implementação da regularização cadastral via grafo planar estanque no app interativo | Novo endpoint `/api/paddocks/regularize`, botão na UI e regularização no export GeoJSON. |
| **2026-10-28** | *Interatividade* | Implementação de corte por bisseção perpendicular para sementes negativas | Função `_clip_polygon_by_bisector` em `hq_sam_engine.py` (Eq. 1 do paper). |
| **2026-10-29** | *Formatos / I/O* | Permitir upload de dados vetoriais em `.kml` e `.kmz` | Suporte a KMZ em `app.py`, dropzone no frontend e remoção forçada da coordenada 3D Z para GEE. |
| **2026-10-29** | *Governança* | Criação da skill `project-decision-logger` e documento `PROJECT_DECISIONS.md` | Skill criada em `.agents/skills/project-decision-logger/` e registro persistente centralizado. |
| **2026-10-29** | *Terminologia* | Substituição estrita de "piquete/piquetes" por "parcela/parcelas" para designar os polígonos | Atualização em todos os códigos (Python, JS), UI do aplicativo, documentação técnica e repositório GitHub (mantendo o artigo acadêmico intocado conforme instrução do usuário). |
