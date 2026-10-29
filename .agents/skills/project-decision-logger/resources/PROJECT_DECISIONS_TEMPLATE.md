# 📋 Registro de Decisões do Projeto & Regras de Engenharia (PROJECT_DECISIONS.md)

Este documento registra de forma persistente e cumulativa todas as decisões de projeto, regras de negócio, convenções terminológicas, restrições arquiteturais e acordos estabelecidos durante o desenvolvimento do repositório.

---

## 1. 🔤 Padrões de Terminologia & Vocabulário Obrigatório

| Termo Proibido / Evitar | Termo Obrigatório / Adotado | Contexto & Justificativa |
| :--- | :--- | :--- |
| *Exemplo: Composto* | **Composite** (ou *composites*) | Convenção em sensoriamento remoto e literatura internacional (Sentinel-2 Composite). |
| *Exemplo: Hermética* | **Estanque** / **Estanqueidade topológica** | Termo técnico rigoroso para planar graph / sem frestas ou sobreposições. |

---

## 2. ⚖️ Restrições Estritas & Limites de Publicação / Negócio

*Diretrizes e limites que **NUNCA** podem ser violados.*

- **Limite de Páginas (SBSR):** Estritamente **4 páginas**. Não ultrapassar 4 páginas sob hipótese alguma.
- **Topologia de Parcelas:** **0% de sobreposição** entre polígonos adjacentes.

---

## 3. 🏛️ Decisões de Arquitetura & Implementação Técnica

### 3.1. Pipeline e Modelos
- Decisões sobre modelos (ViT-H SAMGeo, HQ-SAM), bandas espectrais, etc.

### 3.2. Formatos de Dados & Tratamentos Especiais
- Regras para shapefiles, GPKG, GeoJSON, KML e KMZ (ex.: remoção forçada da dimensão Z/altitude antes do envio ao Google Earth Engine).

---

## 4. 🌐 Parâmetros de Ambiente & Infraestrutura

| Parâmetro | Valor Padrão | Descrição |
| :--- | :--- | :--- |
| **Porta da UI Interativa** | `8088` | Porta padrão para `run_interactive_app.py` |
| **GEE Project** | `ee-vieiramesquita` | Conta de serviço do Google Earth Engine |

---

## 5. 📜 Registro Cronológico de Decisões (Audit Log)

| Data / Etapa | Categoria | Decisão / Instrução | Impacto no Código / Documento |
| :--- | :--- | :--- | :--- |
| *YYYY-MM-DD* | *Terminologia* | *...* | *...* |
