---
name: project-decision-logger
description: >-
  Extract, log, and maintain key architectural decisions, terminology standards,
  hard constraints, and project rules into a persistent Markdown file (PROJECT_DECISIONS.md)
  in the current repository. Use whenever the user defines or corrects terminology, establishes
  architectural patterns, specifies paper/code constraints, asks to log decisions, or when
  monitoring conversation progress and key project agreements.
---

# Project Decision Logger Skill

Esta skill instrui o agente a monitorar ativamente a conversa com o usuário, extrair acordos técnicos, terminologias obrigatórias, restrições rígidas de projeto e decisões arquiteturais, persistindo-as de forma cumulativa e organizada no arquivo:
👉 `PROJECT_DECISIONS.md` (na raiz do repositório).

---

## 1. Quando Acionar esta Skill

O agente deve acionar esta skill e atualizar o arquivo persistente sempre que:
1. **O usuário estabelecer ou corrigir uma terminologia:** Ex: "Não use a palavra Composto e sim Composite", "Seria melhor usar o termo estanque".
2. **Houver definição de limites e restrições estritas:** Ex: "O limite estrito de 4 páginas para o artigo SBSR", "Regra de 0% de sobreposição entre parcelas".
3. **Decisões arquiteturais ou correções estruturais forem tomadas:** Ex: "Remover a dimensão Z (altitude 3D) de KML/KMZ para compatibilidade com o Google Earth Engine", "Corte por bisseção perpendicular para sementes negativas".
4. **Definições de infraestrutura e ambiente:** Ex: Porta padrão do servidor web (`8088`), projeto GEE (`ee-vieiramesquita`), modelo SAM ViT-H na GPU.
5. **Solicitação direta do usuário:** "Salve nossas decisões", "Atualize as regras do projeto", "Registre o que combinamos".

---

## 2. Categorias Obrigatórias de Extração

Ao analisar as interações recentes (ou os registros em `transcript.jsonl`), o agente deve classificar cada decisão em uma das seções:

### A. 🔤 Padrões de Terminologia & Vocabulário Obrigatório
- Termos a evitar versus termos mandatórios.
- Escopo do objeto de estudo (ex.: delimitação de *field boundaries* / parcelas agrícolas e rurais, e não apenas pastagem).
- Tratamento de termos em inglês vs português em artigos e código.

### B. ⚖️ Restrições Estritas & Limites de Publicação / Negócio
- Limites físicos de páginas (ex.: 4 páginas no SBSR — desclassificação imediata com 5 páginas).
- Regras topológicas inegociáveis (0% de sobreposição / *watertight planar graph*).
- Restrições de tempo de execução, bibliotecas ou dependências.

### C. 🏛️ Decisões de Arquitetura & Implementação Técnica
- Algoritmos e formulações matemáticas implementadas (ex.: bisseção perpendicular, suavização Hausdorff, regularização de grafo planar).
- Particularidades de formatos de entrada (ex.: tratamento de KMZ como zip com KML interno, sanitização de geometrias).
- Endpoints de API e contratos de dados (JSON/GeoJSON).

### D. 🌐 Parâmetros de Ambiente & Infraestrutura
- Portas de rede locais (ex.: `:8088`).
- Credenciais e IDs de projetos de nuvem (ex.: `ee-vieiramesquita`).
- Alocação de hardware (GPU RTX 2080 Ti, ViT-H).

### E. 📜 Registro Cronológico de Decisões (Audit Log)
- Tabela com data/hora, autor da decisão (usuário/agente), categoria, resumo da instrução e arquivos impactados.

---

## 3. Procedimento de Execução Passo a Passo

### Passo 1: Inspecionar o Histórico Recente
Caso necessário, execute o script auxiliar da skill para auditar os pedidos recentes do usuário a partir dos logs de transcrição do Antigravity:
```bash
python .agents/skills/project-decision-logger/scripts/extract_decisions.py
```

### Passo 2: Ler o `PROJECT_DECISIONS.md` Atual
Antes de editar, leia o arquivo [PROJECT_DECISIONS.md](file:///d:/samGeoPasture/Vectorization_Pro/PROJECT_DECISIONS.md) para:
- Evitar duplicações.
- Manter o histórico existente intacto.
- Identificar se a nova decisão complementa ou altera uma regra anterior.

### Passo 3: Atualizar o Arquivo Persistente
Utilize a ferramenta `replace_file_content` ou `write_to_file` para incorporar a nova decisão na categoria apropriada e adicionar uma linha correspondente no **Registro Cronológico (Audit Log)**.

### Passo 4: Notificar o Usuário Concisamente
Informe brevemente ao usuário que a regra/decisão foi consolidada e referencie o link direto para o arquivo [PROJECT_DECISIONS.md](file:///d:/samGeoPasture/Vectorization_Pro/PROJECT_DECISIONS.md).

---

## 4. Estrutura Padrão do Arquivo

O arquivo deve seguir fielmente o modelo disponível em:
[PROJECT_DECISIONS_TEMPLATE.md](./resources/PROJECT_DECISIONS_TEMPLATE.md)
