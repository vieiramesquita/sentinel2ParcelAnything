---
name: sbsr-paper-writer
description: >-
  Guide and automate the creation, structuring, formatting, and LaTeX compilation
  of scientific papers for the Brazilian Remote Sensing Symposium (Simpósio Brasileiro
  de Sensoriamento Remoto - SBSR). Use when the user asks to write, edit, format,
  structure, or compile an SBSR paper, enforce the strict 4-page limit, manage two-column
  figure placement, or resolve LaTeX/pdflatex/babel compilation errors.
---

# SBSR Paper Writer Skill

This skill provides comprehensive instructions, best practices, and automation guidelines for writing and formatting scientific papers for the **Simpósio Brasileiro de Sensoriamento Remoto (SBSR)** using LaTeX.

---

## 1. Key SBSR Constraints & Specifications

All SBSR submissions must adhere strictly to these physical and editorial constraints:

| Parameter | Specification | Gotcha / Recommendation |
| :--- | :--- | :--- |
| **Page Limit** | **Maximum 4 pages** (including references) | Papers with 5 pages are automatically disqualified. Target **exactly 4 pages** with balanced columns on the final page. |
| **Layout** | Two columns (`twocolumn`, 10pt Times) | A4 paper, margins: Top 25mm, Bottom 20-25mm, Left 19mm, Right 19mm, Column separation 6mm, Column width 86mm. |
| **Language** | Portuguese, English, or Spanish | If written in Portuguese or Spanish, it **must** include an English *Abstract* and *Key words*. |
| **Resumo / Abstract** | **100 to 150 words each** | Do not exceed 150 words. Overly long abstracts displace the Introduction and cause page overflows. |
| **Page Numbering** | Disabled (`\pagestyle{empty}`) | Do not number pages. Numbers are added during proceedings compilation. |
| **Section Titles** | Numbered, uppercase, centered bold | `1. INTRODUÇÃO`, `2. MATERIAL E MÉTODOS`, `3. RESULTADOS`, `4. DISCUSSÃO`, `5. CONCLUSÕES`. |
| **Subsections** | Bold, lowercase (title case), left-aligned | `2.1. Área de Estudo`, `2.2. Dados Orbitais`. |
| **Captions** | 9pt bold, centered | `Figura 1: ...` or `Tabela 1: ...` (above tables, below figures). |
| **References** | IEEE / unsrt numeric (`[1]`, `[2]`) | Numbered in order of appearance in text using BibTeX `unsrt` style. |

---

## 2. Document Architecture & Modular Files

Follow the official SBSR modular structure:

```text
paper/
├── 00_Template_SBSR.tex      # Master document (documentclass, packages, maketitle, inputs)
├── 01_Resumo.tex             # Portuguese Resumo + English Abstract + Keywords
├── 02_Introducao.tex         # 1. Introdução (Context, problem, 3 bottlenecks, objectives)
├── 03_Material_e_metodos.tex # 2. Material e métodos (Study area, sensors, pipeline, math)
├── 04_Resultados.tex         # 3. Resultados (Table 1 metrics, manual benchmark, interactive UI)
├── 05_Discussao.tex          # 4. Discussão (Foundation models vs geometry, hardware, synergy)
├── 06_Conclusoes.tex         # 5. Conclusões, Agradecimentos (LAPIG/UFG, CNPq, FAPEG)
├── 07_SBSR.bib               # BibTeX references
├── consideracoes_extras.tex  # Template instructions (COMMENT OUT in final submission)
├── sbsr.sty                  # SBSR LaTeX style file
└── figuras/                  # Folder containing all image assets (.png, .jpg, .pdf)
```

---

## 3. Page Budgeting Blueprint (To Guarantee Exactly 4 Pages)

To achieve a balanced, publication-grade 4-page paper, follow this space allocation:

### Page 1
- **Top:** Title in uppercase, Authors, Affiliation (LAPIG/IESA/UFG, email addresses).
- **Left Column:** `RESUMO` (100–130 words) + `Palavras-chave`, `ABSTRACT` (100–130 words) + `Key words`.
- **Right Column:** `1. INTRODUÇÃO` (starts with national context and numbered list of 3 major bottlenecks).

### Page 2
- **Left Column:** Conclusion of Introduction (formal presentation of the proposed system) followed by `2. MATERIAL E MÉTODOS` (`2.1 Área de Estudo` e `2.2 Dados Orbitais/Compostos`).
- **Right Column:** `2.3 Pipeline Metodológico` (telemetria, algoritmos de regularização) e `2.4 Modo Interativo`.
- **Critical LaTeX Timing:** If using a full-width figure (`\begin{figure*}[t]`), declare it inside `03_Material_e_metodos.tex` (Page 2). LaTeX will then place it at the top of **Page 3**.

### Page 3
- **Top:** Full-width Figure (`\begin{figure*}[t]`, e.g., 3-stage progression from satellite imagery to deep learning masks and regularized cadastral vectors).
- **Left Column:** `3. RESULTADOS`, Table 1 (computational profiling / accuracy metrics).
- **Right Column:** Benchmark against manual ground truth, interactive capabilities, and transition to Discussion.

### Page 4
- **Left Column:** Single-column Figure (e.g., web interface / prompt-based editing) and `4. DISCUSSÃO` (numbered technical arguments).
- **Right Column:** `5. CONCLUSÕES`, `AGRADECIMENTOS` (funding agencies), and `6. REFERÊNCIAS` (all 6 to 10 references, with both columns balanced using `flushend`).

---

## 4. LaTeX Quirks & Compilation Troubleshooting

### A. Babel Language in Linux / WSL
- In Debian/Ubuntu/WSL without `texlive-lang-portuguese`, standard `\usepackage[brazil]{babel}` fails.
- **Fix:** In `sbsr.sty`, use:
  ```latex
  \RequirePackage[provide=*,main=brazilian,english]{babel}
  ```
  And in `00_Template_SBSR.tex`:
  ```latex
  \documentclass[10pt,a4paper,twocolumn,english,brazilian]{article}
  ```
  And in `01_Resumo.tex`:
  ```latex
  \selectlanguage{brazilian}
  ```

### B. Two-Column Floating Figures (`figure*`)
- In LaTeX, `\begin{figure*}[t]` is **deferred to the top of the next page**.
- If declared on Page 3, it lands on Page 4 and pushes references onto Page 5.
- **Rule:** Always declare `\begin{figure*}[t]` in the source file that is processed while compiling **Page 2** (e.g., inside `03_Material_e_metodos.tex`), ensuring it appears cleanly at the top of **Page 3**.

### C. Single-Column Tables in 86mm Width
- Standard `tabular` easily produces `overfull \hbox` in an 86mm column.
- **Fix:** Use `tabular*` with `\extracolsep{\fill}`:
  ```latex
  \begin{table}[htbp]
  \centering
  \footnotesize
  \caption{Título da tabela.}
  \label{tab:rotulo}
  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lr}
  \toprule
  \textbf{Parâmetro} & \textbf{Valor} \\
  \midrule
  ...
  \bottomrule
  \end{tabular*}
  \end{table}
  ```

### D. Balanced Columns on the Final Page
- The official SBSR template includes `\usepackage[nospread]{flushend}` in `sbsr.sty`.
- Ensure it is loaded to prevent one column from stretching to the bottom while the other stays half-empty.

---

## 5. Compilation Pipeline Commands

Execute compilation via WSL or native TeX Live:

```bash
cd paper/
# 1. First pdflatex pass (generates aux)
pdflatex -interaction=nonstopmode 00_Template_SBSR.tex
# 2. Bibtex pass (resolves citations)
bibtex 00_Template_SBSR
# 3. Second pass (resolves references)
pdflatex -interaction=nonstopmode 00_Template_SBSR.tex
# 4. Final pass (stabilizes flushend and cross-references)
pdflatex -interaction=nonstopmode 00_Template_SBSR.tex
```

Verify exact page count:
```bash
python3 -c "import pypdf; print('Pages:', len(pypdf.PdfReader('00_Template_SBSR.pdf').pages))"
```
Or check log:
```bash
grep -E "Output written on .* \([0-9]+ pages" 00_Template_SBSR.log
```
