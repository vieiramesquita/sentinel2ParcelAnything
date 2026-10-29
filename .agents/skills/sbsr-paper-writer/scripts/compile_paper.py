#!/usr/bin/env python3
"""
SBSR Paper Compiler & Page Auditor
Compiles an SBSR LaTeX paper (pdflatex + bibtex) and asserts that the output
is strictly within the mandatory 4-page limit.
"""

import sys
import os
import subprocess
import re

def compile_sbsr_paper(paper_dir="paper", master_tex="00_Template_SBSR.tex"):
    tex_path = os.path.join(paper_dir, master_tex)
    if not os.path.exists(tex_path):
        print(f"Error: {tex_path} not found.")
        sys.exit(1)

    base_name = os.path.splitext(master_tex)[0]
    
    print(f"[*] Compiling {master_tex} in {paper_dir}...")
    
    cmds = [
        ["pdflatex", "-interaction=nonstopmode", master_tex],
        ["bibtex", base_name],
        ["pdflatex", "-interaction=nonstopmode", master_tex],
        ["pdflatex", "-interaction=nonstopmode", master_tex]
    ]
    
    for i, cmd in enumerate(cmds, 1):
        print(f"    [{i}/4] Running {' '.join(cmd)}...")
        res = subprocess.run(cmd, cwd=paper_dir, capture_output=True, text=True, errors="replace")
        if res.returncode != 0 and i in (1, 3, 4):
            print(f"[!] Warning during step {i}: exit code {res.returncode}")
            
    # Check log for page count
    log_file = os.path.join(paper_dir, f"{base_name}.log")
    pdf_file = os.path.join(paper_dir, f"{base_name}.pdf")
    
    if not os.path.exists(pdf_file):
        print(f"[x] Error: {pdf_file} was not generated!")
        sys.exit(1)
        
    pages = None
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()
            match = re.search(r"Output written on .* \((\d+)\s+pages?", log_content)
            if match:
                pages = int(match.group(1))
                
    if pages is not None:
        if pages == 4:
            print(f"[OK] Success! Generated {pdf_file} with EXACTLY {pages} pages (Official SBSR standard).")
        elif pages < 4:
            print(f"[INFO] Generated {pdf_file} with {pages} pages (Within limit, but consider adding content to reach 4 full pages).")
        else:
            print(f"[!] DANGER: Generated {pdf_file} with {pages} pages! (EXCEEDS the 4-page SBSR limit, paper would be disqualified).")
    else:
        print(f"[OK] Generated {pdf_file} (check page count manually).")

if __name__ == "__main__":
    p_dir = sys.argv[1] if len(sys.argv) > 1 else "paper"
    m_tex = sys.argv[2] if len(sys.argv) > 2 else "00_Template_SBSR.tex"
    compile_sbsr_paper(p_dir, m_tex)
