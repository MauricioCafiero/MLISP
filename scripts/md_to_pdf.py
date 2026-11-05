"""Markdown -> PDF (markdown + xhtml2pdf, Arial Unicode for ω, Δ, −, ≈ ...).

    .venv/bin/python scripts/md_to_pdf.py ROTX7.md            # -> ROTX7.pdf
"""
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
CSS = f"""
@font-face {{ font-family: uni; src: url("{FONT}"); }}
@page {{ size: a4; margin: 1.5cm; }}
body {{ font-family: uni; font-size: 9pt; line-height: 1.3; }}
h1 {{ font-size: 16pt; }} h2 {{ font-size: 13pt; margin-top: 12pt; }} h3 {{ font-size: 11pt; }}
table {{ border-collapse: collapse; margin: 6pt 0; }}
th, td {{ border: 0.5pt solid #888; padding: 2pt 3pt; font-size: 7pt; }}
th {{ background-color: #e8e8e8; }}
code {{ font-family: uni; background-color: #f0f0f0; }}
"""

src = Path(sys.argv[1])
dest = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
html = markdown.markdown(src.read_text(), extensions=["tables"])
with open(dest, "wb") as f:
    status = pisa.CreatePDF(f"<html><head><style>{CSS}</style></head><body>{html}</body></html>", dest=f)
assert not status.err, f"{status.err} errors"
print("wrote", dest)
