#!/usr/bin/env python3
"""Split the compiled manuscript into the main text and the Supplementary
Information, which the journal takes as a separate PDF. The Supplementary
Information starts at the first page whose label begins with "S" (main.tex
numbers its pages S1, S2, ...).

Usage: split_supplement.py <compiled main.pdf> <main text pdf> <supplementary pdf>
"""
import logging
import sys

from pypdf import PdfReader, PdfWriter

src, main_out, sup_out = sys.argv[1:4]
logging.getLogger("pypdf").setLevel(logging.ERROR)   # link-mapping notices only
reader = PdfReader(src)
first = next(i for i, label in enumerate(reader.page_labels) if label.startswith("S"))
for out, pages in ((main_out, (0, first)), (sup_out, (first, len(reader.pages)))):
    writer = PdfWriter()
    writer.append(reader, pages=pages)
    with open(out, "wb") as f:
        writer.write(f)
    print(f"{out}: pages {pages[0] + 1}-{pages[1]} of {len(reader.pages)}")
