# -*- coding: utf-8 -*-
#
# Turns the cv.md written by cv_project.py into a PDF.
#
#   python cv_to_pdf.py                    cv.md -> cv.pdf
#   python cv_to_pdf.py my_cv.md out.pdf   any input and output path
#
# Or from other code: markdown_to_pdf(markdown_text, Path("cv.pdf"))
#
# Needs: pip install markdown xhtml2pdf  (both pure Python, so they work in
# Colab and locally with no system libraries). xhtml2pdf uses the built-in
# Helvetica font, which covers accented Latin letters but not other scripts.
#

import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

DEFAULT_MARKDOWN_PATH = Path("cv.md")

#
# xhtml2pdf supports a subset of CSS 2: plain selectors, fonts, margins and
# borders. Keep the style simple.
#
CV_CSS = """
@page { size: a4; margin: 1.8cm 2cm; }
body { font-family: Helvetica; font-size: 10.5pt; line-height: 1.35; color: #222222; }
h1 { font-size: 22pt; margin: 0 0 2pt 0; color: #111111; }
h2 { font-size: 13pt; margin: 14pt 0 4pt 0; padding-bottom: 2pt;
     border-bottom: 1px solid #999999; color: #111111; }
h3 { font-size: 11pt; margin: 8pt 0 0 0; }
p { margin: 0 0 4pt 0; }
ul { margin: 2pt 0 4pt 0; }
li { margin: 0 0 2pt 0; }
em { color: #555555; }
"""


def separate_lists(markdown_text: str) -> str:
    #
    # Python-Markdown only starts a list after a blank line, but cv.md puts
    # the bullets straight under the *dates* line (fine in Colab's renderer).
    # Without this, the dates and bullets merge into one paragraph.
    #
    lines = []
    for line in markdown_text.split("\n"):
        previous_line = lines[-1] if lines else ""
        if line.startswith("- ") and previous_line.strip() and not previous_line.startswith("- "):
            lines.append("")
        lines.append(line)
    return "\n".join(lines)


def markdown_to_html(markdown_text: str) -> str:
    body = markdown.markdown(separate_lists(markdown_text))
    return (f'<html><head><meta charset="utf-8"><style>{CV_CSS}</style></head>'
            f"<body>{body}</body></html>")


def markdown_to_pdf(markdown_text: str, pdf_path: Path) -> None:
    with pdf_path.open("wb") as pdf_file:
        result = pisa.CreatePDF(markdown_to_html(markdown_text), dest=pdf_file, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"xhtml2pdf reported {result.err} errors while writing {pdf_path}")


def main() -> None:
    markdown_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MARKDOWN_PATH
    pdf_path = Path(sys.argv[2]) if len(sys.argv) > 2 else markdown_path.with_suffix(".pdf")
    markdown_to_pdf(markdown_path.read_text(encoding="utf-8"), pdf_path)
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
