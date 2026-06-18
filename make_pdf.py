"""Render PREPRINT-draft.md to a clean A4 PDF for Zenodo (pure-Python: markdown -> HTML ->
xhtml2pdf). Uses DejaVu Sans (full Unicode coverage: minus sign, subscripts, Greek, arrows)
so nothing renders as a missing-glyph box."""
import os
import markdown
from xhtml2pdf import pisa
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

BASE = r"E:\BiniruProjects\psyche-sim"
FONTDIR = os.path.join(BASE, "tvb-env", "Lib", "site-packages", "matplotlib", "mpl-data", "fonts", "ttf")
def fp(n): return os.path.join(FONTDIR, n)

# register DejaVu Sans with reportlab directly (reliable; xhtml2pdf then resolves by name)
def reg(name, fname):
    p = fp(fname)
    if os.path.exists(p):
        pdfmetrics.registerFont(TTFont(name, p)); return True
    return False
reg("dv", "DejaVuSans.ttf")
hb = reg("dv-b", "DejaVuSans-Bold.ttf")
hi = reg("dv-i", "DejaVuSans-Oblique.ttf")
hbi = reg("dv-bi", "DejaVuSans-BoldOblique.ttf")
addMapping("dv", 0, 0, "dv")
addMapping("dv", 1, 0, "dv-b" if hb else "dv")
addMapping("dv", 0, 1, "dv-i" if hi else "dv")
addMapping("dv", 1, 1, "dv-bi" if hbi else ("dv-b" if hb else "dv"))

src = open(os.path.join(BASE, "PREPRINT-draft.md"), encoding="utf-8").read()

# PDF-only font fallback: DejaVu Sans lacks a few math-notation glyphs; substitute ASCII-safe
# equivalents for the PDF (the GitHub markdown keeps the original Unicode; same meaning).
src = src.replace("⟨", "&lt;").replace("⟩", "&gt;").replace("ⱼ", "j")

# quick guard: warn on glyphs DejaVu Sans lacks (e.g. emoji) so we don't ship boxes
suspect = sorted({c for c in src if ord(c) > 0x2600})
if suspect:
    print("WARNING: chars beyond DejaVu coverage present:", [hex(ord(c)) for c in suspect])
else:
    print("[glyphs] no emoji/exotic chars beyond DejaVu Sans coverage.")

body = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists"])

css = """
@page {{ size: a4; margin: 2.0cm 1.9cm 2.2cm 1.9cm;
         @frame footer {{ -pdf-frame-content: footerContent; bottom: 1.1cm; height: 1cm; left: 1.9cm; right: 1.9cm; }} }}
body {{ font-family: dv; font-size: 10pt; line-height: 1.42; text-align: justify; color: #111; }}
h1 {{ font-size: 17pt; line-height: 1.22; margin-bottom: 2px; }}
h2 {{ font-size: 12.5pt; margin-top: 15px; margin-bottom: 4px; border-bottom: 1px solid #999; padding-bottom: 2px; }}
h3 {{ font-size: 10.5pt; margin-top: 11px; margin-bottom: 3px; }}
p {{ margin: 4px 0; }}
strong {{ font-weight: bold; }}
em {{ font-style: italic; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6px 0; }}
th, td {{ border: 0.5px solid #888; padding: 3px 5px; }}
th {{ background-color: #ececec; font-weight: bold; }}
code {{ font-family: "Courier"; font-size: 9pt; background-color: #f3f3f3; }}
blockquote {{ border-left: 3px solid #aaa; margin-left: 0; padding-left: 10px; color: #333; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 8px 0; }}
a {{ color: #1a4f8b; text-decoration: none; }}
ul, ol {{ margin: 4px 0 4px 0; }}
li {{ margin: 1px 0; }}
""".format()

html = ("<html><head><meta charset='utf-8'><style>" + css + "</style></head><body>"
        + body
        + "<div id='footerContent' style='font-size:8pt; color:#777; text-align:center;'>"
          "Ego dissolution as a collapse of integration | preprint draft | "
          "page <pdf:pagenumber> of <pdf:pagecount></div>"
        + "</body></html>")

def link_callback(uri, rel):
    return uri  # our font url()s are absolute filesystem paths

out = os.path.join(BASE, "PREPRINT-ego-dissolution-dmf.pdf")
with open(out, "wb") as f:
    res = pisa.CreatePDF(html, dest=f, encoding="utf-8", link_callback=link_callback)
print("pisa errors:", res.err)
print("saved:", out, os.path.getsize(out), "bytes")
