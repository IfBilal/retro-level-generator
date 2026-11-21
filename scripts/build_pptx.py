"""Build slides/slides.pptx (editable PowerPoint) from the same slide definitions as the PDF: python scripts/build_pptx.py

Explain-me:
  What: takes SLIDES from scripts/build_slides.py (title + small HTML body), fills the {{keys}} from the saved results exactly like the PDF,
        and writes one PowerPoint slide per entry: bullets as text, <img> as pictures, <table> as a table.  So the PDF and the PPTX
        always say the same thing and every number still comes from a saved artifact.
  In/Out: report/figures + reports/** -> slides/slides.pptx.  Previous: scripts/build_slides.py.  Next: the submission zip.
  Try: python scripts/build_pptx.py && open slides/slides.pptx in PowerPoint / LibreOffice.
"""
import html as htmllib
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_report as BR
import build_slides as BS
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

NAVY = RGBColor(0x1B, 0x2A, 0x49)


class Body(HTMLParser):
    """Flatten a slide body into [("p"|"li", text), ("img", src), ("table", rows)] in document order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items, self._buf, self._tag, self._table, self._row, self._cell = [], [], None, None, None, None

    def _flush(self):
        text = " ".join("".join(self._buf).split())
        if text and self._tag in ("p", "li"):
            self.items.append((self._tag, text))
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("p", "li"):
            self._flush()
            self._tag = tag
        elif tag == "img" and a.get("src"):
            self.items.append(("img", a["src"]))
        elif tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("p", "li"):
            self._flush()
            self._tag = None
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.items.append(("table", self._table))
            self._table = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)
        elif self._tag:
            self._buf.append(data)


def add_text(slide, paras, left, top, width, height, big=False):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (kind, text) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if kind == "li" else "") + text
        p.font.size = Pt(34 if big else 15)
        p.font.bold = big
        p.font.color.rgb = NAVY
        p.space_after = Pt(6)


def main():
    vals = BR.values()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    best = sorted((ROOT / "report/figures").glob("fig12_render_best_*.png"))
    best_name = best[0].name if best else "fig08_cond_samples.png"
    for i, (title, body) in enumerate(BS.SLIDES, 1):
        body = body.replace("{FIG}", "").replace("fig12_render_best_000.png", best_name)
        parser = Body()
        parser.feed(BR.render(body, vals))
        parser._flush()
        slide = prs.slides.add_slide(blank)
        tt = slide.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12.1), Inches(0.9))
        tt.text_frame.text = htmllib.unescape(title)
        tt.text_frame.paragraphs[0].font.size = Pt(30)
        tt.text_frame.paragraphs[0].font.bold = True
        tt.text_frame.paragraphs[0].font.color.rgb = NAVY
        texts = [it for it in parser.items if it[0] in ("p", "li")]
        imgs = [it[1] for it in parser.items if it[0] == "img"]
        tables = [it[1] for it in parser.items if it[0] == "table"]
        has_visual = bool(imgs or tables)
        add_text(slide, texts, Inches(0.6), Inches(1.3), Inches(12.1), Inches(3.6 if has_visual else 5.6), big=(i == 1))
        top = Inches(5.0 if texts else 1.4)
        for rows in tables:
            shape = slide.shapes.add_table(len(rows), max(len(r) for r in rows), Inches(0.6), Inches(1.4 if not texts else 4.2), Inches(8), Inches(0.4 * len(rows)))
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    shape.table.cell(r, c).text = cell
                    shape.table.cell(r, c).text_frame.paragraphs[0].font.size = Pt(14)
        x = Inches(0.6)
        for src in imgs:
            path = (ROOT / "report/figures" / Path(src).name)
            if not path.exists():
                continue
            h = Inches(2.3 if texts else 5.2)
            pic = slide.shapes.add_picture(str(path), x, top, height=h)
            if pic.width > Emu(Inches(12.1)):                                  # scale a wide picture to the slide width
                ratio = Inches(12.1) / pic.width
                pic.width, pic.height = int(pic.width * ratio), int(pic.height * ratio)
            x += pic.width + Inches(0.3)
        n = slide.shapes.add_textbox(Inches(12.3), Inches(7.0), Inches(0.9), Inches(0.4))
        n.text_frame.text = f"{i}/{len(BS.SLIDES)}"
        n.text_frame.paragraphs[0].font.size = Pt(11)
    out = ROOT / "slides" / "slides.pptx"
    prs.save(out)
    print("wrote", out.relative_to(ROOT), out.stat().st_size // 1024, "KB,", len(BS.SLIDES), "slides")


if __name__ == "__main__":
    main()
