"""Render the whitepaper report.md + charts to a PDF for the repo."""
import base64
import re
from pathlib import Path

ROOT = Path("/home/hatch/workspace/ecfr-mcp")
RN = Path("/home/hatch/workspace/research_notes/ecfr-retrieval-whitepaper")
md = (RN / "report.md").read_text()


def img_tag(name: str) -> str:
    b64 = base64.b64encode((RN / name).read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" '
            f'style="width:100%"/></figure>')


# map chart pngs to figures placed after their section
figures = {
    "progression.png": "Figure 1 — Recall@1 progression across the seven rounds.",
    "recall_all.png": "Figure 2 — Recall across all nine configurations.",
    "recall_by_type.png": "Figure 3 — Recall@1 by query type for the three leaders.",
    "cost_frontier.png": "Figure 4 — The efficiency frontier: recall@1 vs experiment cost.",
}

html = ["<html><head><meta charset='utf-8'><style>",
        "body{font-family:Georgia,serif;max-width:720px;margin:40px auto;"
        "padding:0 24px;color:#1a1a1a;line-height:1.6;font-size:15px}",
        "h1{font-size:26px;border-bottom:2px solid #1a56db;padding-bottom:8px}",
        "h2{font-size:19px;color:#1a56db;margin-top:34px}",
        "table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}",
        "th,td{border:1px solid #d1d5db;padding:6px 8px;text-align:left}",
        "th{background:#eff6ff}",
        "figure{margin:20px 0}figcaption{font-size:12px;color:#555;"
        "text-align:center;margin-top:6px}",
        "code{background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:13px}",
        "@page{size:A4;margin:22mm}",
        "</style></head><body>"]

lines = md.split("\n")
i = 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith("# "):
        html.append(f"<h1>{ln[2:]}</h1>")
    elif ln.startswith("## "):
        html.append(f"<h2>{ln[3:]}</h2>")
        if ln.startswith("## 3."):
            for n in ("progression.png", "recall_all.png"):
                html.append(img_tag(n)[:-9] +
                            f"<figcaption>{figures[n]}</figcaption></figure>")
        elif ln.startswith("## 5."):
            n = "cost_frontier.png"
            html.append(img_tag(n)[:-9] +
                        f"<figcaption>{figures[n]}</figcaption></figure>")
    elif ln.startswith("|"):
        tbl, hdr = [], True
        is_pertype = False
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c.strip() for c in lines[i].strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                i += 1
                continue
            if "natural (n=33)" in cells:
                is_pertype = True
            tag = "th" if hdr else "td"
            tbl.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            hdr = False
            i += 1
        html.append("<table>" + "".join(tbl) + "</table>")
        if is_pertype:
            n = "recall_by_type.png"
            html.append(img_tag(n)[:-9] +
                        f"<figcaption>{figures[n]}</figcaption></figure>")
        continue
    elif ln.strip() == "":
        pass
    else:
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", ln)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        if re.match(r"\d+\. ", t):
            html.append(f"<p>{t}</p>")
        elif t.startswith("- "):
            html.append(f"<p>&bull; {t[2:]}</p>")
        else:
            html.append(f"<p>{t}</p>")
    i += 1

html.append("</body></html>")
out = ROOT / "whitepaper" / "ecfr-retrieval-whitepaper.pdf"
from weasyprint import HTML
HTML(string="".join(html)).write_pdf(str(out))
print("pdf written:", out, out.stat().st_size, "bytes")
