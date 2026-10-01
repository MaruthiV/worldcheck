import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

# dataviz reference palette; gray is the de-emphasis slot, so it fails chroma on purpose
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "accent": "#2a78d6", "context": "#898781"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "accent": "#3987e5", "context": "#898781"},
}
W = 720
LEFT = 186


def text(x, y, s, fill, size=13, weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" font-family=\'{FONT}\'>{s}</text>')


def frame(t, h, title, subtitle, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}">'
            f'<rect width="{W}" height="{h}" rx="8" fill="{t["surface"]}"/>'
            + text(24, 34, title, t["ink"], 15, 600) + text(24, 56, subtitle, t["ink2"], 12.5)
            + body + "</svg>\n")


# square at the baseline, 4px round at the data end
def column(x, w, base, top, fill):
    r = min(4, (base - top) / 2, w / 2)
    return (f'<path d="M{x},{base} V{top + r} Q{x},{top} {x + r},{top} H{x + w - r} '
            f'Q{x + w},{top} {x + w},{top + r} V{base} Z" fill="{fill}"/>')


def pages_figure(t):
    pages = json.loads((ROOT / "results" / "t1_pagination.json").read_text())["cases"]["ascending_sweep"]["pages"]
    h, base, top, x0, plot_w = 300, 236, 92, 72, 600
    scale = (base - top) / 20
    body = ""
    for v in (0, 10, 20):
        y = base - v * scale
        body += f'<line x1="{x0}" x2="{x0 + plot_w}" y1="{y}" y2="{y}" stroke="{t["grid"] if v else t["axis"]}" stroke-width="1"/>'
        body += text(x0 - 10, y + 4, v, t["muted"], 12, anchor="end")
    slot = plot_w / len(pages)
    for i, p in enumerate(pages):
        cx = x0 + slot * i + slot / 2
        g, s = len(p["guard_ids"]), len(p["scheduler_ids"])
        body += column(cx - 25, 24, base, base - g * scale, t["context"])
        if s:
            body += column(cx + 1, 24, base, base - s * scale, t["accent"])
        body += text(cx + 13, base - s * scale - 7, s, t["ink"], 12.5, 600, "middle")
        body += text(cx, base + 20, f'Page {p["page_index"]}', t["ink2"], 12.5, anchor="middle")
    ly = h - 22
    body += f'<rect x="{x0}" y="{ly - 9}" width="10" height="10" rx="2" fill="{t["context"]}"/>'
    body += text(x0 + 16, ly, "What the responder computed", t["ink2"], 12.5)
    body += f'<rect x="{x0 + 230}" y="{ly - 9}" width="10" height="10" rx="2" fill="{t["accent"]}"/>'
    body += text(x0 + 246, ly, "What the agent received", t["ink2"], 12.5)
    return frame(t, h, "The agent gets one page, then nothing",
                 "Songs per page on row 692c77d_2, an 80-song library at 20 per page. The agent stops at page 1.",
                 body)


def dot_strip(t, title, subtitle, rows, ticks, fmt, labels):
    row_h, top = 58, 96
    h = top + row_h * len(rows) + 44
    x0, x1 = LEFT, W - 128
    lo, hi = ticks[0], ticks[-1]
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    base = top + row_h * len(rows) - 10
    body = ""
    for v in ticks:
        body += f'<line x1="{sx(v)}" x2="{sx(v)}" y1="{top - 8}" y2="{base}" stroke="{t["grid"] if v != lo else t["axis"]}" stroke-width="1"/>'
        body += text(sx(v), base + 18, fmt(v), t["muted"], 12, anchor="middle")
    for i, (name, vals) in enumerate(rows):
        y = top + row_h * i + row_h / 2 - 6
        body += text(24, y + 4, name, t["ink2"], 12.5)
        # dodge dots that would collide so every run stays visible
        placed = []
        for v in sorted(vals):
            x = sx(v)
            dy = next((d for d in (0, -12, 12, -24, 24)
                       if all(abs(x - px) >= 12 or abs(d - pd) >= 12 for px, pd in placed)), 0)
            placed.append((x, dy))
            body += (f'<circle cx="{x}" cy="{y + dy}" r="5" fill="{t["accent"]}" '
                     f'stroke="{t["surface"]}" stroke-width="2"/>')
        body += text(x1 + 16, y + 4, labels[i], t["ink"], 12.5, 600)
    return frame(t, h, title, subtitle, body)


def lost_pages_figure(t):
    losses = json.loads((ROOT / "results" / "grpo_summary.json").read_text())["live_page_losses"]
    rows = [(name, [losses[f"grpo-{arm}-s{s}"]["held_records_but_came_back_empty"] for s in range(5)])
            for name, arm in (("Shipped plugin", "shipped"), ("Fixed plugin", "patched"))]
    labels = [f"{min(v)} to {max(v)} per run" if max(v) else "0 in every run" for _, v in rows]
    return dot_strip(t, "During real training, pages with records came back empty",
                     "Lost pages per GRPO run, one dot per seed. Each later-page request replayed through the responder.",
                     rows, [0, 40, 80, 120], str, labels)


def skip_work_figure(t):
    d = json.loads((ROOT / "results" / "grpo_summary.json").read_text())["exploratory_finishing_without_acting"]["patched"]
    rows = [("Fine-tuned only", [100 * d["sft-s0"]]),
            ("GRPO, shipped plugin", [100 * d[f"grpo-shipped-s{s}"] for s in range(5)]),
            ("GRPO, fixed plugin", [100 * d[f"grpo-patched-s{s}"] for s in range(5)])]
    labels = [f"{round(min(v))}%" if len(v) == 1 else f"{round(min(v))} to {round(max(v))}%" for _, v in rows]
    return dot_strip(t, "After GRPO, agents 'finish' tasks without doing them",
                     "Action-task attempts scoring 0.9+ with no state-changing call. One dot per seed. Exploratory.",
                     rows, [0, 20, 40, 60], lambda v: f"{v}%", labels)


def main():
    for name, fn in (("pages", pages_figure), ("lost-pages", lost_pages_figure), ("skip-work", skip_work_figure)):
        for mode, t in THEMES.items():
            (OUT / f"{name}-{mode}.svg").write_text(fn(t))
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
