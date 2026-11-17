"""McKinsey-style charts for L2Kotak.md. All numbers are copied from L2Kotak.md (sections 3.2 and 6)."""
import sys, os

OUT = sys.argv[1]
NAVY, MID, LIGHT = "#051C2C", "#8B939E", "#B3B9C2"
INK, MUTED, RULE = "#1F2329", "#5B6470", "#D5D9DE"
FONT = "Arial, Helvetica, sans-serif"


def doc(w, bottom, title, subtitle, body):
    """No source line. With title=None the header is dropped and the canvas starts just above the legend."""
    if title:
        top = 0
        head = f'''<text x="40" y="44" font-size="21" font-weight="700" fill="{INK}">{title}</text>
<text x="40" y="70" font-size="14" fill="{MUTED}">{subtitle}</text>
<line x1="40" y1="86" x2="{w-40}" y2="86" stroke="{INK}" stroke-width="1"/>'''
    else:
        top, head = 80, ""
    h = bottom - top
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {top} {w} {h}" width="{w}" height="{h}" font-family="{FONT}">
<rect y="{top}" width="{w}" height="{h}" fill="#FFFFFF"/>
{head}
{body}
</svg>
'''


def t(x, y, s, size=13, fill=INK, anchor="start", weight="400"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'


def legend(x, y, items):
    out = []
    for color, label in items:
        out.append(f'<rect x="{x}" y="{y-11}" width="12" height="12" rx="2" fill="{color}"/>')
        out.append(t(x + 18, y, label, 13, INK))
        x += 18 + len(label) * 7.2 + 26
    return "\n".join(out)


# ---------- Chart 1: leave-one-provider-out spellF1 ----------
def chart1():
    W, H = 900, 540
    # last field: the report's own "mean vs A" difference (computed on unrounded scores)
    data = [("IDSSE", 0.558, 0.560, 0.563, "+0.005"),
            ("SkillCorner", 0.514, 0.520, 0.528, "+0.013"),
            ("PFF games 32–63", 0.622, 0.619, 0.664, "+0.043")]
    base, top, vmax = 440, 150, 0.70
    sy = (base - top) / vmax
    x0, gw, bw, gap = 80, 260, 58, 3
    b = [legend(40, 116, [(LIGHT, "Model A (TCN)"), (MID, "Model B (MS-TCN)"), (NAVY, "Equal mean of both (submitted recipe)")])]
    for i, (name, a, bb, m, d) in enumerate(data):
        gx = x0 + i * gw + (gw - (3 * bw + 2 * gap)) / 2
        for j, (v, c) in enumerate([(a, LIGHT), (bb, MID), (m, NAVY)]):
            x = gx + j * (bw + gap)
            h = v * sy
            b.append(f'<rect x="{x:.1f}" y="{base-h:.1f}" width="{bw}" height="{h:.1f}" fill="{c}"/>')
            b.append(t(x + bw / 2, base - h - 8, f"{v:.3f}", 13, INK, "middle", "700" if j == 2 else "400"))
        cx = gx + (3 * bw + 2 * gap) / 2
        b.append(t(cx, base + 24, name, 14, INK, "middle", "700"))
        b.append(t(cx, base + 44, f"mean vs A: {d}", 12.5, MUTED, "middle"))
    b.append(f'<line x1="40" y1="{base}" x2="{W-40}" y2="{base}" stroke="{INK}" stroke-width="1"/>')
    return doc(W, 500, None, None, "\n".join(b))


# ---------- Chart 2: gain of the mean over model A, with 95% CI ----------
def chart2():
    W, H = 900, 440
    rows = [("IDSSE", 0.005, -0.006, 0.014),
            ("SkillCorner", 0.013, 0.007, 0.019),
            ("PFF games 32–63", 0.043, 0.037, 0.049)]
    lo, hi, px0, px1 = -0.01, 0.055, 220, 700
    sx = lambda v: px0 + (v - lo) / (hi - lo) * (px1 - px0)
    b = []
    ys = [150, 225, 300]
    for (name, m, l, h), y in zip(rows, ys):
        c = MID if l < 0 else NAVY
        b.append(t(40, y + 5, name, 14, INK, "start", "700"))
        b.append(f'<line x1="{sx(l):.1f}" y1="{y}" x2="{sx(h):.1f}" y2="{y}" stroke="{c}" stroke-width="3" stroke-linecap="round"/>')
        b.append(f'<circle cx="{sx(m):.1f}" cy="{y}" r="7" fill="{c}" stroke="#fff" stroke-width="2"/>')
        b.append(t(sx(h) + 14, y + 5, f'<tspan font-weight="700">+{m:.3f}</tspan>  [{l:+.3f}, {h:+.3f}]', 13, INK))
    b.append(t(sx(0.014) + 14, ys[0] + 24, "interval includes zero: no clear gain", 12, MUTED))
    zx = sx(0)
    b.append(f'<line x1="{zx:.1f}" y1="118" x2="{zx:.1f}" y2="335" stroke="{INK}" stroke-width="1"/>')
    for v in [-0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05]:
        b.append(f'<line x1="{sx(v):.1f}" y1="335" x2="{sx(v):.1f}" y2="341" stroke="{MUTED}"/>')
        b.append(t(sx(v), 358, f"{v:+.2f}" if v else "0", 12, MUTED, "middle"))
    b.append(t((px0 + px1) / 2, 380, "spellF1 gain of the equal mean over model A", 12.5, MUTED, "middle"))
    return doc(W, 400, None, None, "\n".join(b))


# ---------- Chart 3: training hours per model ----------
def chart3():
    W, H = 900, 400
    rows = [("Model A", "PFF games 0–31"), ("Model B", "PFF games 32–63")]
    idsse, sc, pff_half = 11.1, 32.8, 108.1 / 2
    px0, pxw, vmax = 150, 620, 100
    s = pxw / vmax
    b = [legend(40, 116, [(LIGHT, "IDSSE / DFL (stadium camera)"), (MID, "SkillCorner (broadcast)"), (NAVY, "PFF FC World Cup 2022 (broadcast)")])]
    for i, (name, pname) in enumerate(rows):
        y = 160 + i * 90
        b.append(t(40, y + 28, name, 14, INK, "start", "700"))
        x = px0
        for v, c, lab, tc in [(idsse, LIGHT, "11.1 h", INK), (sc, MID, "32.8 h", "#FFFFFF"), (pff_half, NAVY, f"{pname} · ≈54 h", "#FFFFFF")]:
            w = v * s
            b.append(f'<rect x="{x:.1f}" y="{y}" width="{w-2:.1f}" height="44" fill="{c}"/>')
            b.append(t(x + (w - 2) / 2, y + 27, lab, 12.5, tc, "middle", "700" if c == NAVY else "400"))
            x += w
        b.append(t(x + 10, y + 27, "≈ 98 h", 14, INK, "start", "700"))
    return doc(W, 325, None, None, "\n".join(b))


# ---------- Chart 4: two development steps ----------
def chart4():
    W, H = 900, 430
    groups = [("Final selection step", "tested once on PFF games 0–31", "Previous final model", 0.6778, "0.6778", "Submitted model", 0.7114, "0.7114", "+0.0336, better in 32 of 32 games"),
              ("Adding PFF to model B", "tested on unseen SkillCorner", "Without PFF", 0.457, "0.457", "With PFF", 0.516, "0.516", "+0.059, better in 20 of 20 matches")]
    px0, pxw, vmax = 450, 360, 0.8
    s = pxw / vmax
    b = [legend(40, 116, [(LIGHT, "Before"), (NAVY, "After")])]
    for i, (gt, gs, l1, v1, s1, l2, v2, s2, delta) in enumerate(groups):
        y = 150 + i * 120
        b.append(t(40, y + 16, gt, 14, INK, "start", "700"))
        b.append(t(40, y + 35, gs, 12.5, MUTED))
        b.append(t(40, y + 56, delta, 13, NAVY, "start", "700"))
        for j, (lab, v, sv, c) in enumerate([(l1, v1, s1, LIGHT), (l2, v2, s2, NAVY)]):
            yy = y + j * 33
            b.append(t(px0 - 10, yy + 19, lab, 12.5, INK, "end"))
            b.append(f'<rect x="{px0}" y="{yy}" width="{v*s:.1f}" height="28" fill="{c}"/>')
            b.append(t(px0 + v * s + 8, yy + 19, sv, 13, INK, "start", "700" if j else "400"))
    b.append(f'<line x1="{px0}" y1="140" x2="{px0}" y2="{150+120+66+10}" stroke="{INK}" stroke-width="1"/>')
    return doc(W, 355, None, None, "\n".join(b))


for name, fn in [("chart1_unseen_provider.svg", chart1), ("chart2_gain_ci.svg", chart2),
                 ("chart3_training_hours.svg", chart3), ("chart4_dev_steps.svg", chart4)]:
    with open(os.path.join(OUT, name), "w") as f:
        f.write(fn())
    print("wrote", name)
