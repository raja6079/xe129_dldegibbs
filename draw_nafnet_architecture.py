"""Generate a publication-quality NAFNet-Small architecture diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

C_INPUT  = "#4A90D9"
C_INTRO  = "#5B9BD5"
C_ENC    = "#70AD47"
C_MIDDLE = "#C00000"
C_DEC    = "#FFC000"
C_SKIP   = "#A6A6A6"
C_RESID  = "#9B59B6"

WIDTH       = 32
ENC_BLKS    = [2, 2, 4, 8]
MIDDLE_BLKS = 12
DEC_BLKS    = [2, 2, 2, 2]
IN_CH       = 2
OUT_CH      = 2

fig_w, fig_h = 24, 17
fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#1A1A2E")
ax.set_facecolor("#1A1A2E")
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

def rbox(ax, x, y, w, h, fc, text, fs=8.5, tc="white", lw=2, ec=None):
    if ec is None: ec = fc
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold", zorder=4)

def varrow(ax, x, y0, y1, color="#AAAAAA", lw=1.8):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=12), zorder=5)

# ── geometry ────────────────────────────────────────────────────────────────
BOX_W = 3.2
BOX_H = 0.72
CX    = (fig_w - BOX_W) / 2   # left edge of centre column

Y = {
    "input" : 16.00,
    "intro" : 14.90,
    "enc1"  : 13.70,
    "enc2"  : 12.50,
    "enc3"  : 11.30,
    "enc4"  : 10.10,
    "mid"   :  8.55,
    "dec4"  :  7.10,
    "dec3"  :  5.90,
    "dec2"  :  4.70,
    "dec1"  :  3.50,
    "end"   :  2.30,
    "out"   :  1.20,
}

LABELS = {
    "input": f"Input  [B, {IN_CH}, H, W]\nComplex: real + imag",
    "intro": "Intro  Conv 3×3\n2 → 32 ch",
    "enc1" : f"Encoder L1  |  W=32\n×{ENC_BLKS[0]} NAFBlocks   [B,32,H,W]",
    "enc2" : f"Encoder L2  |  W=64\n×{ENC_BLKS[1]} NAFBlocks   [B,64,H/2,W/2]",
    "enc3" : f"Encoder L3  |  W=128\n×{ENC_BLKS[2]} NAFBlocks   [B,128,H/4,W/4]",
    "enc4" : f"Encoder L4  |  W=256\n×{ENC_BLKS[3]} NAFBlocks   [B,256,H/8,W/8]",
    "mid"  : f"Bottleneck  |  W=512\n×{MIDDLE_BLKS} NAFBlocks   [B,512,H/16,W/16]",
    "dec4" : f"Decoder L4  |  W=256\n×{DEC_BLKS[3]} NAFBlocks   [B,256,H/8,W/8]",
    "dec3" : f"Decoder L3  |  W=128\n×{DEC_BLKS[2]} NAFBlocks   [B,128,H/4,W/4]",
    "dec2" : f"Decoder L2  |  W=64\n×{DEC_BLKS[1]} NAFBlocks   [B,64,H/2,W/2]",
    "dec1" : f"Decoder L1  |  W=32\n×{DEC_BLKS[0]} NAFBlocks   [B,32,H,W]",
    "end"  : "Ending  Conv 3×3\n32 → 2 ch",
    "out"  : f"Output  [B, {OUT_CH}, H, W]\n+ Global Residual  (out = net(x) + x)",
}

COLS = {
    "input": C_INPUT, "intro": C_INTRO,
    "enc1": C_ENC, "enc2": C_ENC, "enc3": C_ENC, "enc4": C_ENC,
    "mid": C_MIDDLE,
    "dec4": C_DEC, "dec3": C_DEC, "dec2": C_DEC, "dec1": C_DEC,
    "end": C_INTRO, "out": C_INPUT,
}

EC = {
    "input": "#2166AC", "intro": "#2166AC",
    "enc1": "#2D6A0A", "enc2": "#2D6A0A", "enc3": "#2D6A0A", "enc4": "#2D6A0A",
    "mid": "#7B0000",
    "dec4": "#8B6914", "dec3": "#8B6914", "dec2": "#8B6914", "dec1": "#8B6914",
    "end": "#2166AC", "out": "#2166AC",
}

ORDER = ["input","intro","enc1","enc2","enc3","enc4","mid","dec4","dec3","dec2","dec1","end","out"]

for k in ORDER:
    bw = BOX_W * 1.35 if k == "mid" else BOX_W
    bx = CX - (bw - BOX_W) / 2
    rbox(ax, bx, Y[k]-BOX_H/2, bw, BOX_H,
         COLS[k], LABELS[k], fs=8, ec=EC[k])

# ── spine arrows ─────────────────────────────────────────────────────────────
for i in range(len(ORDER)-1):
    k0, k1 = ORDER[i], ORDER[i+1]
    y0 = Y[k0] - BOX_H/2
    y1 = Y[k1] + BOX_H/2
    enc_down = k1 in ("enc1","enc2","enc3","enc4","mid")
    dec_up   = k0 in ("mid","dec4","dec3","dec2","dec1")
    c = "#33DD77" if enc_down else ("#FF9944" if dec_up else "#9999CC")
    varrow(ax, CX + BOX_W/2, y0, y1, color=c)

# stride-2 / PixelShuffle labels
for k0, k1 in [("enc1","enc2"),("enc2","enc3"),("enc3","enc4"),("enc4","mid")]:
    ym = (Y[k0]+Y[k1])/2
    ax.text(CX+BOX_W/2+0.15, ym, "↓ stride-2 Conv2×2",
            ha="left", va="center", fontsize=6.8, color="#33DD77", zorder=5)

for k0, k1 in [("mid","dec4"),("dec4","dec3"),("dec3","dec2"),("dec2","dec1")]:
    ym = (Y[k0]+Y[k1])/2
    ax.text(CX+BOX_W/2+0.15, ym, "↑ PixelShuffle ×2",
            ha="left", va="center", fontsize=6.8, color="#FF9944", zorder=5)

# ── additive skip connections ─────────────────────────────────────────────────
skip_pairs  = [("enc1","dec1"),("enc2","dec2"),("enc3","dec3"),("enc4","dec4")]
x_offsets   = [-2.2, -3.1, -4.0, -4.9]

for (ek, dk), xoff in zip(skip_pairs, x_offsets):
    ye, yd = Y[ek], Y[dk]
    xpipe  = CX + xoff
    ax.plot([CX, xpipe], [ye, ye], color=C_SKIP, lw=1.6, ls="--", zorder=2)
    ax.plot([xpipe, xpipe], [ye, yd], color=C_SKIP, lw=1.6, ls="--", zorder=2)
    ax.annotate("", xy=(CX, yd), xytext=(xpipe, yd),
                arrowprops=dict(arrowstyle="-|>", color=C_SKIP, lw=1.6,
                                mutation_scale=10), zorder=5)
    ax.text(CX-0.15, yd+0.05, "+", color=C_SKIP, fontsize=12,
            fontweight="bold", ha="right", va="center", zorder=5)

# ── global residual (right side) ────────────────────────────────────────────
xr = CX + BOX_W + 2.1
ax.plot([CX+BOX_W, xr], [Y["input"], Y["input"]], color=C_RESID, lw=2.2, zorder=2)
ax.plot([xr, xr], [Y["input"], Y["out"]], color=C_RESID, lw=2.2, zorder=2)
ax.annotate("", xy=(CX+BOX_W, Y["out"]), xytext=(xr, Y["out"]),
            arrowprops=dict(arrowstyle="-|>", color=C_RESID, lw=2.2,
                            mutation_scale=12), zorder=5)
ax.text(xr+0.12, (Y["input"]+Y["out"])/2,
        "Global Residual\nout = net(x) + x",
        color=C_RESID, fontsize=8, va="center", ha="left",
        fontweight="bold", zorder=5)
ax.text(CX+BOX_W+0.08, Y["out"], "+",
        color=C_RESID, fontsize=13, fontweight="bold",
        ha="left", va="center", zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# NAFBlock detail panel (lower-left)
# ═══════════════════════════════════════════════════════════════════════════════
PX, PY, PW, PH = 0.30, 0.10, 8.6, 8.2
ax.add_patch(FancyBboxPatch((PX,PY), PW, PH, boxstyle="round,pad=0.1",
             linewidth=1.5, edgecolor="#4466AA", facecolor="#0D0D20",
             alpha=0.96, zorder=1))
ax.text(PX+PW/2, PY+PH-0.22, "NAFBlock  (internal detail)",
        ha="center", va="top", fontsize=10.5,
        color="white", fontweight="bold", zorder=6)

IW, IH = 1.9, 0.52
IGAP    = 0.18
start_y = PY+PH-0.60

def draw_column(items, cx, title, tc):
    ax.text(cx+IW/2, start_y, title, ha="center", va="top",
            fontsize=8.2, color=tc, fontstyle="italic", zorder=6)
    centers = []
    y = start_y - 0.42
    for lbl, fc in items:
        ec = "#888899"
        p = FancyBboxPatch((cx, y-IH), IW, IH, boxstyle="round,pad=0.05",
                           linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=3)
        ax.add_patch(p)
        ax.text(cx+IW/2, y-IH/2, lbl, ha="center", va="center",
                fontsize=7.0, color="white", fontweight="bold", zorder=4)
        cy = y-IH/2
        centers.append((cx+IW/2, cy))
        y -= IH+IGAP
    for i in range(len(centers)-1):
        x0,y0 = centers[i]
        x1,y1 = centers[i+1]
        ax.annotate("", xy=(x1,y1+IH/2), xytext=(x0,y0-IH/2),
                    arrowprops=dict(arrowstyle="-|>", color="#AAAAAA",
                                   lw=0.9, mutation_scale=9), zorder=5)
    return centers

sb1 = [
    ("LayerNorm 2D",          "#3B3B6B"),
    ("Conv 1×1  C → 2C",      "#1F5FA6"),
    ("DWConv 3×3  (groups=2C)","#1F5FA6"),
    ("SimpleGate  2C → C",    "#007070"),
    ("SCA  (Avg+Conv1×1)",    "#8B5000"),
    ("Conv 1×1  C → C",       "#1F5FA6"),
    ("× β   +  residual",     "#1A6B1A"),
]

sb2 = [
    ("LayerNorm 2D",          "#3B3B6B"),
    ("Conv 1×1  C → 4C",      "#1F5FA6"),
    ("SimpleGate  4C → 2C",   "#007070"),
    ("Conv 1×1  2C → C",      "#1F5FA6"),
    ("× γ   +  residual",     "#1A6B1A"),
]

cx1 = PX + 0.3
cx2 = PX + 0.3 + IW + 0.8

draw_column(sb1, cx1, "Sub-block 1  (Spatial + SCA)", "#88BBFF")
draw_column(sb2, cx2, "Sub-block 2  (FFN)", "#FFDD88")

ax.text(PX+PW/2, PY+0.48,
        "Input  →  Sub-block 1  →  Sub-block 2  →  Output",
        ha="center", va="center", fontsize=8, color="#CCCCCC",
        style="italic", zorder=6)
ax.text(PX+PW/2, PY+0.16,
        "SimpleGate: splits channel dim in half, element-wise product  |  "
        "SCA: Simplified Channel Attention (global avg-pool + Conv1×1 scale)",
        ha="center", va="center", fontsize=6.5, color="#9999BB", zorder=6)

# ═══════════════════════════════════════════════════════════════════════════════
# Legend panel (lower-right)
# ═══════════════════════════════════════════════════════════════════════════════
LX, LY, LW, LH = fig_w-5.5, 0.10, 5.2, 5.2
ax.add_patch(FancyBboxPatch((LX,LY), LW, LH, boxstyle="round,pad=0.1",
             linewidth=1.5, edgecolor="#4466AA", facecolor="#0D0D20",
             alpha=0.96, zorder=1))
ax.text(LX+LW/2, LY+LH-0.22, "Legend", ha="center", va="top",
        fontsize=10.5, color="white", fontweight="bold", zorder=6)

legend_entries = [
    (C_INPUT,   "Input / Output  (2-ch complex)"),
    (C_INTRO,   "Intro / Ending  Conv 3×3"),
    (C_ENC,     "Encoder stages  (NAFBlocks)"),
    (C_MIDDLE,  "Bottleneck  (NAFBlocks)"),
    (C_DEC,     "Decoder stages  (NAFBlocks)"),
    (C_SKIP,    "Skip connection  (additive +)"),
    (C_RESID,   "Global residual  (out=net(x)+x)"),
    ("#33DD77",  "Stride-2 Conv 2×2  (down)"),
    ("#FF9944",  "PixelShuffle ×2  (up)"),
]

for i, (fc, lbl) in enumerate(legend_entries):
    yi = LY+LH-0.62 - i*0.48
    ax.add_patch(mpatches.FancyBboxPatch((LX+0.18, yi-0.15), 0.55, 0.30,
                 boxstyle="round,pad=0.03",
                 linewidth=1.0, edgecolor="#777788", facecolor=fc, zorder=6))
    ax.text(LX+0.85, yi, lbl, va="center", ha="left",
            fontsize=7.8, color="#DDDDDD", zorder=6)

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(fig_w/2, 16.80,
        "NAFNet-Small  ─  DLDeGibbs  Architecture",
        ha="center", va="center",
        fontsize=15, color="white", fontweight="bold", zorder=6)
ax.text(fig_w/2, 16.50,
        f"width=32  |  enc_blks=[2,2,4,8]  |  middle_blks=12  "
        f"|  dec_blks=[2,2,2,2]  |  in_ch=2  out_ch=2",
        ha="center", va="center",
        fontsize=9, color="#AAAACC", zorder=6)

out_path = "/home/raja/xe129_dldegibbs/nafnet_small_architecture.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
plt.close(fig)
