"""
visual_sanity_check.py
======================
Plot gdpc1 + 5 representative top-35 series from data_tf_monthly.csv.
Catches sign-flip / scale-error bugs that ADF/KPSS pass but eyeball fails.

Eyeball checks per panel:
  - gdpc1            should oscillate around ~0.005 (qoq log diff),
                     huge negative spike in 2020Q2.
  - unrate           first diff of unemployment %, near 0 typically,
                     positive spikes in recessions.
  - cpiaucsl         monthly log diff (~0.002-0.005), oil shocks visible.
  - houstne          monthly log diff of NE housing starts, very volatile.
  - m2sl             monthly log diff of M2 money supply, smooth positive.
  - compapff         CP-FF spread in level, swings in financial stress.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = "C:/Users/asus/Desktop/nowcasting_benchmark-main/nowcasting_benchmark-main/data"
df = pd.read_csv(os.path.join(BASE, "data_tf_monthly.csv"), parse_dates=["date"])

panels = [
    ("gdpc1",     "Real GDP qoq log diff"),
    ("unrate",    "Unemployment first diff"),
    ("cpiaucsl",  "CPI monthly log diff"),
    ("houstne",   "Housing Starts NE log diff"),
    ("m2sl",      "M2 monthly log diff"),
    ("compapff",  "CP-FF spread (level)"),
]

fig, axes = plt.subplots(3, 2, figsize=(14, 9))
for ax, (col, title) in zip(axes.flat, panels):
    if col not in df.columns:
        ax.set_title(f"{title} -- MISSING")
        continue
    s = df[["date", col]].dropna()
    ax.plot(s["date"], s[col], lw=0.8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(f"{col} -- {title}")
    ax.grid(alpha=0.3)

fig.suptitle("Visual sanity check on transformed series", fontsize=14)
fig.tight_layout()
out_path = os.path.join(BASE, "visual_sanity_check.png")
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"Saved {out_path}")
