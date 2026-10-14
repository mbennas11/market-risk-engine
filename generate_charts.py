"""Generate the README / LinkedIn visuals. Usage: python generate_charts.py [output_dir]"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm
from arch import arch_model

OUT = sys.argv[1] if len(sys.argv) > 1 else "images"

# Light charts
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# Dark cards (cover, conclusion, banner)
BG, CARD, EDGE, WHITE, SOFT = "#101418", "#1b2128", "#2a323b", "#ffffff", "#a9b1ba"
BLUE_D, ORANGE_D = "#3987e5", "#d95926"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 16, "text.parse_math": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})

# ---------------------------------------------------------------- data & risk numbers
tickers = ["SPY", "QQQ", "VGK", "TLT", "GLD"]
names = ["Actions US", "Tech US", "Actions Europe", "Obligations US", "Or"]
w = np.array([.30, .20, .15, .20, .15])
V = 1_000_000
END = "2026-10-14"  # exclusive: data through the 13/10/2026 close, so figures are reproducible
raw = yf.download(tickers, start="2020-01-01", end=END, auto_adjust=True, progress=False)
prices = raw["Close"].reindex(columns=tickers).ffill().dropna()
r = prices.pct_change().dropna()
p = r.dot(w)
alpha = 0.01
mu, sd = p.mean(), p.std()
z = norm.ppf(alpha)

# Same steps and seed as market_risk_engine.py
np.random.seed(42)
sim = np.random.multivariate_normal(r.mean().values, r.cov().values, 100_000) @ w
h_var = -np.quantile(p, alpha)
h_es = -p[p <= -h_var].mean()
g_var = -(mu + z * sd)
g_es = -(mu - sd * norm.pdf(z) / alpha)
mc_q = np.quantile(sim, alpha)
mc_var, mc_es = -mc_q, -sim[sim <= mc_q].mean()

fvol = []
for t in tickers:
    res = arch_model(r[t] * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="normal",
                     rescale=False).fit(disp="off")
    fvol.append(np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0]) / 100)
D = np.diag(fvol)
garch_var = -(mu + z * np.sqrt(w @ D @ r.corr().values @ D @ w))

cov = r.cov().values
rc = w * (cov @ w) / (w @ cov @ w)

sc = {
    "Crise de liquidité": [-.30, -.35, -.32, -.08, -.05],
    "Récession mondiale": [-.25, -.30, -.28, .10, .12],
    "Krach actions": [-.20, -.28, -.22, .05, .08],
    "Choc d'inflation": [-.12, -.18, -.10, -.18, .15],
    "Choc de taux": [-.08, -.12, -.07, -.15, -.05],
}
pnl = {k: np.dot(w, v) * V for k, v in sc.items()}
worst = min(pnl.values())

port = (1 + p).cumprod()
max_dd = (port / port.cummax() - 1).min()

pv = p * 100
fit = arch_model(pv, vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
cv = fit.conditional_volatility / 100 * np.sqrt(252)


def fr(x, d=2):
    return f"{x*100:.{d}f}".replace(".", ",") + " %"


def usd(x):
    return f"{x:,.0f}".replace(",", " ") + " $"


PERIOD = f"données {r.index[0]:%m/%Y}–{r.index[-1]:%m/%Y}"


# ---------------------------------------------------------------- light chart template
def chart(title, sub, left=0.08):
    fig = plt.figure(figsize=(10, 10))
    lines = title.count("\n") + 1
    sub_y = 0.95 - 0.052 * lines - 0.025
    fig.text(0.06, 0.95, title, fontsize=30, weight="bold", color=INK, va="top", linespacing=1.15)
    fig.text(0.06, sub_y, sub, fontsize=17, color=INK2, va="top", linespacing=1.4)
    fig.text(0.06, 0.025, f"Portefeuille multi-actifs 1 M$ · {PERIOD}", fontsize=12, color=MUTED)
    top = sub_y - 0.13
    return fig, fig.add_axes((left, 0.10, 0.94 - left, top - 0.10))


def save(fig, name, bg=SURF, dpi=120):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, facecolor=bg)
    plt.close(fig)


# ---------------------------------------------------------------- dark card template
def dark(figsize=(10, 10), w_=1000, h_=1000):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w_); ax.set_ylim(0, h_); ax.axis("off")
    return fig, ax


def card(ax, x, y, cw, ch, accent):
    ax.add_patch(FancyBboxPatch((x, y), cw, ch, boxstyle="round,pad=0,rounding_size=16",
                                fc=CARD, ec=EDGE, lw=1.2))
    ax.add_patch(FancyBboxPatch((x, y + ch - 7), cw, 7, boxstyle="round,pad=0,rounding_size=3",
                                fc=accent, ec="none"))


def vol_trace(ax, w_, height, base=0):
    vol = cv.resample("W").mean().values
    xs = np.linspace(0, w_, len(vol))
    ys = base + (vol - vol.min()) / (vol.max() - vol.min()) * height
    ax.fill_between(xs, 0, ys, color=BLUE_D, alpha=0.12, lw=0)
    ax.plot(xs, ys, color=BLUE_D, alpha=0.5, lw=1.6)


# 1. Cover ---------------------------------------------------------------------------
fig, ax = dark()
vol_trace(ax, 1000, 70, 0)
ax.text(64, 920, "PROJET PYTHON  ·  RISQUE DE MARCHÉ", color=SOFT, fontsize=14, weight="bold")
ax.text(64, 870, "Combien un portefeuille\nde 1 M$ peut-il perdre\nen un jour ?", color=WHITE,
        fontsize=40, weight="bold", va="top", linespacing=1.15)
ax.text(64, 610, "4 méthodes de VaR (99 %), 4 réponses différentes :", color=SOFT, fontsize=19)
tiles = [("Historique", h_var), ("Paramétrique", g_var), ("Monte Carlo", mc_var), ("GARCH", garch_var)]
cw, ch, gx, gy = 418, 170, 36, 30
for i, (lab, v) in enumerate(tiles):
    x = 64 + (i % 2) * (cw + gx)
    y = 380 - (i // 2) * (ch + gy)
    card(ax, x, y, cw, ch, ORANGE_D if i == 0 else BLUE_D)
    ax.text(x + 28, y + 100, "−" + usd(v * V), color=WHITE, fontsize=36, weight="bold")
    ax.text(x + 28, y + 42, lab, color=SOFT, fontsize=17)
ax.text(936, 115, "Laquelle croire ?  →", color=WHITE, fontsize=22, weight="bold", ha="right")
save(fig, "1_couverture.png", BG, dpi=120)

# 2. Methods comparison ---------------------------------------------------------------
fig, ax = chart("La loi normale sous-estime\nles pertes extrêmes",
                f"4 méthodes, 4 réponses pour la perte à 1 jour (99 %) sur 1 M$.\n"
                f"L'Expected Shortfall historique est {fr(h_es / g_es - 1, 0)} au-dessus du gaussien.",
                left=0.20)
meths = ["Historique", "Paramétrique\ngaussienne", "Monte Carlo\ngaussien", "GARCH(1,1)"]
vars_ = [h_var, g_var, mc_var, garch_var]
ess = [h_es, g_es, mc_es, np.nan]
y = np.arange(4)[::-1]
hgt = 0.36
ax.barh(y + hgt / 2 + 0.02, np.array(vars_) * V, height=hgt, color=BLUE, label="VaR")
ax.barh(y - hgt / 2 - 0.02, np.nan_to_num(ess) * V, height=hgt, color=ORANGE, label="Expected Shortfall (CVaR)")
for yi, v, e in zip(y, vars_, ess):
    ax.text(v * V + 500, yi + hgt / 2, usd(v * V), va="center", fontsize=15, color=INK)
    if not np.isnan(e):
        ax.text(e * V + 500, yi - hgt / 2, usd(e * V), va="center", fontsize=15, color=INK, weight="bold")
ax.set_yticks(y, meths, fontsize=16, color=INK)
ax.tick_params(axis="y", length=0)
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v/1e3:.0f} k$"))
ax.set_xlim(0, h_es * V * 1.3)
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right", fontsize=15)
save(fig, "2_methodes.png")

# 3. GARCH volatility -----------------------------------------------------------------
fig, ax = chart("La volatilité arrive par vagues",
                f"Volatilité GARCH(1,1) annualisée. Le marché est calme aujourd'hui,\n"
                f"d'où une VaR GARCH plus basse ({usd(garch_var * V)}) que la VaR historique.")
ax.plot(cv.index, cv.values, color=BLUE, lw=2)
ax.axhline(sd * np.sqrt(252), color=MUTED, lw=1.5, ls="--", label=f"Moyenne historique ({fr(sd * np.sqrt(252), 0)})")
ax.legend(frameon=False, loc="upper right", fontsize=15)
peak = cv.idxmax()
ax.annotate(f"Covid · {fr(cv.max(), 0)}", (peak, cv.max()), xytext=(40, -10), textcoords="offset points",
            color=INK, fontsize=15, weight="bold", arrowprops=dict(arrowstyle="-", color=MUTED))
apr = cv["2025-03-01":"2025-06-30"].idxmax()
ax.annotate(f"Avril 2025 · {fr(cv[apr], 0)}", (apr, cv[apr]), xytext=(-175, 10), textcoords="offset points",
            color=INK, fontsize=15, weight="bold", arrowprops=dict(arrowstyle="-", color=MUTED))
ax.scatter([cv.index[-1]], [cv.iloc[-1]], color=ORANGE, s=70, zorder=3)
ax.annotate(f"Aujourd'hui · {fr(cv.iloc[-1], 0)}", (cv.index[-1], cv.iloc[-1]), xytext=(-10, 150),
            textcoords="offset points", color=ORANGE, fontsize=15, weight="bold", ha="right",
            arrowprops=dict(arrowstyle="-", color=ORANGE))
ax.yaxis.set_major_formatter(mt.PercentFormatter(1, decimals=0))
save(fig, "3_volatilite_garch.png")

# 4. Weight vs risk contribution ------------------------------------------------------
fig, ax = chart("Diversifié en capital,\npas en risque",
                f"Les actions pèsent 65 % du portefeuille\nmais portent {fr(rc[:3].sum(), 0)} du risque.",
                left=0.25)
y = np.arange(len(names))[::-1]
hgt = 0.36
ax.barh(y + hgt / 2 + 0.02, w, height=hgt, color="#c9c8c3", label="Poids dans le portefeuille")
ax.barh(y - hgt / 2 - 0.02, rc, height=hgt, color=BLUE, label="Part du risque")
for yi, wi, ri in zip(y, w, rc):
    ax.text(wi + 0.006, yi + hgt / 2, fr(wi, 0), va="center", fontsize=14, color=INK2)
    ax.text(ri + 0.006, yi - hgt / 2, fr(ri, 0), va="center", fontsize=15, color=INK, weight="bold")
ax.set_yticks(y, names, fontsize=16, color=INK)
ax.tick_params(axis="y", length=0)
ax.xaxis.set_major_formatter(mt.PercentFormatter(1, decimals=0))
ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(rc.max(), w.max()) * 1.18)
ax.legend(frameon=False, loc="lower right", fontsize=15)
save(fig, "4_contribution_risque.png")

# 5. Stress tests ---------------------------------------------------------------------
fig, ax = chart("En crise, tout baisse ensemble",
                f"Stress tests sur 1 M$. La crise de liquidité coûte\n"
                f"{-worst / (h_var * V):.0f} fois la VaR historique journalière ({usd(h_var * V)}).",
                left=0.28)
ks = list(pnl)[::-1]
vals = [-pnl[k] for k in ks]
cols = [ORANGE if k == "Crise de liquidité" else BLUE for k in ks]
ax.barh(ks, vals, color=cols, height=0.6)
for i, v in enumerate(vals):
    ax.text(v + 3000, i, f"−{usd(v)}", va="center", fontsize=15, color=INK,
            weight="bold" if i == len(vals) - 1 else "normal")
ax.axvline(h_var * V, color=INK, lw=1.5, ls="--")
ax.set_ylim(-0.6, 4.95)
ax.text(h_var * V + 3000, 4.55, "VaR 1 jour", fontsize=14, color=INK2)
ax.tick_params(axis="y", length=0, labelsize=16, labelcolor=INK)
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v/1e3:.0f} k$"))
ax.set_xlim(0, max(vals) * 1.4)
ax.grid(axis="y", visible=False)
save(fig, "5_stress_tests.png")

# 6. Conclusion -----------------------------------------------------------------------
fig, ax = dark()
ax.text(64, 920, "CONCLUSION", color=SOFT, fontsize=14, weight="bold")
ax.text(64, 870, "Aucune mesure ne suffit seule.\nUn risk manager en suit trois :", color=WHITE,
        fontsize=34, weight="bold", va="top", linespacing=1.2)
rows = [
    ("Expected Shortfall historique", "le risque extrême", usd(h_es * V), ORANGE_D),
    ("VaR GARCH", "le risque du moment", usd(garch_var * V), BLUE_D),
    ("Stress tests", "ce que l'historique n'a pas encore connu", "−" + usd(-worst), BLUE_D),
]
ch, gy, y0 = 170, 30, 530
for i, (t, sub, val, col) in enumerate(rows):
    y = y0 - i * (ch + gy)
    card(ax, 64, y, 872, ch, col)
    ax.text(100, y + 100, t, color=WHITE, fontsize=22, weight="bold")
    ax.text(100, y + 45, sub, color=SOFT, fontsize=17)
    ax.text(900, y + 72, val, color=WHITE, fontsize=28, weight="bold", ha="right")
ax.text(64, 60, "Code complet : github.com/mbennas11/market-risk-engine", color=SOFT, fontsize=16)
save(fig, "6_conclusion.png", BG, dpi=120)

# 0. Banner (README header + GitHub social preview, 1280x640) -------------------------
fig, ax = dark(figsize=(12.8, 6.4), w_=1280, h_=640)
vol_trace(ax, 1280, 150, 40)
ax.text(64, 560, "PYTHON  ·  RISQUE DE MARCHÉ  ·  PORTEFEUILLE MULTI-ACTIFS", color=SOFT, fontsize=12,
        weight="bold")
ax.text(64, 490, "Market Risk Engine", color=WHITE, fontsize=46, weight="bold")
ax.text(64, 440, "Combien un portefeuille de 1 M$ peut-il perdre en une journée ?", color=SOFT, fontsize=18)
tiles = [
    (usd(h_var * V), "VaR 99 % historique", BLUE_D),
    (usd(h_es * V), "Expected Shortfall 99 %", BLUE_D),
    (fr(rc[:3].sum(), 0), "du risque vient des actions", BLUE_D),
    ("−" + usd(-worst), "pire stress test", ORANGE_D),
]
tw, th, gap, x0, y0 = 270, 150, 20, 64, 230
for i, (val, lab, col) in enumerate(tiles):
    x = x0 + i * (tw + gap)
    card(ax, x, y0, tw, th, col)
    ax.text(x + 24, y0 + 72, val, color=WHITE, fontsize=27, weight="bold")
    ax.text(x + 24, y0 + 30, lab, color=SOFT, fontsize=13)
ax.text(1216, 20, "VaR · CVaR · Monte Carlo · GARCH(1,1) · Stress tests", color=SOFT, fontsize=11, ha="right")
save(fig, "0_banner.png", BG, dpi=100)

print(f"VaR hist {h_var:.4%} ES {h_es:.4%} | gauss {g_var:.4%} {g_es:.4%} | MC {mc_var:.4%} {mc_es:.4%} | "
      f"GARCH {garch_var:.4%} | RC eq {rc[:3].sum():.3f} | worst {worst:,.0f} | maxDD {max_dd:.4f}")
