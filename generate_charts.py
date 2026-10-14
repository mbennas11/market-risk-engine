import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from scipy.stats import norm
from arch import arch_model

OUT = sys.argv[1] if len(sys.argv) > 1 else "images"

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "text.parse_math": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
})

tickers = ["SPY", "QQQ", "VGK", "TLT", "GLD"]
names = ["Actions US", "Tech US", "Actions Europe", "Obligations US", "Or"]
w = np.array([.30, .20, .15, .20, .15])
raw = yf.download(tickers, start="2020-01-01", auto_adjust=True, progress=False)
prices = raw["Close"].reindex(columns=tickers).ffill().dropna()
r = prices.pct_change().dropna()
p = r.dot(w)
alpha = 0.01
mu, sd = p.mean(), p.std()
h_var = -np.quantile(p, alpha)
g_var = -(mu + norm.ppf(alpha) * sd)
h_cvar = -p[p <= -h_var].mean()
g_cvar = -(mu - sd * norm.pdf(norm.ppf(alpha)) / alpha)


def fr(x, d=2):
    return f"{x*100:.{d}f}".replace(".", ",") + " %"


def header(fig, title, sub):
    fig.text(0.06, 0.95, title, fontsize=24, weight="bold", color=INK, va="top")
    fig.text(0.06, 0.895, sub, fontsize=15, color=INK2, va="top")


def footer(fig):
    fig.text(0.06, 0.025, f"Portefeuille multi-actifs 1 M$ · données {r.index[0]:%m/%Y}–{r.index[-1]:%m/%Y}",
             fontsize=12, color=MUTED)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=120, facecolor=SURF)
    plt.close(fig)


# 1. Fat tails: historical vs gaussian
fig = plt.figure(figsize=(10, 10))
header(fig, "Les pertes extrêmes sont plus fréquentes\nque ne le prévoit la loi normale",
       "")
fig.texts[-1].set_text("")
fig.text(0.06, 0.845, f"VaR 99 % historique {fr(h_var)} vs gaussienne {fr(g_var)}  ·  "
         f"CVaR {fr(h_cvar)} vs {fr(g_cvar)}", fontsize=15, color=INK2, va="top")
ax = fig.add_axes([0.1, 0.12, 0.85, 0.66])
ax.hist(p, bins=90, density=True, color=BLUE, alpha=0.85, edgecolor=SURF, linewidth=0.6,
        label="Rendements journaliers observés")
x = np.linspace(p.min(), p.max(), 500)
ax.plot(x, norm.pdf(x, mu, sd), color=ORANGE, lw=2.5, label="Loi normale (même moyenne, même vol)")
ax.axvline(-h_var, color=INK, lw=2, ls="--")
ax.axvline(-g_var, color=ORANGE, lw=2, ls="--")
ymax = ax.get_ylim()[1]
ax.text(-h_var - 0.001, ymax * 0.62, f"VaR historique\n−{h_var:.2%}", ha="right", color=INK, fontsize=13)
ax.text(-g_var + 0.001, ymax * 0.80, f"VaR gaussienne\n−{g_var:.2%}", ha="left", color=INK, fontsize=13)
ax.xaxis.set_major_formatter(mt.PercentFormatter(1, decimals=0))
ax.set_xlabel("Rendement journalier du portefeuille")
ax.set_ylabel("Densité")
ax.legend(frameon=False, loc="upper right", fontsize=13)
footer(fig)
save(fig, "1_queues_epaisses.png")

# 2. GARCH conditional volatility
fit = arch_model(p * 100, vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
cv = fit.conditional_volatility / 100 * np.sqrt(252)
fig = plt.figure(figsize=(10, 10))
header(fig, "La volatilité n'est pas constante :\nelle arrive par vagues",
       "")
fig.texts[-1].set_text("")
fig.text(0.06, 0.845, f"Volatilité conditionnelle GARCH(1,1) annualisée  ·  moyenne historique {fr(sd*np.sqrt(252), 1)}",
         fontsize=15, color=INK2, va="top")
ax = fig.add_axes([0.1, 0.12, 0.85, 0.66])
ax.plot(cv.index, cv.values, color=BLUE, lw=2)
ax.axhline(sd * np.sqrt(252), color=MUTED, lw=1.5, ls="--", label="Volatilité historique moyenne")
ax.legend(frameon=False, loc="upper right", fontsize=13)
apr = cv["2025-03-01":"2025-06-30"].idxmax()
ax.annotate(f"Avril 2025 · {fr(cv[apr], 0)}", (apr, cv[apr]), xytext=(-150, 10), textcoords="offset points",
            color=INK, fontsize=13, arrowprops=dict(arrowstyle="-", color=MUTED))
peak = cv.idxmax()
ax.annotate(f"Covid · {fr(cv.max(), 0)}", (peak, cv.max()), xytext=(40, -10), textcoords="offset points",
            color=INK, fontsize=13, arrowprops=dict(arrowstyle="-", color=MUTED))
ax.yaxis.set_major_formatter(mt.PercentFormatter(1, decimals=0))
ax.set_ylabel("Volatilité annualisée")
footer(fig)
save(fig, "2_volatilite_garch.png")

# 3. Weight vs risk contribution
cov = r.cov().values
pv = np.sqrt(w @ cov @ w)
rc = w * (cov @ w) / pv / pv
fig = plt.figure(figsize=(10, 10))
header(fig, "Le poids d'un actif n'est pas son risque", "")
fig.texts[-1].set_text("")
fig.text(0.06, 0.895, f"Les actions pèsent 65 % du portefeuille mais portent {fr(rc[:3].sum(), 0)} du risque",
         fontsize=15, color=INK2, va="top")
ax = fig.add_axes([0.25, 0.12, 0.7, 0.70])
y = np.arange(len(names))[::-1]
hgt = 0.36
ax.barh(y + hgt / 2 + 0.01, w, height=hgt, color=MUTED, label="Poids")
ax.barh(y - hgt / 2 - 0.01, rc, height=hgt, color=BLUE, label="Contribution au risque")
for yi, wi, ri in zip(y, w, rc):
    ax.text(wi + 0.006, yi + hgt / 2, fr(wi, 0), va="center", fontsize=13, color=INK2)
    ax.text(ri + 0.006, yi - hgt / 2, fr(ri, 0), va="center", fontsize=13, color=INK, weight="bold")
ax.set_yticks(y, names)
ax.xaxis.set_major_formatter(mt.PercentFormatter(1, decimals=0))
ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(rc.max(), w.max()) * 1.15)
ax.legend(frameon=False, loc="lower right", fontsize=13)
footer(fig)
save(fig, "3_contribution_risque.png")

# 4. Stress tests
sc = {
    "Crise de liquidité": [-.30, -.35, -.32, -.08, -.05],
    "Récession mondiale": [-.25, -.30, -.28, .10, .12],
    "Krach actions": [-.20, -.28, -.22, .05, .08],
    "Choc d'inflation": [-.12, -.18, -.10, -.18, .15],
    "Choc de taux": [-.08, -.12, -.07, -.15, -.05],
}
pnl = {k: np.dot(w, v) * 1e6 for k, v in sc.items()}
fig = plt.figure(figsize=(10, 10))
header(fig, "Quand tout baisse ensemble,\nla diversification disparaît", "")
fig.texts[-1].set_text("")
fig.text(0.06, 0.845, f"Stress tests · perte sur 1 M$  ·  VaR 99 % journalière historique : {h_var*1e6/1e3:,.0f} k$",
         fontsize=15, color=INK2, va="top")
ax = fig.add_axes([0.27, 0.12, 0.66, 0.66])
ks = list(pnl)[::-1]
vals = [-pnl[k] for k in ks]
cols = [ORANGE if k == "Crise de liquidité" else BLUE for k in ks]
ax.barh(ks, vals, color=cols, height=0.6)
for i, v in enumerate(vals):
    ax.text(v + 3000, i, f"−{v/1e3:,.0f} k$ ({fr(v/1e6, 1)})", va="center", fontsize=13, color=INK)
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v/1e3:.0f} k$"))
ax.set_xlim(0, max(vals) * 1.35)
ax.grid(axis="y", visible=False)
footer(fig)
save(fig, "4_stress_tests.png")
print("done", h_var, g_var, h_cvar, g_cvar, rc)
