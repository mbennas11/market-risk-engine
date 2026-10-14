# ============================================================
# MARKET RISK ENGINE - MULTI-ASSET PORTFOLIO
# VaR | CVaR | Monte Carlo | GARCH | Stress Testing
# ============================================================

# ---------- 0. INSTALLATION ----------
import sys
import subprocess
import importlib.util
import warnings

warnings.filterwarnings("ignore")

required_packages = {
    "yfinance": "yfinance",
    "arch": "arch"
}

for module, package in required_packages.items():
    if importlib.util.find_spec(module) is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package]
        )

# ---------- 1. IMPORTS ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.stats import norm
from arch import arch_model

np.random.seed(42)

# ============================================================
# 2. PORTFOLIO PARAMETERS
# ============================================================

# Multi-asset portfolio:
# US Equity / Tech / Europe / Bonds / Gold
tickers = ["SPY", "QQQ", "VGK", "TLT", "GLD"]

asset_names = {
    "SPY": "US Equity",
    "QQQ": "US Technology",
    "VGK": "European Equity",
    "TLT": "US Treasury Bonds",
    "GLD": "Gold"
}

# Portfolio weights
weights = np.array([
    0.30,   # SPY
    0.20,   # QQQ
    0.15,   # VGK
    0.20,   # TLT
    0.15    # GLD
])

weights = weights / weights.sum()

portfolio_value = 1_000_000       # $1m portfolio
confidence_level = 0.99           # 99% VaR
alpha = 1 - confidence_level
trading_days = 252
mc_simulations = 100_000

start_date = "2020-01-01"
end_date = None

print("=" * 70)
print("MULTI-ASSET MARKET RISK ENGINE")
print("=" * 70)

portfolio_table = pd.DataFrame({
    "Asset": [asset_names[t] for t in tickers],
    "Ticker": tickers,
    "Weight": weights
})

print("\nPORTFOLIO")
print(portfolio_table.to_string(index=False))
print(f"\nPortfolio value: ${portfolio_value:,.0f}")
print(f"VaR confidence level: {confidence_level:.0%}")

# ============================================================
# 3. DOWNLOAD MARKET DATA
# ============================================================

print("\nDownloading market data...")

raw = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)

# yfinance can return slightly different column structures
if isinstance(raw.columns, pd.MultiIndex):
    if "Close" in raw.columns.get_level_values(0):
        prices = raw["Close"].copy()
    else:
        prices = raw.xs("Close", axis=1, level=0).copy()
else:
    prices = raw[["Close"]].copy()
    prices.columns = tickers[:1]

# Ensure desired order
prices = prices.reindex(columns=tickers)

# Handle occasional missing observations
prices = prices.ffill().dropna()

if prices.empty:
    raise ValueError("No market data downloaded. Check internet connection/tickers.")

print(
    f"Data: {prices.index.min().date()} -> "
    f"{prices.index.max().date()}"
)
print(f"Observations: {len(prices):,}")

# ============================================================
# 4. RETURNS
# ============================================================

returns = prices.pct_change().dropna()

portfolio_returns = returns.dot(weights)
portfolio_returns.name = "Portfolio Return"

# Basic statistics
mean_daily = portfolio_returns.mean()
vol_daily = portfolio_returns.std()

annual_return = mean_daily * trading_days
annual_volatility = vol_daily * np.sqrt(trading_days)

print("\n" + "=" * 70)
print("PORTFOLIO STATISTICS")
print("=" * 70)

print(f"Annualized expected return : {annual_return:.2%}")
print(f"Annualized volatility      : {annual_volatility:.2%}")
print(f"Worst daily return         : {portfolio_returns.min():.2%}")
print(f"Best daily return          : {portfolio_returns.max():.2%}")

# ============================================================
# 5. HISTORICAL VaR & CVaR
# ============================================================

historical_quantile = np.quantile(portfolio_returns, alpha)

historical_var_pct = -historical_quantile

tail_returns = portfolio_returns[
    portfolio_returns <= historical_quantile
]

historical_cvar_pct = -tail_returns.mean()

historical_var_money = historical_var_pct * portfolio_value
historical_cvar_money = historical_cvar_pct * portfolio_value

# ============================================================
# 6. PARAMETRIC VaR & CVaR
# Variance-Covariance / Gaussian model
# ============================================================

mu = portfolio_returns.mean()
sigma = portfolio_returns.std()

z_alpha = norm.ppf(alpha)

parametric_quantile = mu + z_alpha * sigma

parametric_var_pct = -parametric_quantile

# Expected Shortfall for Gaussian distribution
parametric_cvar_pct = -(
    mu - sigma * norm.pdf(z_alpha) / alpha
)

parametric_var_money = parametric_var_pct * portfolio_value
parametric_cvar_money = parametric_cvar_pct * portfolio_value

# ============================================================
# 7. MONTE CARLO VaR & CVaR
# Multivariate Normal simulation
# ============================================================

mean_vector = returns.mean().values
cov_matrix = returns.cov().values

simulated_asset_returns = np.random.multivariate_normal(
    mean=mean_vector,
    cov=cov_matrix,
    size=mc_simulations
)

simulated_portfolio_returns = (
    simulated_asset_returns @ weights
)

mc_quantile = np.quantile(
    simulated_portfolio_returns,
    alpha
)

mc_var_pct = -mc_quantile

mc_tail = simulated_portfolio_returns[
    simulated_portfolio_returns <= mc_quantile
]

mc_cvar_pct = -mc_tail.mean()

mc_var_money = mc_var_pct * portfolio_value
mc_cvar_money = mc_cvar_pct * portfolio_value

# ============================================================
# 8. GARCH(1,1) VOLATILITY MODELLING
# ============================================================

print("\n" + "=" * 70)
print("GARCH(1,1) VOLATILITY MODELLING")
print("=" * 70)

garch_results = {}
garch_forecast_vol = {}

for ticker in tickers:

    # ARCH works better with returns expressed in %
    series = returns[ticker].dropna() * 100

    model = arch_model(
        series,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False
    )

    result = model.fit(disp="off")

    forecast = result.forecast(horizon=1)

    variance_forecast = forecast.variance.iloc[-1, 0]

    daily_vol_forecast = np.sqrt(variance_forecast) / 100

    garch_results[ticker] = result
    garch_forecast_vol[ticker] = daily_vol_forecast

    print(
        f"{ticker:5s} | "
        f"Forecast daily vol: {daily_vol_forecast:.3%} | "
        f"Annualized: {daily_vol_forecast*np.sqrt(252):.2%}"
    )

# ============================================================
# 9. PORTFOLIO GARCH VOLATILITY
# ============================================================

# Historical correlation matrix
correlation_matrix = returns.corr().values

# Forecast individual volatilities
forecast_vol_vector = np.array([
    garch_forecast_vol[t] for t in tickers
])

# D * Correlation * D
D = np.diag(forecast_vol_vector)

forecast_covariance_matrix = (
    D @ correlation_matrix @ D
)

portfolio_garch_variance = (
    weights.T
    @ forecast_covariance_matrix
    @ weights
)

portfolio_garch_vol = np.sqrt(
    portfolio_garch_variance
)

portfolio_garch_annual_vol = (
    portfolio_garch_vol * np.sqrt(trading_days)
)

# Dynamic VaR using GARCH forecast
garch_var_pct = -(
    mu + z_alpha * portfolio_garch_vol
)

garch_var_money = (
    garch_var_pct * portfolio_value
)

print("\nPortfolio GARCH forecast")
print(f"Daily volatility     : {portfolio_garch_vol:.3%}")
print(f"Annualized volatility: {portfolio_garch_annual_vol:.2%}")
print(f"GARCH 99% VaR        : {garch_var_pct:.2%}")
print(f"GARCH 99% VaR ($)    : ${garch_var_money:,.0f}")

# ============================================================
# 10. STRESS TESTING
# ============================================================

# Hypothetical extreme market scenarios
stress_scenarios = {
    "Equity Crash": {
        "SPY": -0.20,
        "QQQ": -0.28,
        "VGK": -0.22,
        "TLT": +0.05,
        "GLD": +0.08
    },

    "Interest Rate Shock": {
        "SPY": -0.08,
        "QQQ": -0.12,
        "VGK": -0.07,
        "TLT": -0.15,
        "GLD": -0.05
    },

    "Global Recession": {
        "SPY": -0.25,
        "QQQ": -0.30,
        "VGK": -0.28,
        "TLT": +0.10,
        "GLD": +0.12
    },

    "Inflation Shock": {
        "SPY": -0.12,
        "QQQ": -0.18,
        "VGK": -0.10,
        "TLT": -0.18,
        "GLD": +0.15
    },

    "Liquidity Crisis": {
        "SPY": -0.30,
        "QQQ": -0.35,
        "VGK": -0.32,
        "TLT": -0.08,
        "GLD": -0.05
    }
}

stress_results = []

for scenario_name, shocks in stress_scenarios.items():

    shock_vector = np.array([
        shocks[ticker]
        for ticker in tickers
    ])

    portfolio_shock = weights @ shock_vector

    pnl = portfolio_value * portfolio_shock

    stress_results.append([
        scenario_name,
        portfolio_shock,
        pnl,
        portfolio_value + pnl
    ])

stress_df = pd.DataFrame(
    stress_results,
    columns=[
        "Scenario",
        "Portfolio Return",
        "P&L",
        "Portfolio Value After Stress"
    ]
)

# ============================================================
# 11. MAXIMUM DRAWDOWN
# ============================================================

cumulative_returns = (
    1 + portfolio_returns
).cumprod()

running_max = cumulative_returns.cummax()

drawdown = (
    cumulative_returns / running_max
) - 1

max_drawdown = drawdown.min()

# ============================================================
# 12. RISK CONTRIBUTION
# ============================================================

cov = returns.cov().values

portfolio_variance = (
    weights.T @ cov @ weights
)

portfolio_vol = np.sqrt(portfolio_variance)

marginal_risk = (
    cov @ weights
) / portfolio_vol

component_risk = (
    weights * marginal_risk
)

risk_contribution_pct = (
    component_risk / portfolio_vol
)

risk_contribution = pd.DataFrame({
    "Asset": [asset_names[t] for t in tickers],
    "Weight": weights,
    "Risk Contribution": risk_contribution_pct
})

# ============================================================
# 13. RESULTS TABLE
# ============================================================

risk_results = pd.DataFrame({
    "Method": [
        "Historical",
        "Parametric Gaussian",
        "Monte Carlo"
    ],

    "VaR (%)": [
        historical_var_pct,
        parametric_var_pct,
        mc_var_pct
    ],

    "VaR ($)": [
        historical_var_money,
        parametric_var_money,
        mc_var_money
    ],

    "CVaR (%)": [
        historical_cvar_pct,
        parametric_cvar_pct,
        mc_cvar_pct
    ],

    "CVaR ($)": [
        historical_cvar_money,
        parametric_cvar_money,
        mc_cvar_money
    ]
})

print("\n" + "=" * 70)
print(f"RISK RESULTS - {confidence_level:.0%} CONFIDENCE")
print("=" * 70)

display_table = risk_results.copy()

display_table["VaR (%)"] = (
    display_table["VaR (%)"] * 100
).round(3)

display_table["CVaR (%)"] = (
    display_table["CVaR (%)"] * 100
).round(3)

display_table["VaR ($)"] = (
    display_table["VaR ($)"]
    .round(0)
    .map("${:,.0f}".format)
)

display_table["CVaR ($)"] = (
    display_table["CVaR ($)"]
    .round(0)
    .map("${:,.0f}".format)
)

print(display_table.to_string(index=False))

# ============================================================
# 14. STRESS TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STRESS TEST RESULTS")
print("=" * 70)

stress_display = stress_df.copy()

stress_display["Portfolio Return"] = (
    stress_display["Portfolio Return"] * 100
).round(2).astype(str) + "%"

stress_display["P&L"] = (
    stress_display["P&L"]
    .round(0)
    .map("${:,.0f}".format)
)

stress_display["Portfolio Value After Stress"] = (
    stress_display["Portfolio Value After Stress"]
    .round(0)
    .map("${:,.0f}".format)
)

print(stress_display.to_string(index=False))

# ============================================================
# 15. RISK CONTRIBUTION RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RISK CONTRIBUTION BY ASSET")
print("=" * 70)

risk_contribution_display = risk_contribution.copy()

risk_contribution_display["Weight"] = (
    risk_contribution_display["Weight"] * 100
).round(1).astype(str) + "%"

risk_contribution_display["Risk Contribution"] = (
    risk_contribution_display["Risk Contribution"] * 100
).round(1).astype(str) + "%"

print(risk_contribution_display.to_string(index=False))

print(f"\nMaximum Drawdown: {max_drawdown:.2%}")

# ============================================================
# 16. GRAPH 1 - PORTFOLIO PERFORMANCE
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    cumulative_returns.index,
    cumulative_returns.values
)

plt.title(
    "Multi-Asset Portfolio - Cumulative Performance",
    fontsize=14
)

plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 17. GRAPH 2 - RETURN DISTRIBUTION + VaR
# ============================================================

plt.figure(figsize=(12, 5))

plt.hist(
    portfolio_returns,
    bins=80,
    density=True,
    alpha=0.7
)

plt.axvline(
    -historical_var_pct,
    linestyle="--",
    linewidth=2,
    label=f"Historical VaR 99% = {historical_var_pct:.2%}"
)

plt.axvline(
    -parametric_var_pct,
    linestyle="--",
    linewidth=2,
    label=f"Parametric VaR 99% = {parametric_var_pct:.2%}"
)

plt.title("Distribution of Daily Portfolio Returns")
plt.xlabel("Daily Return")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 18. GRAPH 3 - ROLLING VOLATILITY
# ============================================================

rolling_volatility = (
    portfolio_returns
    .rolling(21)
    .std()
    * np.sqrt(252)
)

plt.figure(figsize=(12, 5))

plt.plot(
    rolling_volatility.index,
    rolling_volatility.values
)

plt.title(
    "Portfolio 21-Day Rolling Annualized Volatility"
)

plt.xlabel("Date")
plt.ylabel("Annualized Volatility")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 19. GRAPH 4 - DRAWDOWN
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    drawdown.index,
    drawdown.values
)

plt.fill_between(
    drawdown.index,
    drawdown.values,
    0,
    alpha=0.3
)

plt.title(
    f"Portfolio Drawdown - Maximum Drawdown: {max_drawdown:.2%}"
)

plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 20. GRAPH 5 - GARCH CONDITIONAL VOLATILITY
# ============================================================

portfolio_garch_model = arch_model(
    portfolio_returns * 100,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal",
    rescale=False
)

portfolio_garch_fit = portfolio_garch_model.fit(
    disp="off"
)

conditional_volatility = (
    portfolio_garch_fit.conditional_volatility
    / 100
    * np.sqrt(252)
)

plt.figure(figsize=(12, 5))

plt.plot(
    conditional_volatility.index,
    conditional_volatility.values
)

plt.title(
    "GARCH(1,1) Conditional Portfolio Volatility"
)

plt.xlabel("Date")
plt.ylabel("Annualized Volatility")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 21. GRAPH 6 - MONTE CARLO DISTRIBUTION
# ============================================================

plt.figure(figsize=(12, 5))

plt.hist(
    simulated_portfolio_returns,
    bins=100,
    density=True,
    alpha=0.7
)

plt.axvline(
    -mc_var_pct,
    linestyle="--",
    linewidth=2,
    label=f"Monte Carlo VaR 99% = {mc_var_pct:.2%}"
)

plt.axvline(
    -mc_cvar_pct,
    linestyle=":",
    linewidth=2,
    label=f"Monte Carlo CVaR 99% = {mc_cvar_pct:.2%}"
)

plt.title(
    f"Monte Carlo Portfolio Returns ({mc_simulations:,} simulations)"
)

plt.xlabel("Simulated Daily Return")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 22. FINAL EXECUTIVE RISK REPORT
# ============================================================

print("\n" + "=" * 70)
print("EXECUTIVE MARKET RISK REPORT")
print("=" * 70)

print(f"""
Portfolio Value             : ${portfolio_value:,.0f}

Expected Annual Return      : {annual_return:.2%}
Historical Annual Volatility: {annual_volatility:.2%}
GARCH Forecast Volatility   : {portfolio_garch_annual_vol:.2%}

99% Historical VaR          : ${historical_var_money:,.0f}
99% Historical CVaR         : ${historical_cvar_money:,.0f}

99% Parametric VaR          : ${parametric_var_money:,.0f}
99% Parametric CVaR         : ${parametric_cvar_money:,.0f}

99% Monte Carlo VaR         : ${mc_var_money:,.0f}
99% Monte Carlo CVaR        : ${mc_cvar_money:,.0f}

99% GARCH VaR               : ${garch_var_money:,.0f}

Maximum Historical Drawdown : {max_drawdown:.2%}

Number of Monte Carlo Sims  : {mc_simulations:,}
Number of Assets            : {len(tickers)}
""")

print("=" * 70)
print("END OF MARKET RISK ENGINE")
print("=" * 70)