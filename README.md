# Mine-style Credit Builder Simulation Project

This project simulates how two credit products affect repayment behavior, liquidity stress, default risk, and long-term credit-building outcomes.

## What it models
- synthetic users with different income volatility, spending behavior, credit limits, and repayment discipline
- daily cash flow over a 1-year horizon
- random financial shocks
- two product designs:
  - Traditional Revolving Card
  - Daily-Pay Credit Builder
- outcomes:
  - default rate
  - utilization
  - cumulative interest and fees
  - FICO-style proxy score

## Why this is useful in an interview
It shows:
- Monte Carlo simulation
- probabilistic user behavior modeling
- product experimentation mindset
- consumer finance intuition
- ability to convert simulations into dashboards and product insights

## Project structure
- `src/main.py` — full simulation pipeline
- `outputs/` — generated CSVs and HTML dashboards after execution

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python src/main.py --users 5000 --days 365 --seed 42 --outdir outputs
```

## Output files
- `synthetic_population.csv`
- `simulation_user_metrics.csv`
- `simulation_daily_snapshots.csv`
- `summary.csv`
- `dashboard/fico_trajectory.html`
- `dashboard/default_rate.html`
- `dashboard/fico_distribution.html`
- `dashboard/summary_table.html`

## Interview explanation flow
1. Define the problem: traditional revolving debt can hurt young users with volatile income.
2. Explain Monte Carlo: simulate thousands of possible user paths instead of one fixed scenario.
3. Explain user state: income, spending, shocks, cash, balance, utilization, missed payments.
4. Explain product comparison: monthly revolving vs daily-pay structure.
5. Explain outputs: default rate, FICO proxy, interest, fees, net financial health.
6. Explain insight: more frequent repayment can lower utilization and reduce long-term risk.

## Possible extensions
- calibrate behavior with real transaction data
- train a survival/default model on simulated outcomes
- segment results by paycheck cadence or income volatility
- add underwriting / personalized limit assignment
- turn HTML outputs into a Streamlit dashboard
