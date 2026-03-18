import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class ProductConfig:
    name: str
    apr: float
    billing_cycle_days: int
    autopay_frequency_days: int
    min_payment_rate: float
    late_fee: float
    utilization_soft_cap: float


TRADITIONAL_CARD = ProductConfig(
    name="Traditional Revolving Card",
    apr=0.24,
    billing_cycle_days=30,
    autopay_frequency_days=30,
    min_payment_rate=0.08,
    late_fee=35.0,
    utilization_soft_cap=0.85,
)

DAILY_PAY_CARD = ProductConfig(
    name="Daily-Pay Credit Builder",
    apr=0.12,
    billing_cycle_days=1,
    autopay_frequency_days=1,
    min_payment_rate=1.00,
    late_fee=0.0,
    utilization_soft_cap=0.35,
)


@dataclass
class SimulationSettings:
    n_users: int = 5000
    horizon_days: int = 365
    seed: int = 42
    credit_limit_mean: float = 1200.0
    credit_limit_std: float = 250.0
    income_mean: float = 110.0
    income_std: float = 22.0
    expense_mean: float = 72.0
    expense_std: float = 18.0
    shock_probability: float = 0.015
    shock_min: float = 75.0
    shock_max: float = 350.0
    missed_payment_streak_default: int = 4


def clipped_normal(rng, mean, std, size=None, low=0.0):
    x = rng.normal(mean, std, size)
    return np.maximum(x, low)


def generate_user_population(settings: SimulationSettings, rng: np.random.Generator) -> pd.DataFrame:
    n = settings.n_users
    income_volatility = rng.uniform(0.05, 0.45, n)
    spending_bias = rng.uniform(0.8, 1.25, n)
    paycheck_frequency = rng.choice([1, 7, 14], size=n, p=[0.15, 0.55, 0.30])
    liquidity_buffer = clipped_normal(rng, 180, 120, n, low=20)
    credit_limit = clipped_normal(rng, settings.credit_limit_mean, settings.credit_limit_std, n, low=300)
    repayment_discipline = rng.beta(5, 2, n)
    return pd.DataFrame(
        {
            "user_id": np.arange(n),
            "income_volatility": income_volatility,
            "spending_bias": spending_bias,
            "paycheck_frequency": paycheck_frequency,
            "liquidity_buffer": liquidity_buffer,
            "credit_limit": credit_limit,
            "repayment_discipline": repayment_discipline,
        }
    )


def fico_proxy(payment_on_time_rate: float, avg_utilization: float, defaulted: bool) -> float:
    score = 610
    score += 170 * payment_on_time_rate
    score -= 120 * avg_utilization
    if defaulted:
        score -= 110
    return float(np.clip(score, 300, 850))


def simulate_product_for_population(
    population: pd.DataFrame,
    product: ProductConfig,
    settings: SimulationSettings,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_metrics = []
    daily_records = []
    daily_rate = product.apr / 365

    for row in population.itertuples(index=False):
        cash = row.liquidity_buffer
        balance = 0.0
        missed_streak = 0
        defaulted = False
        total_due_paid = 0
        total_due_events = 0
        payment_on_time_events = 0
        utilization_samples = []
        cumulative_fees = 0.0
        cumulative_interest = 0.0
        net_worth_proxy = cash

        for day in range(1, settings.horizon_days + 1):
            if (day - 1) % int(row.paycheck_frequency) == 0:
                income_today = clipped_normal(
                    rng,
                    settings.income_mean,
                    settings.income_std * (1 + row.income_volatility),
                    None,
                    low=0,
                )
                cash += float(income_today)

            expense_today = clipped_normal(
                rng,
                settings.expense_mean * row.spending_bias,
                settings.expense_std,
                None,
                low=0,
            )

            if rng.random() < settings.shock_probability:
                expense_today += rng.uniform(settings.shock_min, settings.shock_max)

            if cash >= expense_today:
                cash -= float(expense_today)
            else:
                shortfall = float(expense_today - cash)
                available_credit = max(float(row.credit_limit) - balance, 0.0)
                draw = min(shortfall, available_credit)
                balance += draw
                cash = max(cash - (expense_today - draw), 0.0)

            if balance > 0:
                interest_today = balance * daily_rate
                balance += interest_today
                cumulative_interest += interest_today

            if day % product.autopay_frequency_days == 0 and balance > 0:
                total_due_events += 1
                due_amount = balance * product.min_payment_rate if product.autopay_frequency_days >= 30 else balance
                due_amount = min(due_amount, balance)

                willingness = row.repayment_discipline
                pressure = np.clip((cash / max(row.credit_limit, 1.0)), 0, 1)
                pay_probability = 0.45 + 0.35 * willingness + 0.20 * pressure
                can_pay = cash >= due_amount

                if can_pay and rng.random() < pay_probability:
                    cash -= due_amount
                    balance -= due_amount
                    payment_on_time_events += 1
                    total_due_paid += due_amount
                    missed_streak = 0
                else:
                    missed_streak += 1
                    if product.late_fee > 0:
                        balance += product.late_fee
                        cumulative_fees += product.late_fee

            utilization = balance / max(float(row.credit_limit), 1.0)
            utilization_samples.append(utilization)
            net_worth_proxy = cash - balance

            if missed_streak >= settings.missed_payment_streak_default:
                defaulted = True

            if day in {30, 90, 180, 270, 365}:
                payment_rate = payment_on_time_events / total_due_events if total_due_events else 1.0
                avg_util = float(np.mean(utilization_samples)) if utilization_samples else 0.0
                daily_records.append(
                    {
                        "user_id": int(row.user_id),
                        "product": product.name,
                        "day": day,
                        "payment_on_time_rate": payment_rate,
                        "avg_utilization": avg_util,
                        "fico_proxy": fico_proxy(payment_rate, avg_util, defaulted),
                        "defaulted": int(defaulted),
                        "balance": balance,
                        "cash": cash,
                        "net_worth_proxy": net_worth_proxy,
                    }
                )

        payment_rate = payment_on_time_events / total_due_events if total_due_events else 1.0
        avg_util = float(np.mean(utilization_samples)) if utilization_samples else 0.0
        final_fico = fico_proxy(payment_rate, avg_util, defaulted)
        user_metrics.append(
            {
                "user_id": int(row.user_id),
                "product": product.name,
                "credit_limit": float(row.credit_limit),
                "income_volatility": float(row.income_volatility),
                "spending_bias": float(row.spending_bias),
                "repayment_discipline": float(row.repayment_discipline),
                "final_cash": cash,
                "final_balance": balance,
                "net_worth_proxy": net_worth_proxy,
                "payment_on_time_rate": payment_rate,
                "avg_utilization": avg_util,
                "defaulted": int(defaulted),
                "cumulative_interest": cumulative_interest,
                "cumulative_fees": cumulative_fees,
                "final_fico_proxy": final_fico,
            }
        )

    return pd.DataFrame(user_metrics), pd.DataFrame(daily_records)


def build_dashboard(user_df: pd.DataFrame, daily_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fico_traj = (
        daily_df.groupby(["product", "day"], as_index=False)["fico_proxy"].mean()
    )
    fig1 = px.line(
        fico_traj,
        x="day",
        y="fico_proxy",
        color="product",
        markers=True,
        title="Average FICO Proxy Trajectory",
    )
    fig1.write_html(outdir / "fico_trajectory.html")

    default_rates = (
        user_df.groupby("product", as_index=False)["defaulted"].mean()
        .assign(default_rate=lambda d: d["defaulted"] * 100)
    )
    fig2 = px.bar(
        default_rates,
        x="product",
        y="default_rate",
        title="Default Rate Comparison (%)",
    )
    fig2.write_html(outdir / "default_rate.html")

    fig3 = px.histogram(
        user_df,
        x="final_fico_proxy",
        color="product",
        barmode="overlay",
        nbins=35,
        title="Distribution of Final FICO Proxy",
    )
    fig3.write_html(outdir / "fico_distribution.html")

    summary = summarize_results(user_df)
    table = go.Figure(
        data=[go.Table(
            header=dict(values=list(summary.columns)),
            cells=dict(values=[summary[c] for c in summary.columns]),
        )]
    )
    table.update_layout(title="Simulation Summary")
    table.write_html(outdir / "summary_table.html")



def summarize_results(user_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        user_df.groupby("product", as_index=False)
        .agg(
            avg_final_fico=("final_fico_proxy", "mean"),
            median_final_fico=("final_fico_proxy", "median"),
            default_rate=("defaulted", "mean"),
            avg_utilization=("avg_utilization", "mean"),
            avg_interest=("cumulative_interest", "mean"),
            avg_fees=("cumulative_fees", "mean"),
            avg_net_worth=("net_worth_proxy", "mean"),
        )
    )
    summary["default_rate"] = summary["default_rate"] * 100
    return summary.round(2)



def run(settings: SimulationSettings, outdir: Path) -> None:
    rng = np.random.default_rng(settings.seed)
    population = generate_user_population(settings, rng)

    user_frames = []
    daily_frames = []
    for product in [TRADITIONAL_CARD, DAILY_PAY_CARD]:
        product_user_df, product_daily_df = simulate_product_for_population(population, product, settings, rng)
        user_frames.append(product_user_df)
        daily_frames.append(product_daily_df)

    user_df = pd.concat(user_frames, ignore_index=True)
    daily_df = pd.concat(daily_frames, ignore_index=True)
    summary_df = summarize_results(user_df)

    outdir.mkdir(parents=True, exist_ok=True)
    population.to_csv(outdir / "synthetic_population.csv", index=False)
    user_df.to_csv(outdir / "simulation_user_metrics.csv", index=False)
    daily_df.to_csv(outdir / "simulation_daily_snapshots.csv", index=False)
    summary_df.to_csv(outdir / "summary.csv", index=False)
    build_dashboard(user_df, daily_df, outdir / "dashboard")

    print("Saved outputs to:", outdir)
    print("\nSummary:\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo credit-builder simulation")
    parser.add_argument("--users", type=int, default=5000)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    settings = SimulationSettings(n_users=args.users, horizon_days=args.days, seed=args.seed)
    run(settings, Path(args.outdir))
