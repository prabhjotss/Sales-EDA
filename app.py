"""
Sales Analytics Dashboard — Flask App
Run: python app.py  →  http://127.0.0.1:5000
"""

import os
import json
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from flask import Flask, render_template, jsonify

# ── Config ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
STATIC_CHARTS = os.path.join("static", "charts")
os.makedirs(STATIC_CHARTS, exist_ok=True)

MONTH_ORDER = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
DAY_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
ACCENT      = "#4C72B0"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


# ── Data loading & cleaning ─────────────────────────────────────────────────
def load_data(path: str = "cleaned_sales.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)

    for col in ["Quantity Ordered", "Price Each", "Order ID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df.dropna(subset=["Order Date", "Quantity Ordered", "Price Each", "Product"], inplace=True)
    df.drop_duplicates(inplace=True)

    df["Quantity Ordered"] = df["Quantity Ordered"].astype(int)
    df["Sales"]      = df["Price Each"] * df["Quantity Ordered"]
    df["Month"]      = df["Order Date"].dt.month_name()
    df["Month_Num"]  = df["Order Date"].dt.month
    df["Hour"]       = df["Order Date"].dt.hour
    df["Day"]        = df["Order Date"].dt.day_name()

    addr = df["Purchase Address"].str.split(",", expand=True)
    df["City"] = addr[1].str.strip()
    return df


DF = load_data()


# ── KPI helpers ─────────────────────────────────────────────────────────────
def kpis():
    total_rev   = DF["Sales"].sum()
    total_orders= DF["Order ID"].nunique()
    avg_order   = total_rev / total_orders

    monthly = DF.groupby(["Month_Num","Month"])["Sales"].sum().reset_index().sort_values("Month_Num")
    best_month  = monthly.loc[monthly["Sales"].idxmax(), "Month"]

    city_rev    = DF.groupby("City")["Sales"].sum()
    top_city    = city_rev.idxmax()

    prod_rev    = DF.groupby("Product")["Sales"].sum()
    top_product = prod_rev.idxmax()

    return {
        "total_revenue": f"${total_rev:,.0f}",
        "total_orders":  f"{total_orders:,}",
        "avg_order":     f"${avg_order:,.0f}",
        "best_month":    best_month,
        "top_city":      top_city,
        "top_product":   top_product,
    }


# ── Chart generators ─────────────────────────────────────────────────────────
def save_fig(name: str):
    path = os.path.join(STATIC_CHARTS, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


def chart_monthly():
    monthly = (DF.groupby(["Month_Num","Month"])["Sales"]
                 .sum().reset_index().sort_values("Month_Num"))
    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = [("#e74c3c" if m == monthly.loc[monthly["Sales"].idxmax(), "Month"]
                else ACCENT) for m in monthly["Month"]]
    ax.bar(monthly["Month"], monthly["Sales"]/1e6, color=colors, edgecolor="white")
    ax.set_title("Monthly Revenue", fontsize=14, fontweight="bold")
    ax.set_ylabel("Revenue ($M)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))
    plt.xticks(rotation=30, ha="right")
    save_fig("monthly")


def chart_hourly():
    hourly = DF.groupby("Hour").agg(Orders=("Order ID","nunique"),
                                     Revenue=("Sales","sum")).reset_index()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(hourly["Hour"], hourly["Orders"], alpha=0.5, color=ACCENT, label="Orders")
    ax2.plot(hourly["Hour"], hourly["Revenue"]/1e3, color="#e74c3c",
             linewidth=2.5, marker="o", markersize=4, label="Revenue ($K)")
    ax1.set_xlabel("Hour of Day"); ax1.set_ylabel("Orders", color=ACCENT)
    ax2.set_ylabel("Revenue ($K)", color="#e74c3c")
    ax1.set_title("Hourly Orders & Revenue", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(0, 24))
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels= ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper left")
    save_fig("hourly")


def chart_products():
    prod = (DF.groupby("Product")["Sales"].sum()
              .reset_index().sort_values("Sales", ascending=False))
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=prod, x="Sales", y="Product", palette="viridis_r", ax=ax)
    ax.set_title("Product Revenue", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.set_xlabel("Revenue"); ax.set_ylabel("")
    save_fig("products")


def chart_cities():
    city = (DF.groupby("City")["Sales"].sum()
              .reset_index().sort_values("Sales", ascending=False))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=city, x="City", y="Sales", palette="Blues_d", ax=ax)
    ax.set_title("Revenue by City", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    plt.xticks(rotation=20, ha="right")
    save_fig("cities")


def chart_heatmap():
    pivot = DF.groupby(["Day","Hour"])["Sales"].sum().unstack(fill_value=0)
    pivot = pivot.reindex([d for d in DAY_ORDER if d in pivot.index])
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot/1e3, cmap="YlOrRd", linewidths=0.2, ax=ax,
                cbar_kws={"label": "Revenue ($K)"})
    ax.set_title("Sales Heatmap: Day × Hour", fontsize=14, fontweight="bold")
    save_fig("heatmap")


def chart_bundles():
    basket = DF.groupby("Order ID")["Product"].apply(list)
    basket = basket[basket.apply(len) > 1]
    pairs  = Counter()
    for prods in basket:
        for combo in combinations(sorted(set(prods)), 2):
            pairs[combo] += 1
    top = pd.DataFrame(pairs.most_common(10), columns=["pair","count"])
    top["label"] = top["pair"].apply(lambda x: f"{x[0]} + {x[1]}")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top, x="count", y="label", palette="rocket_r", ax=ax)
    ax.set_title("Top Frequently Bought Together Pairs", fontsize=14, fontweight="bold")
    ax.set_xlabel("Co-occurrence Count"); ax.set_ylabel("")
    save_fig("bundles")


def generate_all_charts():
    chart_monthly()
    chart_hourly()
    chart_products()
    chart_cities()
    chart_heatmap()
    chart_bundles()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    generate_all_charts()
    return render_template("index.html", kpis=kpis())


@app.route("/api/kpis")
def api_kpis():
    return jsonify(kpis())


@app.route("/api/insights")
def api_insights():
    insights = [
        {"icon": "📅", "title": "December Peak", "text": "December is the #1 revenue month. Pre-stock inventory by October."},
        {"icon": "⏰", "title": "Ad Timing", "text": "Peak buying at 11–12 PM and 7–9 PM. Run ads 1 hour before these windows."},
        {"icon": "🏙️", "title": "SF Premium Buyers", "text": "San Francisco has the highest Average Order Value — target with premium upsells."},
        {"icon": "📦", "title": "iPhone & Macbook Lead", "text": "Top 2 revenue products. Protect stock, prioritise in search ads."},
        {"icon": "🔌", "title": "Bundle Opportunity", "text": "iPhone + Cable is the #1 co-purchase pair. Create official bundles with 5% discount."},
        {"icon": "💤", "title": "Pause Overnight Ads", "text": "Minimal orders 2–6 AM. Reallocate ad budget to peak windows for better ROAS."},
        {"icon": "📉", "title": "Q1 Slowdown", "text": "Jan–Mar is the weakest quarter. Use clearance sales and loyalty rewards to maintain cash flow."},
        {"icon": "🎯", "title": "Austin Untapped", "text": "Lowest order city — test geo-targeted campaigns; growing tech market with upside."},
    ]
    return jsonify(insights)


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Dashboard starting at http://127.0.0.1:5000")
    app.run(debug=True)