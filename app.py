from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------- Page setup --------------------
st.set_page_config(page_title="Student Expense Analytics", page_icon="💸", layout="wide")
st.title("💸 AI-Powered Student Expense & Budget Analytics")
st.caption("Explore spending patterns, estimate next month's expenses, and get practical budget suggestions.")

DATA_PATH = Path(__file__).parent / "data" / "student_expenses.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["amount_inr"] = pd.to_numeric(data["amount_inr"], errors="coerce")
    data = data.dropna(subset=["date", "category", "amount_inr"])
    data = data[data["amount_inr"] >= 0].copy()
    data["month"] = data["date"].dt.to_period("M").astype(str)
    return data

data = load_data()
if data.empty:
    st.error("The dataset is empty or invalid. Check data/student_expenses.csv.")
    st.stop()

# -------------------- Sidebar filters --------------------
st.sidebar.header("Dashboard filters")
min_date = data["date"].min().date()
max_date = data["date"].max().date()
date_range = st.sidebar.date_input("Choose date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
all_categories = sorted(data["category"].unique().tolist())
selected_categories = st.sidebar.multiselect("Categories", all_categories, default=all_categories)
monthly_budget = st.sidebar.number_input("Monthly budget (₹)", min_value=0, value=8000, step=500)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered = data[
    data["date"].dt.date.between(start_date, end_date)
    & data["category"].isin(selected_categories)
].copy()

if filtered.empty:
    st.warning("No expenses match these filters. Choose a wider date range or more categories.")
    st.stop()

# -------------------- Summary metrics --------------------
total_spend = filtered["amount_inr"].sum()
transaction_count = len(filtered)
daily_average = total_spend / max((end_date - start_date).days + 1, 1)
months_in_range = max(filtered["month"].nunique(), 1)
avg_monthly_spend = total_spend / months_in_range

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total spending", f"₹{total_spend:,.0f}")
c2.metric("Transactions", f"{transaction_count:,}")
c3.metric("Average per day", f"₹{daily_average:,.0f}")
c4.metric("Avg. spend per month*", f"₹{avg_monthly_spend:,.0f}")
st.caption("*Average across months represented in the filtered data; sample records are synthetic demo data.")

# -------------------- Charts --------------------
left, right = st.columns(2)
with left:
    st.subheader("Spending by category")
    category_spend = filtered.groupby("category")["amount_inr"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    category_spend.plot(kind="bar", ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Amount (₹)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Monthly spending trend")
    monthly_spend = filtered.groupby("month", as_index=False)["amount_inr"].sum().sort_values("month")
    fig, ax = plt.subplots()
    ax.plot(monthly_spend["month"], monthly_spend["amount_inr"], marker="o")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount (₹)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Expense records")
st.dataframe(
    filtered.sort_values("date", ascending=False)[
        ["date", "category", "description", "amount_inr", "payment_method"]
    ],
    use_container_width=True,
    hide_index=True
)

# -------------------- Simple forecasting model --------------------
st.header("🤖 AI-assisted insights")
st.subheader("Next-month spending estimate")
# Forecast from all available historical data, not only the current category filter.
history = data.groupby("month", as_index=False)["amount_inr"].sum().sort_values("month")
if len(history) >= 2:
    history["month_number"] = np.arange(len(history))
    model = LinearRegression()
    model.fit(history[["month_number"]], history["amount_inr"])
    next_x = np.array([[len(history)]])
    prediction = max(float(model.predict(next_x)[0]), 0.0)
    st.metric("Estimated next-month spending", f"₹{prediction:,.0f}")
    st.caption("A basic linear-regression estimate based on the demo dataset's monthly totals. It is educational, not financial advice, and may be inaccurate with short or irregular histories.")
else:
    st.info("At least two months of records are needed for the simple forecast.")

st.subheader("Budget and spending recommendations")
recommendations = []
if total_spend > monthly_budget and monthly_budget > 0:
    recommendations.append(f"Filtered spending (₹{total_spend:,.0f}) is above the budget you entered (₹{monthly_budget:,.0f}). Review the largest categories first.")
elif monthly_budget > 0:
    recommendations.append(f"Filtered spending is ₹{max(monthly_budget - total_spend, 0):,.0f} below the budget amount you entered. Note that the selected date range may cover more than one month.")
if not category_spend.empty:
    top_category = category_spend.index[0]
    top_amount = float(category_spend.iloc[0])
    recommendations.append(f"Your largest selected spending category is {top_category} (₹{top_amount:,.0f}). Check whether any costs in this category can be planned or reduced.")
    food = float(category_spend.get("Food", 0))
    entertainment = float(category_spend.get("Entertainment", 0))
    if food + entertainment > 0:
        recommendations.append(f"Food and entertainment together account for ₹{food + entertainment:,.0f} in the selected data. Setting a weekly limit may help you track these flexible expenses.")
recommendations.append("Record real expenses consistently and compare the same number of days or months when judging whether spending has increased.")
for item in recommendations:
    st.write("• " + item)

st.info("Privacy note: this starter project uses a synthetic example dataset. Do not upload bank statements, account numbers, or other sensitive personal data to a public repository.")
