"""Optional command-line summary of the demo expense dataset."""
from pathlib import Path
import pandas as pd

path = Path(__file__).parent / "data" / "student_expenses.csv"
df = pd.read_csv(path, parse_dates=["date"])
print(f"Rows: {len(df)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total spend: ₹{df['amount_inr'].sum():,.2f}")
print("\nSpend by category:")
print(df.groupby("category")["amount_inr"].sum().sort_values(ascending=False).round(2))
print("\nSpend by month:")
print(df.groupby(df["date"].dt.to_period("M"))["amount_inr"].sum().round(2))
