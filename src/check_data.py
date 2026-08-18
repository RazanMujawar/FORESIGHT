from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def inspect_file(filename, nrows=5):
    path = RAW_DIR / filename

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    df = pd.read_csv(path, nrows=nrows)

    print("Columns:")
    print(df.columns.tolist())

    print("\nSample:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)


files = [
    "sales_transactions.csv",
    "sku_master.csv",
    "inventory_snapshot.csv",
    "promotions.csv",
    "sku_inventory_flags.csv",
    "store_master.csv",
    "customer_master.csv",
]

for file in files:
    inspect_file(file)