"""Split the `Company` column into one binary column
per category.
"""
import pandas as pd

IN_PATH = "data/cleaned_dataset_split.csv"
OUT_PATH = "data/cleaned_dataset_complete.csv"
PREFIX = "bring_"
DROP_ORIGINAL = True


def parse_company(value):
    out = {"Co-worker": 0, "Friends": 0, "Siblings": 0, "Partner": 0}
    if pd.isna(value):
        return out
    for p in str(value).split(","):
        out[p] = 1
    return out


def split_company(df, prefix=PREFIX, drop_original=DROP_ORIGINAL):
    # Expand the dict-per-row into a DataFrame of columns, sorted for stability
    expanded = df["Company"].apply(parse_company).apply(pd.Series)
    expanded = expanded.reindex(sorted(expanded.columns), axis=1)
    expanded.columns = [f"{prefix}{c}" for c in expanded.columns]
    expanded = expanded.astype("Int64")  # nullable int, just in case

    # Insert the new columns where Relatable used to be
    pos = df.columns.get_loc("Company")
    df = pd.concat([df.iloc[:, :pos], expanded, df.iloc[:, pos:]], axis=1)
    if drop_original:
        df = df.drop(columns=["Company"])
    return df


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df = split_company(df)
    df.to_csv(OUT_PATH, index=False)
    new_cols = [c for c in df.columns if c.startswith(PREFIX)]
    print(f"Wrote {OUT_PATH} ({len(df)} rows)")
    print("New columns:", new_cols)
    print(df[new_cols].head())
