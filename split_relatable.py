"""Split the `Relatable` column ("Category=>rank,...") into one integer column
per category. Each value is a rank 1-6 of how relatable that category is.
"""
import pandas as pd

IN_PATH = "data/cleaned_dataset.csv"
OUT_PATH = "data/cleaned_dataset_split.csv"
PREFIX = "rel_"          # column name prefix, e.g. rel_Skyscrapers
DROP_ORIGINAL = True     # drop the original Relatable column after splitting


def parse_relatable(value):
    """'Skyscrapers=>6,Sport=>4,...' -> {'Skyscrapers': 6, 'Sport': 4, ...}"""
    out = {}
    if pd.isna(value):
        return out
    for pair in str(value).split(","):
        if "=>" in pair:
            key, rank = pair.split("=>", 1)
            rank = rank.strip()
            out[key.strip()] = int(rank) if rank.isdigit() else pd.NA
    return out


def split_relatable(df, prefix=PREFIX, drop_original=DROP_ORIGINAL):
    # Expand the dict-per-row into a DataFrame of columns, sorted for stability
    expanded = df["Relatable"].apply(parse_relatable).apply(pd.Series)
    expanded = expanded.reindex(sorted(expanded.columns), axis=1)
    expanded.columns = [f"{prefix}{c}" for c in expanded.columns]
    expanded = expanded.astype("Int64")  # nullable int, just in case

    # Insert the new columns where Relatable used to be
    pos = df.columns.get_loc("Relatable")
    df = pd.concat([df.iloc[:, :pos], expanded, df.iloc[:, pos:]], axis=1)
    if drop_original:
        df = df.drop(columns=["Relatable"])
    return df


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df = split_relatable(df)
    df.to_csv(OUT_PATH, index=False)
    new_cols = [c for c in df.columns if c.startswith(PREFIX)]
    print(f"Wrote {OUT_PATH} ({len(df)} rows)")
    print("New columns:", new_cols)
    print(df[new_cols].head())
