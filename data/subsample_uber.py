from pathlib import Path
import pandas as pd

INPUT = Path("data/raw/twcs/twcs.csv")
OUTPUT = Path("data/processed/uber_support_sample.csv")

SEED = 42
TARGET_UBER_TWEETS = 4000

COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

print(f"Reading: {INPUT}")
print("This may take a little while...")

df = pd.read_csv(INPUT, usecols=COLUMNS)

print(f"Full dataset: {len(df):,} rows")
print(f"Columns: {list(df.columns)}")

# Normalize IDs as strings. IDs in this dataset are not necessarily numeric.
for col in ["tweet_id", "response_tweet_id", "in_response_to_tweet_id"]:
    df[col] = (
        df[col]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

text = df["text"].fillna("").astype(str)

uber_mask = text.str.contains(
    r"@Uber_Support\b",
    case=False,
    regex=True,
    na=False,
)

uber = df.loc[uber_mask].copy()
print(f"Uber-directed tweets: {len(uber):,}")

if uber.empty:
    raise ValueError("No tweets mentioning @Uber_Support were found.")

seed_n = min(TARGET_UBER_TWEETS, len(uber))
seed_tweets = uber.sample(n=seed_n, random_state=SEED)

seed_ids = set(seed_tweets["tweet_id"].dropna())
parent_ids = set(seed_tweets["in_response_to_tweet_id"].dropna())

response_ids = set()
for value in seed_tweets["response_tweet_id"].dropna():
    for tweet_id in str(value).split(","):
        tweet_id = tweet_id.strip()
        if tweet_id:
            response_ids.add(tweet_id)

related_ids = seed_ids | parent_ids | response_ids

print(f"Seed tweet IDs: {len(seed_ids):,}")
print(f"Parent IDs: {len(parent_ids):,}")
print(f"Response IDs: {len(response_ids):,}")
print(f"Total related IDs: {len(related_ids):,}")

sample = df[df["tweet_id"].isin(related_ids)].copy()

missing_seed_ids = seed_ids - set(sample["tweet_id"].dropna())
if missing_seed_ids:
    raise ValueError(
        f"{len(missing_seed_ids)} sampled Uber tweets could not be recovered."
    )

sample["_created"] = pd.to_datetime(
    sample["created_at"],
    format="%a %b %d %H:%M:%S %z %Y",
    errors="coerce",
    utc=True,
)

if sample["_created"].isna().any():
    raise ValueError("One or more sampled timestamps could not be parsed.")

sample = (
    sample
    .sort_values(["_created", "tweet_id"], na_position="last")
    .drop(columns=["_created"])
    .drop_duplicates(subset=["tweet_id"])
    .reset_index(drop=True)
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(OUTPUT, index=False)

print()
print("=== SAMPLE CREATED ===")
print(f"Output: {OUTPUT}")
print(f"Rows: {len(sample):,}")
print(f"Random seed: {SEED}")
print(f"Uber seed tweets: {seed_n:,}")

sample_text = sample["text"].fillna("").astype(str)
print(
    "Rows mentioning @Uber_Support:",
    sample_text.str.contains(
        r"@Uber_Support\b",
        case=False,
        regex=True,
        na=False,
    ).sum(),
)
print("Inbound rows:", int(sample["inbound"].fillna(False).sum()))
print("Outbound rows:", int((~sample["inbound"].fillna(False)).sum()))

parsed = pd.to_datetime(
    sample["created_at"],
    format="%a %b %d %H:%M:%S %z %Y",
    utc=True,
)
print("Date range:", parsed.min(), "->", parsed.max())
