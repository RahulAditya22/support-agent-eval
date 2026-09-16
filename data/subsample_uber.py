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

# Build a complete parent -> children index from the full corpus.  The prior
# implementation only followed the seed rows' explicit parent/response IDs,
# which truncated multi-turn conversations.  We instead expand the selected
# seed IDs to a fixed point over both graph directions.
by_id = df.dropna(subset=["tweet_id"]).drop_duplicates("tweet_id").set_index("tweet_id")
parent_to_children = {}
for row in df[["tweet_id", "in_response_to_tweet_id"]].dropna(subset=["tweet_id"]).itertuples(index=False):
    tweet_id = row.tweet_id
    parent_id = row.in_response_to_tweet_id
    if pd.notna(parent_id) and parent_id:
        parent_to_children.setdefault(parent_id, set()).add(tweet_id)

related_ids = set(seed_ids)
frontier = set(seed_ids)

while frontier:
    discovered = set()
    for tweet_id in frontier:
        row = by_id.loc[tweet_id] if tweet_id in by_id.index else None
        if row is not None:
            parent_id = row["in_response_to_tweet_id"]
            if pd.notna(parent_id) and parent_id:
                discovered.add(parent_id)
            value = row["response_tweet_id"]
            if pd.notna(value):
                for response_id in str(value).split(","):
                    response_id = response_id.strip()
                    if response_id:
                        discovered.add(response_id)
        discovered.update(parent_to_children.get(tweet_id, set()))

    discovered -= related_ids
    if not discovered:
        break
    related_ids.update(discovered)
    frontier = discovered

print(f"Seed tweet IDs: {len(seed_ids):,}")
print(f"Recursively closed related IDs: {len(related_ids):,}")

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
