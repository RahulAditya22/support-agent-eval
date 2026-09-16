# Phase 1 EDA — Uber_Support Sample

## Source and sampling

- Source: Kaggle **Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`).
- Local full file: `data/raw/twcs/twcs.csv`.
- Full dataset rows observed locally: **2,811,774**.
- Sampling script uses a fixed random seed of **42**.
- Seed selection: **4,000** tweets whose text contains the exact handle `@Uber_Support`.
- For each seed, directly linked parent and response tweet IDs were collected where available.
- The resulting sample contains **7,225 unique tweets**.
- The uploaded sample was independently re-read and validated before processing.

## Schema

The dataset contains seven fields:

`tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id`.

There is **no `company` column** in the actual `twcs.csv`; brand targeting therefore uses the `@Uber_Support` handle in tweet text.

## Validation and cleaning

- Rows: **7,225**
- Unique `tweet_id`: **7,225**
- Duplicate tweet IDs: **0**
- Invalid timestamps using the Twitter format `%a %b %d %H:%M:%S %z %Y`: **0**
- Exact duplicate text values: **89** (these are not removed solely on text equality because different tweets can legitimately contain the same wording).
- Inbound rows: **4,332**
- Outbound rows: **2,893**
- Rows mentioning `@Uber_Support`: **4,326**
- Unique authors: **3,703**

Identifier columns are normalized to string IDs so values such as `1445366.0` produced by CSV parsing are matched consistently with `1445366`.

## Conversation reconstruction

Conversation relationships are represented by `in_response_to_tweet_id` and `response_tweet_id`. Within this sample, **3,378** rows have a parent reference that resolves to another sampled tweet, and **3,203** rows have at least one response reference that resolves within the sample.

Using resolved within-sample edges, the sample forms **3,847 connected conversation components**. Component sizes range from 1 to 9 tweets; **845 components are singletons**. This confirms that the sample contains usable short conversation context, while also showing that the sampling strategy does not recover every full thread from the original dataset.

## Volume and date checks

The validated sample spans **2017-04-27 through 2017-12-03**.

- Mean daily sample volume: **97.6**
- Median daily sample volume: **105**
- Maximum daily sample volume: **227**

For the assignment's specified spike dates:

- **2017-09-22:** 0 sampled tweets. The random Uber-focused sample does **not** contain enough observations on this date to independently verify a spike.
- **2017-11-21:** **159** sampled tweets, including **92** tweets mentioning `@Uber_Support`. This is one of the highest-volume dates in the sample, but the sample alone should not be treated as proof of a full-dataset spike.

The final report should distinguish **sample evidence** from claims about the full 2.8M-row dataset.

## Reproducibility

The processed sample can be regenerated from the locally downloaded Kaggle file using:

```powershell
py data/subsample_uber.py
```

The sampler's random seed is fixed at `42`. Raw data remain excluded from Git by `.gitignore`.

## Phase 1 conclusion

The sample is suitable for the next phase because it contains Uber-directed customer messages plus directly linked context, preserves the original conversation-link fields, and has reproducible sampling and validated timestamps. Phase 2 intent discovery should explicitly account for the fact that this is a linked-context subsample rather than the complete Uber conversation corpus.
