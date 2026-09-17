# Phase 1 — Dataset and EDA Notes

## Source and scope

The assignment uses the Kaggle **Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`). The selected brand is `Uber_Support`.

The raw dataset was downloaded locally and is not committed to the repository. Uber-directed tweets were identified using the case-insensitive text-mention pattern `@Uber_Support\b`.

## Data volume and date validation

The full dataset contains **46,626 Uber-directed tweets** based on the text-mention filter.

The assignment's collection window is **2017-04-27 through 2017-12-03**. After parsing `created_at` using the dataset's timestamp format:

- **46,608 tweets** fall within the assignment's collection window.
- **18 tweets** fall outside the collection window. These are real, successfully parsed timestamps dated in 2014–2016 (and immediately before the stated 2017-04-27 start date), rather than timestamp parsing failures.
- **0 timestamps** failed to parse (`NaT`).
- The in-window period covers **221 calendar days**, of which **97 have zero Uber-directed tweets**.

The 18 out-of-window records were excluded from the daily-volume and incident analysis. This produces a clean in-window population of **46,608 tweets**.

## Daily volume distribution

Across the 221-day collection window:

- **Mean daily volume:** 210.9 tweets/day
- **Median daily volume:** 1 tweet/day

The large difference between the mean and median indicates a highly skewed volume distribution. The mean should therefore not be treated as a typical day's volume.

Volume is concentrated in a relatively small number of high-volume days. Among the 46,608 in-window tweets:

- The **top 10 days account for 11,013 tweets (23.63%)**.
- The **top 15 days account for 15,686 tweets (33.66%)**.

The top-volume days are clustered almost entirely in **November 1, November 14–30, and December 1–2, 2017**. The single highest-volume day is **December 1, 2017, with 1,359 Uber-directed tweets**.

This concentration is important when interpreting the overall mean: the 210.9 tweets/day figure is strongly influenced by a relatively small number of high-volume days, while many days have very little or no Uber-directed activity.

## Incident-date checks

### 2017-09-22 — TfL license revocation

The full in-window dataset contains **7 Uber-directed tweets on 2017-09-22**.

The surrounding week is also low-volume, averaging approximately **5 tweets/day**. The date is therefore **not supported as a volume spike** in this dataset.

The data can establish the observed Twitter-volume pattern, but it does not establish the cause of that pattern.

### 2017-11-21 — breach disclosure

The full in-window dataset contains **917 Uber-directed tweets on 2017-11-21**.

This date falls within a sustained high-volume period, with several nearby dates also reaching hundreds or more than 1,000 Uber-directed tweets. The period includes **December 1, 2017, which has 1,359 tweets and is the single highest-volume day in the dataset**.

Accordingly, the appropriate interpretation is **elevated activity in this period, of which November 21 is one part**.

The volume pattern does **not** by itself confirm the breach-disclosure hypothesis. December 1 and several other nearby dates have equal or higher activity without a corresponding known incident being attached to them in this analysis. Incident causality therefore remains outside what can be established from tweet volume alone.

## Pipeline sample integrity

The reproducible pipeline sample was constructed from the full Uber-directed population using a fixed seed and linked parent/response records.

A direct tweet-ID comparison confirmed that **0 of the 18 out-of-window Uber-directed records are present in `data/processed/uber_support_sample.csv`**.

Therefore, the existing **4,000-row Uber-directed seed pipeline sample is uncontaminated by the identified out-of-window records**.

The processed sample remains the working artifact for subsequent pipeline development and evaluation and was not replaced by the full-data incident analysis.

## Phase 1 conclusion

The full-data check resolves the initial date-range discrepancy and provides a corrected basis for the incident analysis.

The dataset contains **46,608 Uber-directed tweets within the assignment's 2017-04-27–2017-12-03 collection window**, with no timestamp parsing failures. Daily volume is highly skewed: the median is 1 tweet/day while the mean is 210.9 tweets/day, and the top 15 days account for 33.66% of all in-window Uber-directed tweets.

The full-data evidence does **not** support a volume spike on September 22, 2017. November 21, 2017 is part of a sustained period of elevated Twitter activity, but the volume pattern alone does not establish that the period was caused by the breach disclosure.

These findings are used only to characterize the dataset and its observed support-volume patterns; they are not treated as evidence of incident causality.
