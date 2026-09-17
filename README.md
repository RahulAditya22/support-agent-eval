# Hiver SDE Intern Assignment — Uber Support Agent Evaluation

This repository builds and evaluates an AI support agent for `Uber_Support` using the Kaggle Customer Support on Twitter dataset.

The system classifies an initiating customer tweet, retrieves relevant historical support replies, drafts a grounded response, and decides whether the request can be auto-handled or should be escalated.

## 1. Assignment Scope

The system:

1. Reconstructs customer-support conversations from the Twitter support dataset.
2. Derives 6–12 support intents from the selected Uber Support data.
3. Classifies the initiating customer tweet.
4. Retrieves relevant historical support replies.
5. Drafts a concise response grounded in retrieved evidence.
6. Decides whether to auto-handle or escalate, with an explicit reason.
7. Evaluates the system against a 200-example human-labelled golden set.
8. Compares the agent with non-LLM baselines.
9. Uses an LLM judge on a separate sample to assess groundedness and policy adherence.

---

## 2. Dataset and EDA

**Source:** Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)
**Selected brand:** `Uber_Support`

The raw dataset is intentionally not committed to this repository.

### Dataset volume

Full-data analysis identified:

- 46,626 Uber-directed tweets using a case-insensitive `@Uber_Support` mention filter.
- 46,608 tweets within the assignment window: **2017-04-27 through 2017-12-03**.
- 18 tweets outside the window.
- 0 timestamp parsing failures.
- 221 calendar days in the collection window.
- 97 days with zero Uber-directed tweets.

Daily volume is highly skewed:

- Mean: 210.9 tweets/day
- Median: 1 tweet/day
- Top 10 days: 11,013 tweets (23.63%)
- Top 15 days: 15,686 tweets (33.66%)
- Highest-volume day: December 1, 2017, with 1,359 tweets.

The September 22, 2017 data contains only 7 Uber-directed tweets and does not establish a meaningful volume spike. November 21, 2017 contains 917 tweets within a broader elevated-activity period. Tweet volume alone does not establish causality.

### Conversation reconstruction

Recursive conversation-graph closure was used to recover related tweets through `tweet_id`, `in_response_to_tweet_id`, and `response_tweet_id` relationships.

The resulting processed Uber sample contains:

- **20,520** reconstructed rows
- **18,673** validated historical replies

The retrieval corpus uses a documented structural eligibility heuristic for raw TWCS reconstruction. This is a corpus-construction heuristic, not proof that a customer was satisfied or that a support issue was actually resolved.

---

## 3. Intent Taxonomy

The final taxonomy contains 10 single-label intents derived from the observed Uber Support data:

| Intent | Description |
|---|---|
| `cancellation_no_show_charge` | Cancellation fees, no-show charges, or charges associated with cancelled/no-show rides |
| `unauthorized_transaction` | Specific unauthorized or unrecognized account/transaction activity |
| `fare_price_dispute` | Disputes about fares, pricing, route-based charges, or unexpected ride costs |
| `missing_incomplete_ride` | Ride not received, incomplete ride, or ride missing from the expected flow |
| `uber_eats_order_delivery_issue` | Uber Eats order, delivery, or food-order problems |
| `ride_pass_promotion_coupon` | Ride Pass, promotions, discounts, coupons, or promotional eligibility |
| `account_login_app_access` | Login, registration, password, app, or account-access problems |
| `payment_refund_billing` | Refund, duplicate billing, payment processing, or billing-resolution requests |
| `driver_partner_safety_issue` | Driver conduct, safety, or partner-related concerns |
| `other_out_of_scope` | Requests that do not fit the defined support intents |

### Single-label resolution

Ambiguous multi-theme tweets use this hierarchy:

1. Explicit requested resolution
2. Blocking issue
3. Greatest emphasis/detail
4. `other_out_of_scope`

This makes the human labelling rule deterministic and reproducible.

---

## 4. Agent Architecture

The implementation is split into independently testable components:

```text
Customer tweet
      |
      v
Classification
      |
      +------------------+
      |                  |
      v                  v
Intent             Historical retrieval
                         |
                         v
                  Evidence selection
                         |
                         v
                    Draft reply
                         |
                         v
                    Escalation
```

Key modules:

- `src/classify.py` — intent classification
- `src/retrieve.py` — historical reply retrieval
- `src/draft_reply.py` — grounded response generation
- `src/escalate.py` — auto-handle/escalation decision
- `src/pipeline.py` — end-to-end orchestration
- `src/llm_client.py` — OpenAI-compatible Groq client with timeout, retry, and call/output budgets
- `src/corpus.py` and `src/historical_reply_builder.py` — retrieval-corpus construction
- `eval/harness.py` — evaluation metrics and validation
- `eval/llm_judge.py` — structured LLM judging
- `src/baselines.py` — evaluation baselines

LLM calls can be mocked in tests and are bounded by explicit call and output-token budgets.

---

## 5. Golden Evaluation Set

The final evaluation uses **200 human-labelled examples** from the selected Uber Support population.

Human labels are stored separately from model predictions:

- `human_intent_label`
- `human_auto_handle_label`
- `notes`

The model's predicted fields are never treated as ground truth. The human-reviewed set is tracked at `data/eval/golden_set_candidates.csv`.

The set is intentionally not forced to have exactly equal representation across all intents; it contains 200 human-reviewed examples spanning all 10 taxonomy categories.

---

## 6. Automated Evaluation Results

The final agent evaluation processed all **200/200** golden-set rows successfully with **0 failed agent rows**.

### Classification

| Metric | Agent |
|---|---:|
| Accuracy | **79.00%** |
| Macro-F1 | **78.19%** |

Per-intent precision and recall:

| Intent | Precision | Recall |
|---|---:|---:|
| `account_login_app_access` | 76.92% | 95.24% |
| `cancellation_no_show_charge` | 72.73% | 88.89% |
| `driver_partner_safety_issue` | 76.67% | 88.46% |
| `fare_price_dispute` | 74.07% | 86.96% |
| `missing_incomplete_ride` | 83.33% | 50.00% |
| `other_out_of_scope` | 83.33% | 64.52% |
| `payment_refund_billing` | 80.00% | 57.14% |
| `ride_pass_promotion_coupon` | 94.74% | 100.00% |
| `uber_eats_order_delivery_issue` | 92.31% | 75.00% |
| `unauthorized_transaction` | 61.54% | 88.89% |

### Escalation

| Metric | Agent |
|---|---:|
| Precision | **100.00%** |
| Recall | **34.52%** |
| False auto-handle rate | **64.50%** |

The headline 100% escalation precision is intentionally not presented as a standalone quality claim. It occurs alongside low recall and a 64.5% false auto-handle rate, so the complete metric set is necessary to interpret the behaviour.

### Retrieval

Top-k recall is **not reported** because the current golden set does not contain explicit human gold retrieval targets. The evaluation harness supports `gold_retrieval_reply` when such targets are available.

For the sampled retrieval outputs, 1,000 similarity observations had:

- Mean similarity: 0.6788
- Median: 0.6892
- Minimum: 0.3683
- Maximum: 0.8808
- 25th percentile: 0.6259
- 75th percentile: 0.7310

These similarity statistics are descriptive and are not retrieval-recall measurements.

---

## 7. Baselines

Two non-LLM classification baselines were evaluated on the same 200-example population using an 80/20 stratified split with `random_state=42` where training was required:

| Method | Accuracy |
|---|---:|
| Majority-class baseline | 20% |
| TF-IDF + Logistic Regression | 40% |
| Agent | **79%** |

The baselines provide reference points rather than proving causality for the agent's performance difference.

---

## 8. Failure Analysis

There were **42 classification errors** in the 200-example evaluation.

The most frequent mismatch pairs were:

| Human intent → Predicted intent | Count |
|---|---:|
| `payment_refund_billing` → `fare_price_dispute` | 7 |
| `other_out_of_scope` → `unauthorized_transaction` | 4 |
| `other_out_of_scope` → `account_login_app_access` | 4 |
| `missing_incomplete_ride` → `driver_partner_safety_issue` | 3 |
| `missing_incomplete_ride` → `cancellation_no_show_charge` | 2 |

### Failure hypotheses

1. **Billing vs fare overlap:** duplicate charges, fees, refunds, and route-based price disputes often share the same monetary vocabulary. The classifier can select `fare_price_dispute` when the requested resolution is actually billing/refund-related.
2. **Out-of-scope vs fraud/account access:** vague fraud language and account-access complaints can resemble the more specific `unauthorized_transaction` or `account_login_app_access` intents without enough context to establish the intended category.
3. **Ride completion vs safety/cancellation:** reports about a driver not arriving, starting a ride unexpectedly, or a ride not completing can contain safety or cancellation cues that compete with `missing_incomplete_ride`.

These hypotheses motivate clearer boundary examples and explicit requested-resolution cues in future classifier prompts or labelling guidance.

---

## 9. LLM Judge and Human Agreement

A separate LLM judge evaluated successfully generated draft/evidence cases for:

- Groundedness, scored 1–5
- Policy adherence, scored 1–5

The final accumulated judge artifact contains **46 valid judgments**. Invalid JSON responses, runtime failures, and budget-exhausted calls were not converted into scores.

Across the 46 valid judgments:

- Groundedness mean: **4.91/5**
- Policy adherence mean: **4.91/5**
- For each dimension, 45 cases received 5/5 and 1 case received 1/5.

A blind human review sample of **10 cases**, one per taxonomy intent, was compared with the LLM judge:

| Dimension | Exact agreement | Cohen's κ | Human mean | LLM mean |
|---|---:|---:|---:|---:|
| Groundedness | 80% | 0.00 | 4.8 | 5.0 |
| Policy adherence | 80% | 0.00 | 4.8 | 5.0 |

Cohen's kappa is not informative for this sample because the LLM judge assigned the same score to every sampled case, producing no rating variance. Exact agreement is therefore the primary agreement statistic reported here. The agreement artifact is stored at `data/eval/phase6_human_agreement.json`, with the underlying review sheet at `data/eval/judge_human_agreement.csv`.

The LLM judge is treated as secondary evidence, not as a replacement for the 200-example human-labelled evaluation.

---

## 10. Retrieval Provenance Limitation

The historical reply corpus is derived from raw TWCS structure using a documented heuristic. The heuristic uses conversation graph structure, inbound/outbound authorship, and terminal-message relationships to identify eligible support replies.

This does **not** establish that a customer was satisfied or that the issue was genuinely resolved. Therefore retrieval results using this corpus should not be described as performance against a human-verified resolution corpus.

The repository deliberately keeps the provenance boundary explicit rather than silently converting structural metadata into human claims about resolution.

---

## 11. One-Week Next Steps

1. Add a small human-reviewed retrieval benchmark with explicit gold replies or acceptable evidence sets.
2. Review the highest-frequency classification boundary errors and add targeted contrastive examples.
3. Improve escalation calibration to reduce the current false auto-handle rate while preserving useful automation.
4. Add more structured human review of generated replies, especially for low-confidence retrieval cases.
5. Replace the raw-data resolution heuristic with stronger documented thread-resolution evidence if the dataset supports it.
6. Re-run the evaluation after prompt/taxonomy changes and compare against the same frozen golden set.

---

## 12. Decision Log

The repository keeps a decision log at `report/decision_log.md` covering the major design choices across intent discovery, conversation reconstruction, retrieval provenance, evaluation, baselines, LLM budgeting, checkpointing, and judge usage.

There are 12 documented decision entries. Important decisions include:

- derive the taxonomy from Uber Support data rather than importing a benchmark taxonomy;
- use a deterministic single-label hierarchy;
- recursively close the conversation graph;
- keep human labels separate from predictions;
- use same-population baselines;
- bound and retry LLM calls;
- checkpoint long evaluation runs;
- treat LLM-judge scores as sampled secondary evidence.

---

## 13. Testing and Reproduction

The project includes unit tests covering classification, retrieval, drafting, escalation, pipeline orchestration, conversation reconstruction, metrics, LLM client behaviour, evaluation harnesses, baselines, and judge parsing.

The latest complete local test run recorded **95 passing tests**.

The repository also includes a GitHub Actions workflow at `.github/workflows/tests.yml` that installs `requirements.txt` on Python 3.11 and runs `pytest -q`.

Useful evaluation entry points:

```text
scripts/extract_golden_candidates.py
scripts/run_evaluation.py
scripts/run_judge_only.py
```

The real LLM evaluation requires a locally supplied Groq-compatible API key. Evaluation scripts also support mocked/bounded operation so tests do not require credentials.

---

## 14. Security and Data Handling

- The raw TWCS dataset is excluded from Git.
- API credentials are supplied through environment variables and are not committed.
- `.env.example` contains placeholders only.
- Generated evaluation JSON files are ignored by default unless explicitly required as tracked artifacts.
- The tracked golden set and human-agreement artifacts contain only the evaluation material intentionally selected for the assignment.

---

## 15. Credits

Dataset, library, model-provider, and attribution information is documented in [`CREDITS.md`](CREDITS.md).

The real LLM integration uses Groq's OpenAI-compatible API with `openai/gpt-oss-120b`. The project does not claim ownership of the underlying dataset or third-party libraries.
