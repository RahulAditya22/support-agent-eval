# Hiver SDE Intern Assignment â€” Uber Support Agent Evaluation

This repository builds and evaluates an AI support agent for `Uber_Support` using the Kaggle Customer Support on Twitter dataset.

The system classifies an initiating customer tweet, retrieves relevant historical support replies, drafts a grounded response, and decides whether the request can be auto-handled or should be escalated.

## 1. Assignment Scope

The system:

1. Reconstructs customer-support conversations from the Twitter support dataset.
2. Derives 6â€“12 support intents from the selected Uber Support data.
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
