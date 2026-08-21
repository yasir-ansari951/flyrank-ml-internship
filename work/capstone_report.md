# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Abu Yasir
- **Lane:** Refresh / Content Opportunity Scoring
- **Repository:** https://github.com/yasir-ansari951/flyrank-ml-internship
- **Dataset:** FlyRank ML Internship dataset
- **Working sample:** 10,000 rows, shuffled (`seed=42`) before sampling
- **Random state:** 42
- **Revision note:** this report was revised after a self-audit found and fixed a target-leakage bug and an unshuffled-sample bug in the original run. See **Section 5, Audit Trail**, for the full record.

## 1. Problem Framing

### Research Question

Can historical search-performance signals be used to create a useful and interpretable ranking of content items for human review and potential refresh?

The selected capstone lane is **Refresh / Content Opportunity Scoring**.

The practical decision supported by this project is:

> Which content items should a reviewer inspect first?

The system is intentionally designed as decision-support rather than an automated editorial system.

It does not decide that a page must be refreshed, rewritten, deleted, merged, published, or expected to gain traffic.

### Why This Problem Matters

Content teams may have more content items to review than they can manually inspect at one time.

A ranked review queue can help direct attention toward items whose observed search-performance signals indicate that further investigation may be useful.

The purpose of this project is therefore not to automate editorial judgment. Instead, the objective is to test whether historical search and analytics signals can support a practical prioritization workflow.

### Unit of Analysis

The working unit is a historical content-performance observation from the FlyRank `fact_content_daily_performance` table (78.8M rows in the full warehouse).

The final capstone uses a working sample of **10,000 rows**, drawn with `.shuffle(seed=42)` before sampling so the sample is not dominated by whatever client or date happens to sort first in storage.

### Output

The project produces:

- a proxy target;
- a transparent baseline;
- a Decision Tree model (plain and class-weighted);
- a Random Forest model (class-weighted, with threshold tuning);
- grouped validation results, with a minimum-client-count guard;
- a leakage audit, including a quantitative trip-wire in addition to the structural column check;
- error examples;
- feature-importance information;
- a ranked opportunity queue;
- opportunity scores;
- action labels;
- reason codes;
- supporting search-performance metrics.

### Human Action

A content reviewer can use the ranked queue to identify which content items should be investigated first.

For a high-priority item, the reviewer should inspect search demand, search intent, content quality, recent content changes, seasonality, business importance, and other contextual information.

The reviewer then decides whether any editorial action is justified.

### Cost of a Wrong Decision

A false positive can cause a team to spend time investigating or changing content that did not require attention.

A false negative can cause a potentially useful content opportunity to be missed.

Because the project uses a proxy label rather than a verified refresh-success label, the model cannot establish that changing a page will improve search performance.

---

## 2. Data Safety

### Dataset

The capstone uses the **FlyRank ML Internship dataset** with the `fact_content_daily_performance` configuration/table.

The working analysis uses:

```python
RANDOM_STATE = 42
SAMPLE_SIZE = 10000
```

```python
sample = ds["train"].shuffle(seed=RANDOM_STATE).select(range(SAMPLE_SIZE)).to_pandas()
```

The shuffled sample contains **69 unique clients**. This number matters: an earlier, unshuffled version of this exact pipeline (`.select(range(10000))` without `.shuffle()` first) collapsed the sample to roughly 3 unique clients, which made grouped-by-client validation nearly meaningless. See Section 5 for the full record.

### Audited Model Features

The exact feature list used by every audited model is:

```text
gsc_impressions
ga4_pageviews
ga4_sessions
ga4_users
ga4_engaged_sessions
sessions_organic
sessions_direct
sessions_referral
sessions_social
sessions_paid
sessions_ai
scroll_events
```

### Target Definition

The Week-5 proxy target is:

```text
target = 1 when:
gsc_avg_position > 20
AND
gsc_clicks < 5
```

Otherwise:

```text
target = 0
```

In the shuffled 10,000-row sample, this produces:

| Target | Count | Share |
|---|---:|---:|
| 0 (not a candidate) | 9,018 | 90.2% |
| 1 (candidate) | 982 | 9.8% |

This is a real, moderate class imbalance (roughly 9:1), and it directly shaped the results in Section 6 — an unweighted model can score ~90% accuracy by predicting the majority class for every row.

This target is explicitly a **proxy**. It is not a human-verified label stating that a content item actually needs a refresh, and it does not measure whether a refresh would subsequently succeed.

### Target-Defining Fields

The following fields directly define the proxy target:

```text
gsc_avg_position
gsc_clicks
```

Therefore, they are deliberately excluded from every audited model's feature set — enforced programmatically from a single `TARGET_DEFINING_FIELDS` variable that also builds the target, so the exclusion list cannot silently drift out of sync with the target formula the way it did in the original run (Section 5).

### Identifiers

The following identifiers are not used as predictive features:

```text
client_hash_id
content_hash_id
```

`client_hash_id` is used for grouped validation.

`content_hash_id` is retained for content traceability in the action queue.

### Leakage Audit

The audit excludes the following fields structurally:

```text
client_hash_id
content_hash_id
report_date
target
gsc_clicks
gsc_avg_position
```

| Field | Decision | Reason |
|---|---|---|
| `client_hash_id` | EXCLUDED | Grouping identifier |
| `content_hash_id` | EXCLUDED | Content identifier |
| `report_date` | EXCLUDED | Validation/time field |
| `target` | EXCLUDED | Target-derived label |
| `gsc_clicks` | EXCLUDED | Direct target-defining signal |
| `gsc_avg_position` | EXCLUDED | Direct target-defining signal |

Beyond the structural check, a **quantitative trip-wire** trains a single-feature model per candidate feature and compares its accuracy against the dataset's majority-class baseline (90.2%) — flagging a feature only if it clearly beats that baseline *and* achieves non-trivial recall on the minority class. This catches leakage that a hand-maintained exclusion list might miss, and — importantly — does not falsely flag ordinary features just because they match the majority-class rate (see Section 5 for why the first version of this check needed a fix).

### Public Safety

The paper and report do not reproduce client names, private queries, credentials, private domains, or other client-identifying information.

The project does not claim to predict Google's ranking algorithm.

---

## 3. Baseline

### Week-4 Baseline

The project first established a transparent rule-based baseline.

The baseline is intentionally simple so that its reasoning can be understood and inspected directly.

In the final capstone comparison, the baseline is applied to the same held-out grouped test set used to evaluate every model.

```python
baseline_pred = (
    (sample.iloc[test_idx]["gsc_avg_position"] > 25) &
    (sample.iloc[test_idx]["gsc_clicks"] < 5)
).astype(int)
```

### Baseline Results (corrected pipeline)

| Metric | Week-4 Baseline |
|---|---:|
| Accuracy | **98.99%** |
| Precision | **100.00%** |
| Recall | **72.00%** |
| F1 | **83.72%** |

A machine-learning model should not automatically replace a simple rule. The baseline provides a transparent reference point — and, as shown in Section 6, it remains the strongest measured result even after every model fix applied in this revision.

---

## 4. Model / Analysis

### Models Compared

Four learned variants are compared against the baseline, to separate two distinct questions: *"does removing leakage change the result?"* and *"does addressing class imbalance change the result?"*

| Model | Configuration |
|---|---|
| Decision Tree (plain) | `max_depth=5`, `random_state=42` |
| Decision Tree (class-weighted) | `max_depth=5`, `class_weight="balanced"`, `random_state=42` |
| Random Forest (class-weighted) | `n_estimators=300`, `max_depth=8`, `class_weight="balanced"`, `random_state=42` |
| Random Forest (tuned threshold) | Same as above, with the classification threshold swept via `precision_recall_curve` and set to maximize F1 |

All four use the identical audited feature set — differences in results isolate the effect of class weighting and threshold tuning, not a change in features.

### Exact Features

Every model uses:

```text
gsc_impressions
ga4_pageviews
ga4_sessions
ga4_users
ga4_engaged_sessions
sessions_organic
sessions_direct
sessions_referral
sessions_social
sessions_paid
sessions_ai
scroll_events
```

### Deliberately Excluded Features

No model uses:

```text
gsc_clicks
gsc_avg_position
client_hash_id
content_hash_id
```

because the first two directly define the proxy target and the identifiers are used for grouping/reporting rather than prediction.

---

## 5. Audit Trail — What Was Wrong, and How It Was Found and Fixed

In the interest of an honest, reproducible capstone, three issues that inflated or distorted earlier results are documented here rather than silently corrected.

### 5.1 Target Leakage (found in an earlier run of `w05_model.ipynb` / `w06_validation_audit.ipynb`)

**Symptom:** `gsc_clicks` and `gsc_avg_position` were used to *build* the proxy target, but were also left in the *feature list* given to the model. The Decision Tree scored a suspicious **100% accuracy, precision, recall, and F1**, and its feature-importance output showed `gsc_avg_position` at exactly 1.0 with every other feature at 0.0 — a textbook leakage signature.

**Root cause:** an earlier leakage-check notebook (Week 3) marked both fields "safe" before any target existed. That judgement was correct at the time, but was never revisited once the target was defined using those same fields in Week 5.

**Fix:** `TARGET_DEFINING_FIELDS` is now defined once and used both to build the target and to filter the feature list, so the two cannot drift apart again.

### 5.2 Unshuffled Sampling (found in `w05_model.ipynb`, `w06_validation_audit.ipynb`, and an earlier version of `capstone.ipynb`)

**Symptom:** `ds["train"].select(range(10000))` without a preceding `.shuffle()` call drew the sample from the front of the dataset's storage order. This collapsed the working sample to roughly **3 unique clients**, and a grouped train/test split left only **1 client** in the test set — meaning "grouped validation" was, in practice, evaluating on a single client's idiosyncratic data rather than testing generalization across clients.

**Fix:** `.shuffle(seed=42)` is now applied before `.select()`, and the pipeline asserts a minimum client count (≥20 clients in-sample, ≥5 clients per side of the grouped split) before trusting any downstream metric. The corrected sample contains 69 unique clients; the grouped test split contains 14.

### 5.3 A Miscalibrated Leakage Trip-Wire (found while re-auditing after fix 5.1 and 5.2)

**Symptom:** while adding the quantitative leakage trip-wire described in Section 2, an early version flagged *every single feature* as "suspiciously predictive" at an identical accuracy of 0.902.

**Root cause:** the trip-wire compared each feature's single-feature accuracy against a fixed absolute threshold (0.90), without accounting for class imbalance. On a 90.2%-majority target, a shallow tree trained on almost any feature — including an uninformative one — will default to predicting the majority class for every row, landing at exactly the base rate (0.902). This is not leakage; it's an artifact of imbalance.

**Fix:** the trip-wire now compares each feature's accuracy against the dataset's actual majority-class baseline, and additionally requires the feature to achieve non-trivial recall on the minority class before it is flagged. Re-run with this fix, the audit correctly reports **PASS** for the corrected feature set.

---

## 6. Evaluation

### Validation Design

The final audit uses `GroupShuffleSplit` with:

```text
test_size = 0.20
random_state = 42
group = client_hash_id
```

| | Rows | Unique clients |
|---|---:|---:|
| Training | 9,306 | 55 |
| Testing | 694 | 14 |

**Client overlap: 0**

### Model Comparison (corrected pipeline)

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Week-4 baseline rule | **98.99%** | **100.00%** | 72.00% | **83.72%** |
| Decision Tree (plain) | 96.40% | 0.00% | 0.00% | 0.00% |
| Decision Tree (class-weighted) | 87.03% | 21.74% | **100.00%** | 35.71% |
| Random Forest (class-weighted) | 87.75% | 22.73% | **100.00%** | 37.04% |
| Random Forest (tuned threshold = 0.679) | — | 25.50% | **100.00%** | 40.74% |

All five rows are evaluated on the identical grouped held-out test set (694 rows, 14 clients).

### Confusion Matrix — Best Model (Week-4 Baseline)

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 669 | 0 |
| Actual 1 | 7 | 18 |

- True negatives: **669**
- False positives: **0**
- False negatives: **7**
- True positives: **18**

### Confusion Matrix Pattern — Plain Decision Tree

The unweighted Decision Tree predicted class 0 for every row in the test set (0 true positives, 25 false negatives out of 25 actual positives) — the model defaulted entirely to the majority class, which explains its 96.4% accuracy despite 0% recall and 0% F1.

### Interpretation

Two separate findings emerge once the audit fixes in Section 5 are applied:

1. **Leakage removal changed the result correctly.** Once `gsc_clicks`/`gsc_avg_position` are genuinely excluded (not just claimed to be), no model — including the plain Decision Tree — reaches anywhere near the earlier 100% score. This confirms the earlier result was leakage, not genuine model skill.
2. **Class imbalance, not weak features, explains the plain tree's 0% recall.** Adding `class_weight="balanced"` and, separately, tuning the classification threshold on Random Forest's probabilities, both **fully recovered recall to 100%** — every true refresh-candidate page was caught by the weighted/tuned models. The audited features therefore do carry usable minority-class signal; a plain, unweighted shallow tree simply wasn't using it.
3. **On F1, the simple baseline still wins.** The cost of recovering 100% recall in the learned models is precision: only 21.7%–25.5% of the pages they flagged were true positives, versus 100% for the rule-based baseline. F1, which balances both, favors the baseline (83.72%) over every learned variant (35.7%–40.7%).

This is a valid, and now leakage-free, negative result for F1 — but not a uniformly negative one. Whether the baseline or a balanced/tuned model is preferable depends on which mistake costs the team more: missing a real refresh candidate, or asking reviewers to check extra false alarms.

---

## 7. Interpretation

The strongest measured findings, after the audit and fixes documented in Section 5, are:

> The transparent baseline outperforms every tested learned model on F1 (83.72% vs. a best of 40.74% for a threshold-tuned Random Forest), on the same grouped, leakage-audited, properly-shuffled held-out test set.

> Class weighting and threshold tuning fully recover recall to 100% in the learned models — confirming that the audited features carry real, if limited, independent predictive signal for this proxy target, and that the plain tree's earlier 0% recall was a class-imbalance artifact rather than a sign the features are useless.

The Decision Tree / Random Forest feature-importance output remains descriptive. It should not be interpreted as evidence that a feature causes search rankings. A feature can be useful to a model without being a causal ranking factor.

---

## 8. Recommendation

The Week-7 action playbook converts observed signals into a practical content-review workflow.

### Recommendation 1 — REVIEW_REFRESH

**Reason code:**

```text
LOW_CLICKS_POOR_POSITION
```

This is the highest-priority review category. It should trigger investigation, not automatic content modification.

### Recommendation 2 — MONITOR

**Reason code:**

```text
MONITOR_SIGNAL
```

Items with weaker or incomplete opportunity signals can be monitored rather than immediately changed.

### Recommendation 3 — PROTECT

**Reason code:**

```text
PROTECT_STRONG_SIGNAL
```

Content without strong refresh signals should not be changed simply because it appears in an analytical ranking.

### Human Review Rules

Before acting on a ranked recommendation, a reviewer should check:

1. Search demand
2. Search intent
3. Content quality
4. Recent content changes
5. Seasonality
6. Business importance
7. Data quality

### What Should NOT Be Automated

The capstone should not automatically:

- publish content;
- rewrite content;
- delete content;
- prune pages;
- merge pages;
- guarantee that a page needs refreshing;
- guarantee ranking improvement;
- guarantee click improvement;
- guarantee traffic improvement;
- make causal claims;
- claim to predict Google's ranking algorithm.

Human review is required before editorial action.

### Cost / Value Thinking

The action queue is most useful when the cost of reviewing an item is lower than the potential value of identifying a meaningful content opportunity.

Practical workflow:

```text
Ranked queue
    ↓
Human review
    ↓
Check demand
    ↓
Check intent
    ↓
Check content quality
    ↓
Check recent changes
    ↓
Check seasonality
    ↓
Check business importance
    ↓
Editorial decision
```

---

## 9. Monitoring / Retrain Triggers

Potential monitoring triggers include:

- major changes in the underlying data distribution;
- changes in search or analytics instrumentation;
- substantial changes in class balance (the current 90.2%/9.8% split materially shapes every result above; a shift here should trigger re-evaluation);
- deterioration in precision;
- deterioration in recall;
- deterioration in F1;
- increasing false-negative rates;
- changes in business requirements;
- changes in the definition of content opportunity.

A future model should be re-evaluated when the underlying data changes materially, the proxy target changes, new features become available, the editorial decision changes, performance is re-measured on a newer validation period, or the baseline changes.

The baseline should remain part of future evaluation. Any future re-run should also re-verify the client count in the shuffled sample and re-run the quantitative leakage trip-wire described in Sections 2 and 5 before trusting new numbers.

---

## 10. Paper Artifacts

The capstone generates reusable figures for the research paper:

1. Baseline vs. model comparison (all five methods)
2. Precision/Recall/F1 vs. decision threshold (Random Forest tuning curve)
3. Opportunity score distribution
4. Reason-code distribution
5. Search impressions vs. clicks
6. Top-20 ranked opportunities
7. Feature importance (leakage-free)

Reusable figures are stored under:

```text
work/figures/
```

---

## 11. Ranked Action Queue

The capstone builds an auditable queue containing:

```text
rank
content_hash_id
opportunity_score
action
reason_code
gsc_impressions
gsc_clicks
gsc_avg_position
ctr
```

The opportunity score combines indicators for poor position, low clicks, and search volume.

The queue is a prioritization artifact, not a production content-management system.

---

## 12. Reproducibility

### Repository

https://github.com/yasir-ansari951/flyrank-ml-internship

### Weekly Notebooks (revised)

```text
work/notebooks/w03_feature_leakage_check.ipynb   (leakage check made conditional on target definition)
work/notebooks/w04_baseline_score.ipynb          (unchanged)
work/notebooks/w05_model.ipynb                   (target-defining fields excluded programmatically)
work/notebooks/w06_validation_audit.ipynb        (shuffle fix, client-count guard, corrected trip-wire)
work/notebooks/w07_action_playbook.ipynb         (unchanged — already shuffled correctly)
work/notebooks/capstone.ipynb                    (final: shuffled sample, leakage-free features,
                                                   class-weighted models, threshold tuning)
```

### Data Access

The final notebook uses:

```text
FlyRank/internship-warehouse
fact_content_daily_performance
```

Access requires the permitted Hugging Face dataset access and token. The token is not included in the repository.

### Configuration

```python
RANDOM_STATE = 42
SAMPLE_SIZE = 10000
```

Decision Tree (plain):

```text
max_depth = 5
random_state = 42
```

Decision Tree (balanced):

```text
max_depth = 5
class_weight = "balanced"
random_state = 42
```

Random Forest (balanced):

```text
n_estimators = 300
max_depth = 8
class_weight = "balanced"
random_state = 42
```

Validation:

```text
GroupShuffleSplit
test_size = 0.20
random_state = 42
groups = client_hash_id
minimum_clients_in_sample = 20   # assertion added after the audit trail in Section 5
minimum_clients_per_split_side = 5
```

### Outputs

The capstone regenerates:

```text
work/outputs/capstone_ranked_action_queue.csv
work/outputs/capstone_metrics.json
```

### Re-run Workflow

1. Clone the repository.
2. Obtain permitted FlyRank dataset access.
3. Provide the Hugging Face token through the notebook environment.
4. Load the dataset and **shuffle before sampling**.
5. Assert the sample contains at least 20 unique clients before proceeding.
6. Construct the proxy target and derive the excluded-feature list from the same `TARGET_DEFINING_FIELDS` variable.
7. Run the structural leakage check and the quantitative (imbalance-aware) trip-wire.
8. Create the grouped train/test split; assert at least 5 clients on each side; check zero client overlap.
9. Train the baseline, plain Decision Tree, class-weighted Decision Tree, class-weighted Random Forest.
10. Tune the Random Forest's decision threshold via `precision_recall_curve`, optimizing F1.
11. Evaluate all five methods on the identical held-out set.
12. Inspect the confusion matrix and errors for the best-F1 method.
13. Generate feature importance.
14. Generate the action queue.
15. Generate the paper figures.
16. Export the metrics receipt and queue.

---

## 13. Limitations & Honest Framing

### Proxy Target

The target is constructed from:

```text
gsc_avg_position > 20
AND
gsc_clicks < 5
```

It is not a human-verified label for whether content needs refreshing and does not measure whether a refresh is successful.

### No Causal Evidence

The project does not measure the effect of actually refreshing content.

Therefore, it cannot establish that refreshing a page causes higher rankings, more clicks, more traffic, or greater engagement.

### Working Sample

The final analysis uses a **10,000-row working sample** rather than the complete 78.8M-row warehouse. The sample is shuffled before selection (69 unique clients); a larger or differently-seeded sample could shift the exact numbers reported here.

### Class Imbalance

The proxy target is imbalanced (~9.8% positive). This materially affects every model's behavior: an unweighted model can score high accuracy while catching none of the minority class, and correcting for imbalance trades precision for recall rather than improving both simultaneously. Any reader of this report should weigh precision and recall according to their own cost of a false positive vs. a false negative, not rely on accuracy alone.

### Validation

Grouped-by-client validation is more conservative than a random row split for the intended unseen-client generalization question, and depends on the sample actually containing enough distinct clients (verified here: 69 in-sample, 14 in the test split) to be meaningful.

However, it does not guarantee performance on future time periods, new populations, or different datasets.

### Missing Editorial Context

The proxy target does not directly measure:

- content quality;
- search intent;
- business value;
- seasonality;
- competitor activity;
- editorial strategy;
- actual content changes.

These factors require human review.

### Safe Claim Language

Use:

- observed
- measured
- directional
- associated
- decision-support

Avoid causal claims about rankings or refresh outcomes.

---

## 14. Final Findings

The capstone tested whether Decision Tree and Random Forest models — plain, class-weighted, and threshold-tuned — could improve content-opportunity scoring beyond a transparent Week-4 baseline, after a self-audit caught and fixed a target-leakage bug and an unshuffled-sampling bug in an earlier run.

The experiment used the FlyRank ML Internship dataset, `fact_content_daily_performance`, a shuffled 10,000-row working sample (69 unique clients), a proxy target based on average position and clicks (9.8% positive), 12 audited historical search/analytics features excluded of all target-defining fields, four model variants, grouped validation by `client_hash_id` with a minimum-client-count guard, a 20% held-out test share (694 rows, 14 clients), and a zero-client-overlap check.

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Week-4 baseline rule | **98.99%** | **100.00%** | 72.00% | **83.72%** |
| Decision Tree (plain) | 96.40% | 0.00% | 0.00% | 0.00% |
| Decision Tree (balanced) | 87.03% | 21.74% | 100.00% | 35.71% |
| Random Forest (balanced) | 87.75% | 22.73% | 100.00% | 37.04% |
| Random Forest (tuned threshold) | — | 25.50% | 100.00% | 40.74% |

The best-F1 method's confusion matrix (Week-4 baseline):

```text
True negatives:  669
False positives:   0
False negatives:   7
True positives:   18
```

None of the learned models provided a useful measured F1 improvement over the transparent baseline. However, class weighting and threshold tuning did fully recover recall — a materially different, and more nuanced, finding than "the model failed."

---

## 15. Overall Conclusion

This project demonstrates two machine-learning lessons, reinforced by the audit process itself:

> **A more complex model is not automatically a better model** — and neither is a result that looks too good to be true. Both need to be checked before they're trusted.

The audited models were evaluated under a conservative grouped validation design, on a properly shuffled sample, after removing fields that directly defined the proxy target and after addressing class imbalance through weighting and threshold tuning. Under that setup, the transparent baseline substantially outperformed every learned variant on F1, while the learned variants fully recovered recall at a real cost to precision.

The capstone therefore does not present machine learning as a guaranteed improvement, and it does not present its own first-pass results as final. Instead, it presents an honest workflow for:

1. defining a practical content question;
2. establishing a transparent baseline;
3. testing machine-learning models, including weighted and threshold-tuned variants;
4. checking for leakage — structurally and quantitatively;
5. validating using grouped data, with a sanity check on group counts;
6. catching and documenting an implausible result (100% accuracy) rather than reporting it;
7. catching and documenting a flawed audit check rather than trusting it uncritically;
8. inspecting errors;
9. reporting a nuanced result — a negative finding on F1 alongside a positive finding on recall recovery;
10. converting observed signals into ranked human-review actions.

The final output is best described as:

> **A public-safe, directional decision-support workflow for prioritizing content review using historical search and analytics signals — built, audited, and corrected in the open.**

It should not be interpreted as a causal model of search rankings or as an automated content-refresh system.

---

## 16. Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset**.

The FlyRank dataset and internship learning environment provided the basis for this practical machine-learning workflow — including the review process that surfaced the leakage, sampling, and audit-calibration issues documented in Section 5.

Data source:

https://flyrank.ai

This report credits the data source while maintaining public-safe handling of the underlying data.

---

## 17. Final Public-Safety Statement

This report contains only public-safe methodological descriptions and aggregated model results.

It does not expose:

- client names;
- private URLs;
- private search queries;
- credentials;
- raw client exports;
- confidential business information.

The findings are limited to the documented sample, proxy target, feature set, models, and validation design.

The correct interpretation is:

> **Observed and measured evidence for directional content-review decision-support — not causal evidence of search-engine behavior or guaranteed refresh outcomes.**
