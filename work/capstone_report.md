# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Yasir Ansari
- **Lane:** Refresh / Content Opportunity Scoring
- **Repository:** https://github.com/yasir-ansari951/flyrank-ml-internship
- **Dataset:** FlyRank ML Internship dataset
- **Table:** `fact_content_daily_performance`
- **Working sample:** 10,000 rows
- **Random state:** 42

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

The working unit is a historical content-performance observation from the FlyRank `fact_content_daily_performance` table.

The final capstone uses a working sample of **10,000 rows**.

### Output

The project produces:

- a proxy target;
- a transparent baseline;
- a Decision Tree model;
- grouped validation results;
- a leakage audit;
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

### Audited Model Features

The exact feature list used by the audited model is:

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

This target is explicitly a **proxy**. It is not a human-verified label stating that a content item actually needs a refresh, and it does not measure whether a refresh would subsequently succeed.

### Target-Defining Fields

The following fields directly define the proxy target:

```text
gsc_avg_position
gsc_clicks
```

Therefore, they are deliberately excluded from the audited Decision Tree feature set.

### Identifiers

The following identifiers are not used as predictive features:

```text
client_hash_id
content_hash_id
```

`client_hash_id` is used for grouped validation.

`content_hash_id` is retained for content traceability in the action queue.

### Leakage Audit

The final audit explicitly excludes:

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

### Public Safety

The paper and report do not reproduce client names, private queries, credentials, private domains, or other client-identifying information.

The project does not claim to predict Google's ranking algorithm.

---

## 3. Baseline

### Week-4 Baseline

The project first established a transparent rule-based baseline.

The baseline is intentionally simple so that its reasoning can be understood and inspected directly.

In the final capstone comparison, the baseline is applied to the same held-out grouped test set used to evaluate the Decision Tree.

The baseline rule used in the final comparison is:

```python
baseline_pred = (
    (sample.iloc[test_idx]["gsc_avg_position"] > 25) &
    (sample.iloc[test_idx]["gsc_clicks"] < 5)
).astype(int)
```

### Baseline Results

| Metric | Week-4 Baseline |
|---|---:|
| Accuracy | **95.35%** |
| Precision | **100.00%** |
| Recall | **88.96%** |
| F1 | **94.16%** |

A machine-learning model should not automatically replace a simple rule. The baseline provides a transparent reference point.

---

## 4. Model / Analysis

### Model

The audited model is a **Decision Tree Classifier** with:

```text
max_depth = 5
random_state = 42
```

The model is relatively interpretable and can represent nonlinear relationships while remaining simple enough for an audit.

### Exact Features

The Decision Tree uses:

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

The Decision Tree does not use:

```text
gsc_clicks
gsc_avg_position
client_hash_id
content_hash_id
```

because the first two directly define the proxy target and the identifiers are used for grouping/reporting rather than prediction.

---

## 5. Evaluation

### Validation Design

The final audit uses `GroupShuffleSplit` with:

```text
test_size = 0.20
random_state = 42
group = client_hash_id
```

The client-overlap check returns:

**0 overlapping clients**

### Model Results

| Metric | Audited Decision Tree |
|---|---:|
| Accuracy | **57.62%** |
| Precision | **39.47%** |
| Recall | **1.23%** |
| F1 | **2.38%** |

### Baseline vs Model

| Metric | Week-4 Baseline | Audited Decision Tree |
|---|---:|---:|
| Accuracy | **95.35%** | **57.62%** |
| Precision | **100.00%** | **39.47%** |
| Recall | **88.96%** | **1.23%** |
| F1 | **94.16%** | **2.38%** |

Both methods were evaluated on the same grouped held-out observations.

### Confusion Matrix

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 1,659 | 23 |
| Actual 1 | 1,208 | 15 |

- True negatives: **1,659**
- False positives: **23**
- False negatives: **1,208**
- True positives: **15**

The dominant error was false negatives.

### Interpretation

The measured result does not support the claim that the Decision Tree is better than the baseline.

Instead, the evidence supports the simpler baseline for this particular proxy target, feature set, working sample, Decision Tree configuration, and grouped validation design.

This is a valid negative result.

---

## 6. Interpretation

The strongest measured finding is:

> The transparent baseline outperformed the audited Decision Tree on the same grouped held-out test set.

The baseline measured **95.35% accuracy and 94.16% F1**.

The audited Decision Tree measured **57.62% accuracy and 2.38% F1**.

The Decision Tree feature-importance chart is descriptive. It should not be interpreted as evidence that a feature causes search rankings.

A feature can be useful to the model without being a causal ranking factor.

The failure to beat the baseline demonstrates why a strong transparent baseline matters.

---

## 7. Recommendation

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

## 8. Monitoring / Retrain Triggers

Potential monitoring triggers include:

- major changes in the underlying data distribution;
- changes in search or analytics instrumentation;
- substantial changes in class balance;
- deterioration in precision;
- deterioration in recall;
- deterioration in F1;
- increasing false-negative rates;
- changes in business requirements;
- changes in the definition of content opportunity.

A future model should be re-evaluated when the underlying data changes materially, the proxy target changes, new features become available, the editorial decision changes, performance is re-measured on a newer validation period, or the baseline changes.

The baseline should remain part of future evaluation.

---

## 9. Paper Artifacts

The capstone generates reusable figures for the research paper:

1. Baseline vs audited model performance
2. Opportunity score distribution
3. Reason-code distribution
4. Search impressions vs clicks
5. Top-20 ranked opportunities
6. Audited Decision Tree feature importance

Reusable figures are stored under:

```text
work/figures/
```

The capstone creates:

```text
work/figures/opportunity_score_distribution.png
work/figures/reason_code_distribution.png
work/figures/impressions_vs_clicks.png
work/figures/top20_opportunities.png
```

---

## 10. Ranked Action Queue

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

## 11. Reproducibility

### Repository

https://github.com/yasir-ansari951/flyrank-ml-internship

### Weekly Notebooks

```text
work/notebooks/w04_baseline_score.ipynb
work/notebooks/w05_model.ipynb
work/notebooks/w06_validation_audit.ipynb
work/notebooks/w07_action_playbook.ipynb
work/notebooks/capstone.ipynb
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

Decision Tree:

```text
max_depth = 5
random_state = 42
```

Validation:

```text
GroupShuffleSplit
test_size = 0.20
random_state = 42
groups = client_hash_id
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
4. Run the capstone notebook.
5. Load the 10,000-row working sample.
6. Construct the proxy target.
7. Apply the audited feature list.
8. Create the grouped train/test split.
9. Check client overlap.
10. Train the Decision Tree.
11. Evaluate the Decision Tree.
12. Evaluate the baseline on the same test set.
13. Inspect the confusion matrix and errors.
14. Run the leakage audit.
15. Generate feature importance.
16. Generate the action queue.
17. Generate the paper figures.
18. Export the metrics receipt and queue.

---

## 12. Limitations & Honest Framing

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

The final analysis uses a **10,000-row working sample** rather than the complete warehouse.

### Validation

Grouped-by-client validation is more conservative than a random row split for the intended unseen-client generalization question.

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

## 13. Final Findings

The capstone tested whether a shallow Decision Tree could improve content-opportunity scoring beyond a transparent Week-4 baseline.

The experiment used the FlyRank ML Internship dataset, `fact_content_daily_performance`, a 10,000-row working sample, a proxy target based on average position and clicks, 12 audited historical search/analytics features, a Decision Tree with `max_depth=5`, `random_state=42`, grouped validation by `client_hash_id`, a 20% held-out test share, and a zero-client-overlap check.

The baseline and Decision Tree were evaluated on the same grouped test set.

| Metric | Baseline | Decision Tree |
|---|---:|---:|
| Accuracy | **95.35%** | **57.62%** |
| Precision | **100.00%** | **39.47%** |
| Recall | **88.96%** | **1.23%** |
| F1 | **94.16%** | **2.38%** |

The Decision Tree confusion matrix showed:

```text
True negatives:   1,659
False positives:     23
False negatives:  1,208
True positives:      15
```

The dominant error was false negatives.

The Decision Tree did not provide a useful measured improvement over the transparent baseline.

---

## 14. Overall Conclusion

This project demonstrates an important machine-learning lesson:

> **A more complex model is not automatically a better model.**

The audited Decision Tree was evaluated under a more conservative grouped validation design after removing fields that directly defined the proxy target.

Under that setup, the transparent baseline substantially outperformed the Decision Tree.

The capstone therefore does not present machine learning as a guaranteed improvement.

Instead, it presents an honest workflow for:

1. defining a practical content question;
2. establishing a transparent baseline;
3. testing a machine-learning model;
4. checking for leakage;
5. validating using grouped data;
6. inspecting errors;
7. reporting negative results;
8. converting observed signals into ranked human-review actions.

The final output is best described as:

> **A public-safe, directional decision-support workflow for prioritizing content review using historical search and analytics signals.**

It should not be interpreted as a causal model of search rankings or as an automated content-refresh system.

---

## 15. Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset**.

The FlyRank dataset and internship learning environment provided the basis for this practical machine-learning workflow.

Data source:

https://flyrank.ai

This report credits the data source while maintaining public-safe handling of the underlying data.

---

## 16. Final Public-Safety Statement

This report contains only public-safe methodological descriptions and aggregated model results.

It does not expose:

- client names;
- private URLs;
- private search queries;
- credentials;
- raw client exports;
- confidential business information.

The findings are limited to the documented sample, proxy target, feature set, model, and validation design.

The correct interpretation is:

> **Observed and measured evidence for directional content-review decision-support — not causal evidence of search-engine behavior or guaranteed refresh outcomes.**
