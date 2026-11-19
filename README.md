# Customer-Repurchase-Propensity-End-to-End-ML-Project

A leakage-proof, chronologically validated machine-learning project that estimates **which customers of an outdoor-gear e-commerce store will buy again within 180 days**, and turns that score into honest, budget-aware marketing recommendations.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange)
![Notebook](https://img.shields.io/badge/Jupyter-End--to--End-brightgreen)

> **Validation rigour is the star of this project.** Purchase behaviour in this dataset is only *weakly* predictable. A trustworthy, modest model is far more valuable than an inflated one, so the notebook proves each claim rather than asserting it.

---

## Table of Contents
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Key Results](#-key-results)
- [Recommendations](#-recommendations)
- [Leakage & Fit Safeguards](#-leakage--fit-safeguards)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Limitations](#-limitations)
- [Author](#-author)

---

## Business Problem

The company sells outdoor equipment (tents, climbing, apparel, footwear and more) through four channels (web, app, marketplace, store) in 12 countries. Customers buy **non-contractually**: nobody "cancels", they just stop ordering. Marketing wants to spend its retention budget (e-mail, push, loyalty perks, discounts) where it matters, **not on everyone**.

**ML formulation**

> At each month-end `T`, for every customer with at least one order up to `T`, estimate
> `p = P(customer places ≥ 1 non-cancelled order in (T, T + 180 days])`

- **Type:** supervised binary classification on a *customer × snapshot* panel
- **Decision:** rank customers each month, then act on the top and bottom of the list

**Success criteria (fixed before modelling)**

| Criterion | Threshold |
|---|---|
| Discrimination on the untouched test period | 95% CI lower bound of ROC-AUC > 0.50 **and** beats the "most-orders-wins" heuristic |
| Business usefulness | Top-decile lift ≥ 1.2× the base rate |
| Generalisation | Train–validation AUC gap ≤ 0.02 |
| Calibration | Expected calibration error ≤ 0.03 on test |
| Validity | Every leakage test passes |

---

## Dataset

- **Scale:** 26,000 customers · 90,000 orders · Jan 2024 to Sep 2026
- **Seven relational tables:** `customers`, `orders`, `order_items`, `payments`, `products`, `returns`, `shipments` (CSV, plus an optional `ecommerce.sqlite` the notebook checks for parity)

**Data audit highlights**
- 0 rows removed; all money, key and time-consistency checks pass
- Flagged: 1,482 `returned` orders with no return record, 33% missing phone numbers, 78 missing weights
- **Hazardous `status` columns**: they reflect the *end of the data*, not the moment of prediction, so they are never used as features
- Returns, cancellations and payment failures turned out to be statistically random in this data (AUC ≈ 0.50), which is why **repurchase** was chosen as the learnable target

---

## Approach

```
raw tables → cleaning & integrity audit → enriched tables
          → 22 monthly snapshots (Jun-2024 … Mar-2026)
               features from events ≤ T   ║   label from (T, T+180d]  (never mixed)
          → customer×snapshot panel (417k rows)
          → PURGED chronological split: train │ gap ≥ 180d │ validation │ gap │ test
          → feature reduction → scaling study → baselines → tuned LR / RF / GBM
          → fit diagnostics · leakage test suite · bootstrap CIs
          → lift / value analysis & recommendations
```

**Key design decisions**
1. **Snapshot panel** instead of a single cut-off: more training signal, and stability can be measured across time
2. **Purged chronological split:** a training snapshot is only used if `T_train + 180d ≤ T_deploy` (enforced with assertions)
3. **As-of features:** every table is cut at its own event timestamp
4. **Mathematically grounded features (112 candidates):** RFM, exponentially decayed frequency, empirical-Bayes-shrunk inter-purchase times, **BG/NBD "probability alive"**, Shannon entropy / HHI of basket mix, as-of experience and cyclical calendar encodings
5. **Model ladder:** prior → "most orders wins" heuristic → regularised logistic regression → random forest → gradient boosting, all judged on the same leak-free validation window
6. **Proof of no leakage:** as-of reconstruction, label shuffling, naive-vs-proper comparison, adversarial validation, customer hold-out

**Final model:** an ensemble of L2 logistic regression and gradient boosting (average of probabilities) on **18 features**, an a-priori domain core. All data-driven subsets were statistically tied with it.

---

## Key Results

| Item | Result |
|---|---|
| **Test AUC** (3 untouched snapshots) | **0.537** (95% CI 0.531 to 0.543) |
| Validation AUC / log-loss | ≈ 0.530 / 0.685 |
| PR-AUC | 0.514 (base rate 0.444) |
| **Top-decile lift** | **1.37×** the base rate |
| Calibration (ECE) | 0.030 |
| vs. "most orders wins" | AUC **+0.007** (95% CI +0.004 to +0.010): statistically better, practically small |
| Train–validation AUC gap | ≈ 0.014 to 0.020 for regularised models; flexible models visibly over-fit and extra capacity doesn't help |

**Reading these honestly:** an AUC of ≈ 0.54 means the model *tilts the odds*. It does **not** identify lapsers. The simple heuristic reaches almost the same top-decile lift, so the model's added value is mainly in the middle of the ranking and in usable probabilities.

---

## Recommendations

1. **Use the score to rank, not to promise.** Keep expectations modest.
2. **Don't discount high-propensity customers.** Under the notebook's explicit assumptions (47.7% gross margin, 3 pp uplift), a blanket 10% discount loses ≈ $16 per contact while a no-discount reminder earns ≈ $7. Customers who would have bought anyway get the discount for free.
3. **Make cheap, non-discount reminders the default.** At a $0.30 contact cost, even a 1 to 3 pp uplift is profitable for typical baskets. Reserve incentives for the **low-propensity / high-value** segment.
4. **Run a randomised 10% hold-out** for the next two campaigns. This turns propensity into measured **uplift by decile**, and is the single most valuable next step.
5. **Treat power buyers (≥ 9 orders) as an account-management segment.** They are the most predictable and most valuable.
6. **Monitor in production:** PSI of top features monthly (alert > 0.25), a 30/60-day partial-window leading indicator, and quarterly retraining with champion/challenger evaluation
7. **Improve the data:** cancellation and refund timestamps, web/app engagement, e-mail response history, stock-outs, true price/discount history, customer-service contacts

> The model estimates **propensity, not uplift**. Whether a campaign changes behaviour must be measured experimentally.

---

## Leakage & Fit Safeguards

- **As-of reconstruction is exact** (difference = 0) and is able to catch an injected bug
- **Label-shuffle test:** AUC ≈ 0.5
- **Three classic mistakes** (such as all-time counts) inflate AUC by up to **+0.26**. The pipeline avoids them.
- **Unseen-customer AUC equals seen-customer AUC**
- **Purged split** with asserted gaps between train, validation and test windows
- **Over/under-fit diagnostics:** strongly regularised models have small train–validation gaps, and added capacity doesn't improve validation

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<my-username>/<my-repo-name>.git
cd <my-repo-name>
```

### 2. Install dependencies
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib jupyter
```

### 3. Add the data
Place the seven CSV files (`customers`, `orders`, `order_items`, `payments`, `products`, `returns`, `shipments`) next to the notebook, or in a `./data` folder. An optional `ecommerce.sqlite` file in the same place enables the CSV-vs-database parity check.

### 4. Run the notebook
```bash
jupyter notebook ecommerce_repurchase_ml_project.ipynb
```
Choose **Kernel → Restart & Run All**.

### Outputs
The notebook writes to `outputs/`:
- `repurchase_model.joblib`: the fitted model, feature list and settings
- `test_scores.csv`: per-customer test-period scores, deciles and actuals
- `model_card.json`: chosen model, features, hyper-parameters and test metrics

---

## Repository Structure

```
├── ecommerce_repurchase_ml_project.ipynb   # Full end-to-end notebook
├── data/                                   # 7 CSV tables (+ optional ecommerce.sqlite), add locally
├── outputs/                                # Generated by the notebook
│   ├── repurchase_model.joblib
│   ├── test_scores.csv
│   └── model_card.json
└── README.md
```

---

## Limitations

- **The ceiling is low.** Purchases in this data behave like a seasonal random stream, with no behavioural, web or service signals.
- **Propensity ≠ uplift**, as noted above.
- **Covariate shift:** customer histories lengthen over time, so monitor and retrain.
- **Left-censored histories** (orders start in 2024 although customers signed up from 2021) and **one test season** (Jan to Mar snapshots, no Q4) limit generalisation claims.
- **Documented process amendments:** the feature-set selection rule and two leakage-test criteria were refined after inspecting results. Both changes are explained in the notebook cells.
- Tuning used a small two-snapshot validation window. The test set was scored once with the final model.

---

## Author

**Abayomi Abiodun**
Senior Data Scientist · Founder, Data Fanatics Community

If you found this useful, feel free to star the repo and open an issue with questions or suggestions.
