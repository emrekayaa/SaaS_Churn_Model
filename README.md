# SaaS Churn Model

Predicting 90-day churn for a B2B SaaS product, and finding out how much signal the data actually has.

🇹🇷 [Türkçe versiyon](README.tr.md)

## Summary

Using the RavenStack synthetic SaaS dataset (500 accounts, 5 tables), I built a leakage-safe churn prediction setup: monthly snapshots, a 90-day prediction horizon and a time-based train/test split with a gap.

The main result isn't a high score. During EDA I found that usage and support-ticket dates don't line up with the account lifecycle (about half of them are dated before the account even signed up). The models confirmed it: those features carry almost no signal. The strongest predictor is account age. A one-line rule, "newer accounts are riskier", beats both logistic regression and LightGBM on the test set.

| Model (test set) | ROC-AUC | PR-AUC | Top-10% lift |
|---|---|---|---|
| Random scores | 0.466 | 0.258 | 0.78 |
| **Rule: newer accounts are riskier** | **0.646** | **0.439** | **1.96** |
| Logistic regression (all features) | 0.574 | 0.338 | 1.27 |
| Logistic regression (small feature set) | 0.621 | 0.386 | 1.54 |
| LightGBM (all features) | 0.537 | 0.300 | 1.18 |
| LightGBM (small feature set) | 0.574 | 0.343 | 1.36 |

Test-set churn rate is 28.1%, which is the PR-AUC of a random model. In the top 10% of accounts ranked by the tenure rule, 55% churn within 90 days.

## Data

[RavenStack](HTTPS://WWW.KAGGLE.COM/DATASETS/RIVALYTICS/SAAS-SUBSCRIPTION-AND-CHURN-ANALYTICS-DATASET/DATA
) is a fictional AI collaboration SaaS, generated with Python and ChatGPT for practice. It covers January 2023 to December 2024.

| Table | Rows | One row is |
|---|---|---|
| accounts | 500 | an account |
| subscriptions | 5,000 | a subscription period (~10 per account) |
| feature_usage | 25,000 | roughly one subscription, day and feature |
| support_tickets | 2,000 | a ticket |
| churn_events | 600 | a churn event |

## Data quality findings

Before modeling, I checked grain, joins and date consistency across all tables. Most problems were small, two were not.

**Usage and ticket dates are inconsistent with accounts.**
- 53% of usage rows and 54% of support tickets are dated before the account's signup date
- 77% of usage rows are dated before the start of the subscription they belong to
- In contrast, accounts, subscriptions and churn events are fully consistent: no subscription or churn happens before signup

My reading is that usage and ticket dates were generated independently from the other tables. I dropped pre-signup rows and treated usage at the account level, and noted early on that these features would probably be weak.

**`churn_flag` and `churn_events` disagree.**
- 110 accounts have `churn_flag = True`, 352 accounts have at least one churn event, and only 75 are in both
- Reactivations don't explain the gap (only 61 events are flagged as reactivations)

I defined the target from churn event dates, which are consistent with signup dates and allow a time-based setup. `churn_flag` is excluded from the model as it gives away the target.

**Smaller issues:** `usage_id` repeats 21 times (different records sharing an ID, not duplicates). `satisfaction_score` is empty for 41% of tickets (unrated tickets, kept as information instead of being filled with zeros).

**Churn rises over time.** Monthly churn per existing account goes from 4–5% in 2023 to about 14% in late 2024, with a spike to 23% in December 2024. This is a real increase in rate, not just more accounts.

## Problem setup

**Question:** At date T, will this account have a churn event in the next 90 days?

**Snapshots.** One snapshot on the 1st of each month from 2023-07-01 to 2024-10-01 (16 snapshots, 4,100 account-month rows). Each snapshot includes accounts that signed up before T. The last snapshot is 2024-10-01 because its 90-day window has to close before the data ends.

**Target.** `churn_90d = 1` if there is a churn event with T ≤ churn_date < T + 90 days. Overall positive rate: 22.5%.

**Features** (all computed from data before T only):
- Account: industry, country, referral source, tenure
- Subscriptions: subscriptions started before T, active MRR and seats at T, latest plan and billing cycle, upgrades and downgrades
- Usage: last 30 and 90 days, trend vs. the previous 30 days, days since last usage
- Support: ticket counts and open tickets at T. Resolution time, satisfaction and escalations only from tickets closed before T, because they aren't known yet for open tickets
- Churn history: past churn events and days since the last one

**Excluded to avoid leakage:** `churn_flag`, subscription `end_date` beyond what's known at T, `auto_renew_flag` (unclear when it changes), and churn reason, refund and feedback fields (recorded at or after churn). Account-level `plan_tier`, `seats` and `is_trial` were replaced with the subscription state at T, since it's unclear whether they reflect the current or original state.

**Train/test split.** Time-based:

```
Train: 2023-07 ... 2024-05     Gap: 2024-06, 2024-07     Test: 2024-08 ... 2024-10
       11 snapshots, 2,236 rows      not used                 3 snapshots, 1,189 rows
       churn rate 18.6%                                       churn rate 28.1%
```

The gap is needed because training labels look 90 days ahead. Without it, the last training labels would be based on churn events from the test period.

## Results

- **The tenure rule wins.** Random scores already reach 0.466 ROC-AUC on this test set, so differences under ±0.03–0.04 aren't meaningful. The tenure rule and the small logistic regression are within that range. Everything else is clearly worse.
- **More features hurt.** Dropping usage and support features raised logistic regression from 0.574 to 0.621. With the full set, the model was learning noise from the training period.
- **The tenure effect grows over time.** Tenure alone scores 0.576 on train and 0.646 on test. The model learns from a period where the relationship was weaker, so it underweights tenure (concept drift).
- **LightGBM overfits.** 0.993 ROC-AUC on train, 0.537 on test, even with small trees. My explanation: the same account appears up to 11 times in training with similar features, so the model can effectively recognize accounts and memorize their labels. Logistic regression can't do that (0.602 on train, 0.621 on test).

I didn't tune LightGBM using test results, as that would just be fitting to the test set.

## What drives churn

From the small logistic regression (standardized features, so effects are per 1 standard deviation):

- **Account age:** older accounts are less likely to churn (odds ~27% lower per SD)
- **Referral source:** organic accounts are more likely to churn (odds ~29% higher)
- **MRR:** higher-paying accounts are less likely to churn (odds ~22% lower)

These three agree with the descriptive analysis. Past-churn features and subscription count are correlated with each other and with tenure, so their individual coefficients aren't reliable. Industry and plan coefficients are small and contradict the descriptive tables, so I treat them as noise. Plan type and billing cycle (monthly vs. annual) show no difference in churn, which is unusual for real SaaS data.

**Practical takeaway:** prioritize new accounts in their first months, especially from organic channels and with low MRR. A simple age-based watchlist already gives about twice the hit rate of random selection.

## Limitations

- The data is synthetic, and the date inconsistencies suggest some relationships a real product would have (usage → churn, support → churn) weren't generated
- 500 accounts is small. Rows from the same account are correlated, so the effective sample size is much smaller than 4,100
- One test window (Aug–Oct 2024) during a period of rising churn. Results could look different in another window
- No hyperparameter tuning and no probability calibration

## If this were real data

- Check with the data owners why event dates don't match the account lifecycle before building anything
- Use time-based cross-validation over several windows instead of a single test period
- Group or deduplicate repeated account rows during training to prevent memorization
- Calibrate probabilities if they're used for prioritization, and measure the effect of interventions with an experiment

## Repository structure

```
SaaS_Churn_Model/
├── data/
│   ├── raw/                  # 5 CSVs from Kaggle (not committed)
│   └── processed/            # model_table.csv, created by notebook 02 (not committed)
├── notebooks/
│   ├── 01_eda.ipynb                   # data quality, grain, joins, target investigation
│   ├── 02_feature_engineering.ipynb   # snapshots, target, features
│   └── 03_modeling.ipynb              # split, baselines, models, interpretation
├── requirements.txt
└── README.md
```

## How to reproduce

```bash
git clone https://github.com/emrekayaa/SaaS_Churn_Model.git
cd SaaS_Churn_Model
python3 -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

1. Download the dataset from [Kaggle](HTTPS://WWW.KAGGLE.COM/DATASETS/RIVALYTICS/SAAS-SUBSCRIPTION-AND-CHURN-ANALYTICS-DATASET/DATA
) and put the 5 CSVs in `data/raw/`
2. Run the notebooks in order: `01_eda` → `02_feature_engineering` → `03_modeling`. Notebook 02 creates `data/processed/model_table.csv`, which notebook 03 reads

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, matplotlib, Jupyter

## Related

[SaaS Growth & Retention Analytics](https://github.com/emrekayaa/saas-growth-retention-analytics): SQL, statistical testing and a Power BI dashboard on a different SaaS dataset.
