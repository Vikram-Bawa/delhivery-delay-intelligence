# Delhivery Last-Mile Delay Intelligence Pipeline

A reproducible, end-to-end pipeline for analysing last-mile delivery
delays, predicting shipment-level delay risk, segmenting hubs by
operational performance, and generating practical actions for newly
arriving shipments.

The dataset used in this project is synthetic. Results demonstrate the
workflow and should not be interpreted as measured performance on live
Delhivery operations.

## 1. How to run

### Requirements

- Python 3.10 or later
- A virtual environment
- Packages listed in `requirements.txt`

### Windows PowerShell setup

Run these commands from the project root:

```powershell
cd C:\Users\vsbaw\delhivery-delay-intelligence

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Execution order

Run the scripts in the following order. Each script reads inputs from
`data/` and writes its results to the relevant `outputs/` or `models/`
folder.

```powershell
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`generate_data.py` creates the synthetic source data. Run it first when
rebuilding the project from scratch. Parts A–D perform cleaning/EDA,
delivery-time regression, delay classification, and hub segmentation.
`package_model.py` fits and saves the final classification pipeline and
its supporting metadata. `predict_delay.py` scores the supplied
new-shipment batch and writes recommended actions.

### Important input and output files

| Path                         | Purpose                                           |
|:-----------------------------|:--------------------------------------------------|
| `data/raw_shipments.csv`     | Historical synthetic shipment records             |
| `data/hubs.csv`              | Hub metadata                                      |
| `data/new_shipments.csv`     | New shipment batch for scoring                    |
| `data/modelling_table.csv`   | Cleaned feature table used by modelling           |
| `src/shipment_prep.py`       | Shared cleaning and feature-engineering logic     |
| `outputs/`                   | EDA, evaluation, segmentation and scoring outputs |
| `models/delay_model.pkl`     | Saved delay-classification pipeline               |
| `models/delay_threshold.csv` | Selected alert probability threshold              |
| `models/feature_order.csv`   | Required model feature order                      |
| `models/impute_values.csv`   | Historical imputation medians                     |

The scoring stage depends on the model artefacts, feature order,
historical imputation values and hub segment assignments created by the
earlier stages. Run the preceding stages in order if these files are
missing or have been regenerated.

## 2. Data findings

All numbers below are copied from the files in `outputs/` written by
`part_a_clean_eda.py`.

After cleaning, the modelling table has 1,200 unique shipments, of which
278 were delayed (delay rate 0.2317).

The data is synthetic.

1.  **Express parcels are delayed most often.** In
    `part_a_delay_by_group.csv`, Express has a delay rate of 0.4897 (119
    of 243 parcels), against 0.2056 for Standard (132 of 642) and 0.0857
    for Economy (27 of 315). Express has the tightest promise (36
    hours), so it has the least room for error.

2.  **Delay rate rises sharply from Tier 1 to Tier 3.** It goes from
    0.0400 (16 of 400 parcels) in Tier 1, to 0.2150 (86 of 400) in Tier
    2, to 0.4400 (176 of 400) in Tier 3. That is 11 times higher in Tier
    3 than in Tier 1.

3.  **Delay rate rises with distance.** By distance band it is 0.0219
    for 0-200 km (6 of 274), 0.1368 for 201-400 km (77 of 563), 0.4364
    for 401-600 km (120 of 275) and 0.8523 for 601+ km (75 of 88). Most
    parcels on very long lanes miss their promise.

4.  **Strongest correlations with `delivery_hours`** (ranked by absolute
    value in `part_a_correlations.csv`, `delivery_hours` itself
    excluded): `promised_hours` +0.8372, `distance_km` +0.4013 and
    `n_stops` +0.3247. The promise level and the lane length matter
    most.

5.  **Why `declared_value_inr` was log-transformed.** In the
    median-filled column the mean is INR 1,533.05 but the median is INR
    1,121.50, so the mean sits about 37% above the median because a few
    high-value parcels stretch the right tail. After the natural log the
    mean is 7.0277 and the median is 7.0224, nearly equal, so
    `log_declared_value` is far more symmetric and no longer dominated
    by a few extreme values.

## 3. Model results

### Regression

All numbers below are copied from `part_b_metrics.csv`,
`part_b_tree_depth.csv`, `part_b_gb_learning_rate.csv` and
`part_b_cv.csv`, written by `part_b_regression.py`. The split is 840
training, 180 validation and 180 test rows, and the scaler was fitted on
the training rows only (`part_b_checks.csv`: `train_scaled_max_abs_mean`
is about 4e-16, `test_scaled_max_abs_mean` is 0.2187).

- **Baseline.** Always predicting the training mean gives a validation
  R² of -0.0212 (RMSE 18.8552 hours). An R² near zero means the baseline
  explains none of the variation in delivery hours, so any real model
  must beat this number.

- **Winner on validation.** Linear regression has the lowest validation
  RMSE, 4.8912 hours (R² 0.9313). Gradient boosting is very close at
  4.9165 hours (R² 0.9306), then random forest at 5.4680 and the
  decision tree at 6.2796.

- **Bias and variance by tree depth.** Depth 1 is high-bias: train RMSE
  11.6334 and validation RMSE 12.1324 are both high, with a gap of only
  0.4990. Depth 2 (10.4114 validation) is also still underfitting. At
  the other end, the unlimited tree is high-variance: train RMSE 0.0000
  against validation RMSE 7.7890, a gap of 7.7890 hours, and depth 12
  shows the same pattern (gap 6.0560). The sweet spot is depth 6, with
  validation RMSE 6.2796.

- **Learning rate.** Of 0.01, 0.1 and 1.0, a rate of 0.1 is best
  (validation R² 0.9306). At 0.01 the model is too slow: 100 trees take
  steps that are too small to fit the pattern, leaving R² at 0.7127. At
  1.0 each step overshoots and the model is unstable, giving R² 0.8440.

- **Cross-validation disagrees with validation.** On the 1,020
  training-plus-validation rows (5 folds, one shared `KFold`), gradient
  boosting has a mean R² of 0.9352 (std 0.0044) and linear regression
  0.9261 (std 0.0048). The single 180-row validation set put linear
  regression ahead by only 0.0253 hours of RMSE.

- **Final model and what I would ship.** By the stated rule the final
  model is linear regression, and on the test rows it scores RMSE 4.5235
  hours and R² 0.9493 (scored once). If it were my choice I would ship
  gradient boosting: cross-validation uses far more data than one
  180-row split, so it is a steadier guide, and boosting can capture
  interactions that a linear model cannot. The gap between the two is
  small, so linear regression is an acceptable simpler fallback.

### Classification

All numbers below are copied from the seven CSV files written by
`part_c_classification.py`. The classification task predicts whether a
shipment will be delayed (`is_delayed`).

The data was split using stratification into 840 training, 180
validation and 180 test rows. The delayed-shipment counts were 195, 42
and 41 respectively.

### Classification baseline and model comparison

The blind baseline predicted every shipment as on-time. It achieved
76.67% accuracy because 138 of the 180 validation shipments were
on-time. However, it detected none of the delayed shipments, giving it
zero recall and zero F1. Therefore, accuracy alone is not a reliable
measure of usefulness for this delay-detection problem.

| Model                   |  TN |  FP |  FN |  TP | Accuracy | Precision | Recall |     F1 |
|:------------------------|----:|----:|----:|----:|---------:|----------:|-------:|-------:|
| Blind baseline          | 138 |   0 |  42 |   0 |   0.7667 |    0.0000 | 0.0000 | 0.0000 |
| Logistic Regression     | 128 |  10 |   7 |  35 |   0.9056 |    0.7778 | 0.8333 | 0.8046 |
| Decision Tree (depth 6) | 128 |  10 |  11 |  31 |   0.8833 |    0.7561 | 0.7381 | 0.7470 |
| Default Random Forest   | 132 |   6 |  11 |  31 |   0.9056 |    0.8378 | 0.7381 | 0.7848 |

### Feature importance and process noise

The seven highest-ranked features according to ANOVA F-scores were:

1.  `distance_km`
2.  `hub_tier`
3.  `hub_daily_capacity`
4.  `n_stops`
5.  `promised_hours`
6.  `rain_flag`
7.  `sale_event_flag`

All five process-noise features ranked outside the top seven. Their
combined Random Forest importance share was 0.2244 (22.44%).

| Noise feature        | F-score | F-rank | Random Forest importance |
|:---------------------|--------:|-------:|-------------------------:|
| `label_print_batch`  |  4.5847 |      8 |                   0.0460 |
| `tape_roll_no`       |  1.9052 |     12 |                   0.0476 |
| `sticker_serial_mod` |  1.2776 |     14 |                   0.0499 |
| `shelf_slot_no`      |  0.1393 |     16 |                   0.0436 |
| `scanner_build_no`   |  0.0230 |     17 |                   0.0374 |

This indicates that the five process-noise columns collectively
contribute some Random Forest importance, but none ranks among the seven
strongest features according to the ANOVA F-score ranking.

### Feature selection experiments

| Number of selected features (k) | Validation F1 |
|--------------------------------:|--------------:|
|                               5 |        0.6988 |
|                               7 |        0.8333 |
|                              10 |        0.7857 |
|                              12 |        0.8049 |
|                              17 |        0.7848 |

Selecting seven features produced the highest validation F1.

### Hyperparameter tuning

Grid Search evaluated 27 parameter combinations across five folds,
resulting in 135 fits. Randomized Search evaluated nine combinations
across five folds, resulting in 45 fits.

Grid Search achieved a higher best cross-validation accuracy (0.8952)
than Randomized Search (0.8905). However, neither tuned candidate
outperformed the default Random Forest using the same seven selected
features on the validation set.

| Candidate                           | Candidates | CV folds | Total fits | Best CV accuracy | Validation F1 |
|:------------------------------------|-----------:|---------:|-----------:|-----------------:|--------------:|
| Grid Search                         |         27 |        5 |        135 |           0.8952 |        0.7750 |
| Randomized Search                   |          9 |        5 |         45 |           0.8905 |        0.7848 |
| Default forest, same seven features |          1 |        0 |          1 |                — |        0.8333 |

The default forest was therefore selected according to the required
validation F1 comparison.

### Alert threshold selection

The operating threshold was selected by choosing the highest tested
probability threshold that maintained validation recall of at least
0.80.

The selected threshold was 0.5, giving:

- Validation precision: 0.7955
- Validation recall: 0.8333
- Validation F1: 0.8140

This threshold provides a balance between identifying delayed shipments
and limiting false alerts while satisfying the required recall
constraint.

### Final test evaluation

The final model was evaluated once on the untouched test set using the
selected threshold of 0.5.

| Metric    | Test result |
|:----------|------------:|
| Accuracy  |      0.8667 |
| Precision |      0.6889 |
| Recall    |      0.7561 |
| F1        |      0.7209 |

Test recall was lower than validation recall (0.7561 compared with
0.8333). This difference is reported transparently rather than hidden.
The test set contained only 41 delayed shipments, making its recall
particularly sensitive to individual classification errors.

### Final classification configuration

- Feature selector: `SelectKBest`
- Number of selected features: 7
- Classifier: Random Forest
- Number of estimators: 100
- Maximum depth: Unlimited
- Minimum samples per leaf: 1
- Alert threshold: 0.5
- Selected candidate: `default_forest_same_7_features`

## 4. Operations recommendation

`part_d_segments.py` profiles the 24 hubs using shipment volume, mean
delivery time, delay rate, mean distance and mean number of stops. The
clustering input consists of four operational profile measures:
`mean_delivery_hours`, `delay_rate`, `mean_distance_km` and
`mean_n_stops`. These measures are standardised with `StandardScaler`
before KMeans clustering.

KMeans was evaluated for K = 2, 3, 4 and 5 using `random_state=42` and
`n_init=10`. The separation ratio is the mean Euclidean distance between
hub pairs in different clusters divided by the mean Euclidean distance
between hub pairs in the same cluster.

|   K | Separation ratio |
|----:|-----------------:|
|   2 |           2.8940 |
|   3 |           5.3928 |
|   4 |           6.3047 |
|   5 |           6.8866 |

Although K=4 and K=5 have higher separation ratios, K=3 was retained for
the final business segmentation. The larger K values create small
clusters: K=4 includes clusters of four and five hubs, while K=5
includes clusters of two, three and four hubs. Such small groups are
less suitable for reliable, independently managed operational playbooks.
K=3 provides a more practical balance between separation and operational
group size, with 7–9 hubs in each segment.

### Final segment profile

The final cluster names were assigned by ordering the three clusters by
their mean delay rate: lowest as `Standard`, middle as `Watch`, and
highest as `Escalate`.

| Segment  | Hubs | Mean delivery hours | Delay rate | Mean distance (km) | Mean stops |
|:---------|-----:|--------------------:|-----------:|-------------------:|-----------:|
| Standard |    7 |             47.2396 |     0.0257 |           188.9296 |    55.3686 |
| Watch    |    8 |             56.8137 |     0.1550 |           314.1762 |    70.7175 |
| Escalate |    9 |             64.6504 |     0.4600 |           470.4757 |    84.3444 |

The segment profiles show a clear operational gradient. Standard hubs
have the lowest average delay rate and shorter average distances. Watch
hubs have intermediate values. Escalate hubs have the highest delay
rate, longest average distance and highest average number of stops.

### Hub membership

| Segment  | Hub IDs and cities                                                                                                                    |
|:---------|:--------------------------------------------------------------------------------------------------------------------------------------|
| Standard | HUB01 Mumbai; HUB02 Delhi; HUB03 Bengaluru; HUB04 Hyderabad; HUB05 Chennai; HUB06 Pune; HUB08 Ahmedabad                               |
| Watch    | HUB07 Kolkata; HUB09 Jaipur; HUB10 Lucknow; HUB11 Chandigarh; HUB12 Indore; HUB14 Coimbatore; HUB16 Surat; HUB22 Dehradun             |
| Escalate | HUB13 Nagpur; HUB15 Kochi; HUB17 Patna; HUB18 Ranchi; HUB19 Guwahati; HUB20 Raipur; HUB21 Bhubaneswar; HUB23 Siliguri; HUB24 Agartala |

### Tier-versus-segment mismatches

Four hubs have a segment label that differs from the simple expectation
implied by their hub tier:

| Hub   | City     | Tier | Segment  |
|:------|:---------|-----:|:---------|
| HUB07 | Kolkata  |    1 | Watch    |
| HUB13 | Nagpur   |    2 | Escalate |
| HUB15 | Kochi    |    2 | Escalate |
| HUB22 | Dehradun |    3 | Watch    |

These mismatches are useful review candidates rather than errors to be
automatically corrected. Tier is a pre-existing hub category, whereas
the segment is derived from observed operational profiles in this
synthetic dataset. The mismatch report helps identify hubs whose
observed performance differs from the tier-level expectation.

### Recommended operating playbooks

- **Standard:** Maintain current operating practices, continue routine
  monitoring, and use these hubs as the comparison group when reviewing
  changes.
- **Watch:** Review lane distance, stop counts, capacity and recent
  delay patterns. Use proactive customer communication when the
  shipment-level model flags a shipment.
- **Escalate:** Prioritise hub-manager review, investigate recurring
  long-distance and high-stop routes, and examine capacity or process
  constraints. Shipment-level flagged records in this segment receive
  the action `Escalate to hub manager`.

These recommendations are proposed actions based on the segment
profiles; they are not claims that an intervention has already improved
delivery performance.

## 5. Production plan

### Model packaging and reproducibility

`package_model.py` rebuilds the final classification pipeline using the
specified stratified 840/180/180 split and fits the model on the 840
training rows. It saves:

- `models/delay_model.pkl` — the fitted `SelectKBest` plus Random Forest
  pipeline.
- `models/delay_threshold.csv` — the selected probability threshold
  (0.5).
- `models/feature_order.csv` — the ordered list of 17 input features
  expected by the pipeline.
- `models/impute_values.csv` — historical imputation medians used during
  preprocessing.
- `outputs/part_d_reload_check.csv` — reload verification output.

The saved historical medians are 314.95 km for distance, 1.83 kg for
parcel weight and INR 1,121.50 for declared value. They are reused when
scoring new shipments rather than recalculated from the new batch.

The packaging verification reloaded the saved model and compared its
probabilities with the in-memory model on the test rows. The maximum
probability difference was 0.0, confirming an identical reload result
for that check.

### New-shipment scoring

`predict_delay.py` reads the 48 records in `data/new_shipments.csv`, hub
metadata, the saved model artefacts and the hub segment assignments. It
applies the shared cleaning and feature-engineering functions from
`src/shipment_prep.py`, uses the saved historical medians, and preserves
the saved 17-feature order.

A shipment is flagged when its predicted delay probability is greater
than or equal to 0.5. The action mapping is:

| Prediction and hub segment                   | Recommended action        |
|:---------------------------------------------|:--------------------------|
| Flagged and segment is `Escalate`            | `Escalate to hub manager` |
| Flagged and segment is `Watch` or `Standard` | `Proactive customer call` |
| Not flagged                                  | `No action`               |

The verified scoring run produced 48 output rows: 8 flagged and 40
unflagged. The resulting actions were 6 hub-manager escalations, 2
proactive customer calls and 40 no-action records. A repeated run
produced the same output; the SHA-256 hashes of the first and second
output files matched. This confirms repeatability for the supplied data
and environment, not universal determinism across all software versions
or platforms.

The final scoring file is `outputs/new_shipment_actions.csv`, with
columns in this order:

`shipment_id, hub_id, segment, delay_probability, flagged, action`

### Performance summary

| Task                     | Final test metric |       Result |
|:-------------------------|:------------------|-------------:|
| Delivery-time regression | RMSE              | 4.5235 hours |
| Delivery-time regression | R²                |       0.9493 |
| Delay classification     | Accuracy          |       0.8667 |
| Delay classification     | Precision         |       0.6889 |
| Delay classification     | Recall            |       0.7561 |
| Delay classification     | F1                |       0.7209 |

The classification test set contains 180 shipments, including only 41
delayed shipments. Test recall is lower than validation recall, so the
reported test metrics should be treated as an initial estimate rather
than a guarantee of live performance.

### Proposed production monitoring plan

The following are suggested operating controls for a future pilot. They
are proposed thresholds, not measured results or requirements already
validated by this synthetic project.

1.  **Daily operational monitoring:** Track shipment volume, flagged
    volume, observed delay rate and action counts by service type, hub
    tier, hub segment and major lane. Investigate abrupt changes and
    unusual alert-volume spikes.
2.  **Outcome-based model monitoring:** Once actual delivery outcomes
    become available, compare predictions with observed delays. Track
    precision, recall and F1 over rolling evaluation windows, and review
    false negatives because missed delays are operationally important.
3.  **Recall trigger:** Review the model and threshold if recall falls
    below 0.70 in two consecutive evaluation windows with sufficient
    labelled cases. Do not change the threshold based on a very small
    sample.
4.  **Feature drift:** Calculate Population Stability Index (PSI) for
    important input features against the training reference
    distribution. Treat PSI above 0.20 as a review trigger and PSI above
    0.30 as a stronger investigation trigger. These are proposed
    heuristics and should be calibrated with production data.
5.  **Periodic review:** Review performance and segment profiles monthly
    during a pilot. Consider retraining quarterly, or earlier when
    sustained drift or a material operational change is confirmed.
    Retraining should use newly labelled, representative data and repeat
    the full validation process.
6.  **Sale-event monitoring:** Monitor `sale_event_flag` and sale-period
    performance separately. A sale-period shift should be investigated
    before combining it with ordinary-period performance or changing the
    model.
7.  **Deployment controls:** Version the model and preprocessing
    artefacts together, validate input schema and missingness, log
    predictions and actions, and retain a rollback copy of the previous
    approved model.

## 6. Limitations

- **Synthetic data:** The dataset is synthetic. It cannot establish real
  Delhivery performance, causal relationships or expected business
  savings.
- **Limited sample size:** The historical dataset contains 1,200
  shipments and 24 hubs. The new scoring batch contains 48 shipments.
  These samples are too small to establish robust performance across all
  routes, seasons and operating conditions.
- **Test uncertainty:** Only 41 delayed shipments occur in the
  classification test set. A small number of changed predictions can
  materially change recall and precision.
- **Validation-to-test difference:** Classification recall decreased
  from 0.8333 on validation to 0.7561 on test. This gap should be
  monitored rather than concealed.
- **Probability calibration:** The model’s predicted probabilities have
  not been demonstrated to be calibrated. The 0.5 threshold is the
  selected operating threshold from the project experiment, not a
  universal risk boundary.
- **Limited live context:** The current feature set does not represent
  all possible real-time factors, such as live traffic, changing weather
  severity, route disruptions, real-time hub workload or dynamic vehicle
  availability.
- **Static hub segments:** Hub segments are based on aggregate
  historical profiles. They may change as operational conditions change
  and should be recalculated only through a controlled review.
- **No live intervention evaluation:** The recommended actions are
  rule-based outputs. This project has not measured whether customer
  calls or hub-manager escalations reduce delays.

A production pilot should therefore begin in monitoring or
decision-support mode, with human review of escalations. Broader
deployment should depend on validation using real, representative and
sufficiently recent operational data.
