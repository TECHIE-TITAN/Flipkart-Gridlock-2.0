# Best solution

Use a feature-engineered gradient-boosted tree pipeline as your main approach, with LightGBM or XGBoost as the core model, time-aware validation, and SHAP for explanation.

That is the best fit I found for your setting because papers on limited tabular spatiotemporal demand data repeatedly show boosted trees beating single trees, random forests, and often neural baselines, while still exposing clear feature importance (Saa17, Din16, Zha15). The more complex graph and sequence models tend to pay off when you have richer lagged demand histories, neighbor states, or OD structure than this dataset clearly provides (Ke17, Zha21c). A stacked ensemble can help later, but only after the boosted-tree baseline is strong (Jin20b).

# What to build first

| Component | Recommendation | Why |
|------------|----------------|------|
| Main model | LightGBM or XGBoost regressor | Best accuracy/explainability tradeoff on this kind of data (Saa17, Din16, Lar21) |
| Validation | Blocked time split using timestamp | Prevents leakage and matches forecasting |
| Features | Heavy feature engineering from timestamp, geohash, weather, road attributes | This is where most of the lift will come from (Din16, Hou23, Dav18b) |
| Explanations | SHAP + feature importance + partial dependence | Standard way to explain boosted trees in transport tasks (Lee23) |
| Final boost | Small ensemble of 2 to 4 tree models | Often gives a modest leaderboard gain (Jin20b) |

# Implementation plan

## 1. Build a leak-safe validation setup

Do this before any modeling.

- Sort by timestamp.
- Use time-based folds, not random K-fold.
- Keep the last time block as your main validation set.
- If data covers multiple locations across the same time range, use 4 to 5 expanding or rolling time folds.

### Rule

Any feature using demand must be computed from past rows only for validation.

If you use grouped averages like mean demand by geohash-hour, build them out of fold on training data.

If you skip this, your CV score will lie to you.

---

## 2. Engineer the highest-value features

This is the core of the solution.

### Time features from timestamp and day

Create:

- hour
- minute
- 10-minute or 15-minute bucket
- day of week
- weekend flag
- part of day
- week of month

### Cyclical encodings

- sin_hour
- cos_hour
- sin_dow
- cos_dow

The literature consistently finds temporal structure among the strongest predictors (Saa17, Din16).

### Spatial features from geohash

Create:

- raw geohash as categorical
- geohash prefixes at shorter lengths
- decoded latitude and longitude from geohash center

### Interactions

- geohash × hour
- geohash × weekday
- geohash × weather

Location and time interactions are usually more useful than either alone (Saa17, Dav18b, Hou23).

### Context features

Use directly and in interactions:

- RoadType
- NumberofLanes
- LargeVehicles
- Landmarks
- Temperature
- Weather

### Additional interactions

- road type × hour
- number of lanes × hour
- weather × hour
- temperature × weather
- geohash × road type

Road attributes may not dominate alone, but they often matter through interactions (Kim20, Huk25).

---

## 3. Add historical demand features carefully

This is where you decide based on the split.

### If test timestamps are strictly after train timestamps

Add causal features such as:

- lag demand by geohash: 1, 2, 3, 6, 12, 24 steps
- rolling mean by geohash
- rolling std by geohash
- same-hour previous day or previous week

### If test is not strictly future or you are unsure

Avoid raw lags at first. Use safer aggregate features:

- mean demand by geohash
- mean demand by geohash × hour
- mean demand by geohash × weekday
- mean demand by road type × hour
- mean demand by weather × hour

Compute these with OOF target encoding on train, then fit lookup tables on full train for test.

This is the most competition-safe way to use demand history without leakage.

---

## 4. Train a strong tree baseline

Start with one model only.

### LightGBM starter settings

```yaml
objective: regression
metric: custom R² tracking if possible, else RMSE/MAE plus offline R²
learning_rate: 0.03 to 0.08
num_leaves: 31 to 127
max_depth: 6 to 10
feature_fraction: 0.7 to 0.95
bagging_fraction: 0.7 to 0.95
min_data_in_leaf: tune
early_stopping: validation fold
```

### Then tune

- number of leaves
- min child samples
- lambda L1 and L2
- subsampling
- interaction depth equivalent via tree size

Boosted trees were the strongest and most stable family in the most relevant forecasting papers I found (Saa17, Din16).

---

## 5. Add a second tree model, then blend

After LightGBM works, train:

- XGBoost
- optionally CatBoost if your categoricals are messy

Then blend predictions:

- simple weighted average first
- only try stacking if base models are already strong

A heterogeneous stack can outperform single models, but the gain is usually incremental compared with just getting the feature pipeline right (Jin20b).

---

## 6. Make it explainable from day one

For your final model:

### Global explainability

- global SHAP summary
- top 20 features by mean absolute SHAP

### SHAP dependence plots for

- hour
- geohash
- temperature
- weather
- number of lanes

### Local explanations

- a few high-demand examples
- a few low-demand examples

This gives you a clear story about what drives predictions, and this pattern has been used effectively for explainable transport forecasting (Lee23).

---

# What I would not prioritize

- Deep GNN traffic papers as your first implementation
- LSTM-only models
- Random forest as the main model
- Pure one-hot linear models except as a sanity baseline

Those are either less transferable to this dataset or usually weaker than boosted trees in comparable tabular forecasting settings (Saa17, Din16).

---

# Practical order of work

1. Build time-split CV
2. Train LightGBM with basic time + geohash + weather + road features
3. Add interaction features
4. Add OOF aggregated demand encodings
5. Train XGBoost
6. Blend
7. Run SHAP
8. Only then test stacking or lag-heavy variants

---

# My strongest recommendation

If you want one concrete starting point:

- **Model:** LightGBM
- **Feature strategy:** timestamp decomposition + decoded geohash + categorical interactions + OOF demand aggregates
- **Validation:** rolling time split
- **Explanation:** SHAP

That is the best balance of leaderboard performance, implementation speed, and explainability supported by the papers I found (Saa17, Din16, Lee23).