# Flipkart Gridlock Hackathon

## Setup

**Create and activate virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Run the Training*

```bash
python train_lgbm.py --train <train_dataset_path> --test <test_dataset_path> --output <prediction_file_path>
```

**Troubleshooting**
- If you see errors about OpenMP or `libomp.dylib` when installing LightGBM, prefer the XGBoost-based pipeline included here or use conda to install the binary packages.
    - I used `brew install libomp` to resolve this on macOS
- If datasets are large, increase system memory or use smaller `num_boost_round` in `train_lgbm.py`.

**Files**
- `train_lgbm.py`: feature engineering + XGBoost training pipeline
- `requirements.txt`: pip-installable deps