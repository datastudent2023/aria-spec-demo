# ARIA-Style Spec-Driven Data Science (Demo)

This repository demonstrates a **spec-driven data science workflow**:
`spec → train → evaluate → explain → document`,
inspired by the research paper  
> *Spec-Driven AI for Science: The ARIA Framework* (arXiv:2510.11143, 2025).

It transforms data science from ad-hoc scripting into an **auditable, reproducible, and governed process**.

Overview
- Spec-driven, human-in-the-loop data analysis demo inspired by ARIA. Define a YAML spec; the runner builds a pipeline, selects a model via CV, evaluates on a test split, and emits artifacts (metrics, plots, model, report, lineage).

Quickstart
- Generate data:
  - python aria_run.py --generate-data
- Train from spec:
  - python aria_run.py --spec analysis.spec.yaml
- Artifacts:
  - runs/aria_demo/<timestamp>/artifacts/
    - model.pkl
    - metrics.json
    - feature_importance.png
    - lineage.json
    - report.md

Spec example (analysis.spec.yaml)
- Edit paths/features as needed.
```yaml
run:
  out_dir: ./runs/aria_demo
  random_state: 42
  test_size: 0.2
  cv_folds: 5
  task: regression
data:
  csv_path: data/housing.csv
  target: house_price
  features: [rooms, distance, crime_rate, age, income, school_score, has_park, near_subway, city]
constraints:
  allow_models: ["linear", "gbdt", "rf"]
  scale_numeric: true
  handle_unknown_category: ignore
explainability:
  permutation_importance: {enabled: true, n_repeats: 5}
deliverables:
  artifacts: ["model.pkl","metrics.json","feature_importance.png","lineage.json","report.md"]
```

What this demo does
- Preprocessing via ColumnTransformer (scaler for numeric, OHE for categoricals).
- Candidate models (configurable): Linear/Logistic, Gradient Boosting, Random Forest, optional XGBoost/LightGBM if installed.
- Model selection via cross_val_score; test evaluation on hold-out split.
- Explainability via permutation importance mapped to original features.

Optional helper
- Draft spec:
  - python aria_llm.py --prompt "predict housing prices" > analysis.spec.yaml
- Summarize report:
  - python aria_llm.py --report runs/aria_demo/<timestamp>/artifacts/report.md

FAQ
- PyCaret vs this:
  - PyCaret: rapid model comparison; great for quick baselines.
  - This demo: explicit spec, artifacts, and lineage for reproducible research.

Requirements
- Python 3.10+
- Core: numpy, pandas, pydantic>=2, pyyaml, scikit-learn, matplotlib
- Optional: xgboost, lightgbm


## ⚙️ Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (A) Generate example data + default spec
python aria_run.py --generate-data

# (B) Run the pipeline
python aria_run.py --spec analysis.spec.yaml