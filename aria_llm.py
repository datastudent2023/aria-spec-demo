#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# optional LLM helper: draft spec from natural language or summarize report
def draft_spec_from_prompt(prompt: str) -> str:
    return f"""# Prompt: {prompt}
run:
  out_dir: ./runs/aria_demo
  random_state: 42
  test_size: 0.2
  cv_folds: 5
  task: regression
data:
  csv_path: ./data/housing.csv
  target: house_price
  features: [rooms, distance, crime_rate, age, income, school_score, has_park, near_subway, city]
constraints:
  allow_models: [linear, gbdt, rf]
  scale_numeric: true
  handle_unknown_category: ignore
explainability:
  permutation_importance: {{enabled: true, n_repeats: 5}}
deliverables:
  artifacts: [model.pkl, metrics.json, feature_importance.png, lineage.json, report.md]
"""


def summarize_report(path: str) -> str:
    try:
        text = open(path, "r", encoding="utf-8").read()
    except Exception as e:
        return f"Error reading report: {e}"
    # Minimal stub; expand as needed
    return "Executive Summary (stub): key model, metrics, top features summarized."


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", help="Free-text prompt to draft a YAML spec")
    ap.add_argument("--report", help="Path to a generated report.md to summarize")
    args = ap.parse_args()

    if args.prompt:
        print(draft_spec_from_prompt(args.prompt))
    elif args.report:
        print(summarize_report(args.report))
    else:
        print(
            "Usage:\n"
            "  python aria_llm.py --prompt 'predict housing prices'\n"
            "  python aria_llm.py --report runs/.../report.md"
        )