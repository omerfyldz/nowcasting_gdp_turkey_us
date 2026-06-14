# Repository Layout

This project is organized as a research-code repository with the active pipeline
files in the main tree.

## Top-Level Files

- `paper .pdf`: the full paper.
- `README.md`: project overview, data, models, results, and reproduction steps.
- `requirements.txt`: core Python dependencies.

## Active Directories

- `data/`: US data construction, feature selection, helper code, evaluation, figure generation, and US result CSVs.
- `turkey_data/`: Türkiye data construction, metadata, variable lists, helper code, and Türkiye result CSVs.
- `model_notebooks/`: US model notebooks (17 models).
- `turkey_model_notebooks/`: Türkiye model notebooks (17 models, plus R scripts for BVAR/MIDAS).
- `predictions/`: finalized US nowcast CSVs (17 models + 7 forecast combinations, per vintage).
- `turkey_predictions/`: finalized Türkiye nowcast CSVs (17 models × 5 vintages).
- `figures/`: paper-facing generated figures plus figure-index files.
- `docs/`: the US BVAR Category-2 fallback log and this layout note.

## Main Reproduction Commands

```bash
python data/evaluate.py
python data/us_improvement.py
python data/generate_figures.py
python data/generate_results_visuals.py
```
