# Archive Index

This directory preserves non-core material that is not required for the main empirical pipeline but may be useful for provenance, debugging, or reconstruction. Files were moved here instead of deleted.

## Directory Map

- `assistant_context/`: local assistant/Codex context files that should not define the public-facing project.
- `handoff_notes/`: detailed development notes, historical decisions, and long-form handoff documents.
- `diagnostics/`: one-off diagnostic scripts used during data and model validation.
- `logs/`: model execution logs and feature-selection logs.
- `original_reference/`: inherited/original benchmark reference material and prior README snapshots.

## Main Pipeline Location

The active project remains outside `archive/`:

- US notebooks: `../model_notebooks/`
- Turkey notebooks: `../turkey_model_notebooks/`
- US data and evaluation: `../data/`
- Turkey data: `../turkey_data/`
- Prediction outputs: `../predictions/` and `../turkey_predictions/`
- Final figures: `../figures/` and `../images/`

## Notes

The files in this archive are intentionally kept for transparency. They are not required for:

```bash
python data/evaluate.py
python data/generate_figures.py
```
