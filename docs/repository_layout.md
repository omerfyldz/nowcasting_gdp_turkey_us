# Repository Layout

This project is organized as a research-code repository with active pipeline files in the main tree and preserved development material in `archive/`.

## Active Directories

- `data/`: US data construction, feature selection, helper code, evaluation, and figure generation.
- `turkey_data/`: Turkey data construction, metadata, variable lists, and helper code.
- `model_notebooks/`: active US model notebooks.
- `turkey_model_notebooks/`: active Turkey model notebooks.
- `predictions/`: finalized US prediction CSVs.
- `turkey_predictions/`: finalized Turkey prediction CSVs.
- `figures/`: main paper-facing generated figures.
- `docs/`: status, audit, and layout documentation.

## Archived Directories

- `archive/handoff_notes/`: development handoff documents.
- `archive/logs/`: execution logs.
- `archive/diagnostics/`: one-off diagnostic scripts.
- `archive/original_reference/`: inherited original README snapshots.
- `archive/assistant_context/`: local assistant context files.

## Main Reproduction Commands

```bash
python data/evaluate.py
python data/generate_figures.py
```

These commands do not depend on `archive/`.
