# GitHub Push Checklist

Use this checklist before publishing the repository.

## Required Validation

Run from the repository root:

```bash
python data/evaluate.py
python data/us_improvement.py
python data/generate_figures.py
python data/generate_results_visuals.py
```

Expected:

- `data/evaluation_results_us.csv`: 272 rows.
- `turkey_data/evaluation_results_tr.csv`: 340 rows.
- `data/evaluation_results_us_improved.csv`: 576 rows.
- `evaluation_summary.md` regenerated.
- 5 main paper PNG figures in `figures/`.
- 7 results-analysis PNG figures in `figures/`.
- 3 US-only robustness PNG figures in `figures/`.

## Repository Hygiene

- Root directory contains only project-facing files and active directories.
- Development handoff notes are under `archive/handoff_notes/`.
- Logs are under `archive/logs/`.
- One-off diagnostics are under `archive/diagnostics/`.
- Original README snapshots are under `archive/original_reference/`.
- Assistant context files are under `archive/assistant_context/`.

## Suggested First Commit

```bash
git init
git add -A
git status
git commit -m "Finalize US Turkey nowcasting benchmark package"
```

Then add your GitHub remote and push:

```bash
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Replace `<your-github-repo-url>` with the repository URL created on GitHub.
