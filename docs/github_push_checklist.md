# GitHub Push Checklist

Use this checklist before publishing the repository.

## Required Validation

Run from the repository root:

```bash
python data/evaluate.py
python data/generate_figures.py
```

Expected:

- `data/evaluation_results_us.csv`: 204 rows.
- `turkey_data/evaluation_results_tr.csv`: 204 rows.
- `evaluation_summary.md` regenerated.
- 25 PNG files across `images/` and `figures/`.

## Repository Hygiene

- Root directory contains only project-facing files and active directories.
- Development handoff notes are under `archive/handoff_notes/`.
- Logs are under `archive/logs/`.
- One-off diagnostics are under `archive/diagnostics/`.
- Original-reference material is under `archive/original_reference/`.
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
