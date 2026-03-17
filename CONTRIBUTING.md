# Contributing

Contributions are welcome. Please follow these steps:

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit with a clear message
4. Push to your fork and open a Pull Request

## Guidelines

- **Notebook changes:** Keep cells self-contained and ordered. Clear all outputs before committing (`Kernel > Restart & Clear Output`).
- **New models:** Add to the comparison table in `adv_model_compare.ipynb` Cell 6 and update results in the README.
- **Bug fixes:** Include a brief description of the bug and the fix in the PR description.
- **Code style:** Follow PEP 8. Use descriptive variable names.

## Adding a New Stock

To extend the analysis to additional tickers:

1. Add the ticker and sector to the `stocks` dict in Cell 2
2. Rerun Cells 9–10 to regenerate cross-stock results
3. Update the results tables in `README.md`

## Reporting Issues

Open a GitHub Issue with:
- A clear title describing the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python and library versions (`pip freeze`)
