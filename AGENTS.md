# Repository conventions

Follow [docs/CODE_STYLE.md](docs/CODE_STYLE.md) for Python modules, tests, and notebook code cells.

Keep runtime ranking based solely on image scores. Do not add stocking counts or annotation-dependent inference. Human labels belong to supervised training and evaluation.

After logic changes, run the relevant automated tests and a small real-input check. After notebook changes, execute the affected cells and compare cell counts. Keep training runs bounded during verification and preserve the released model files.
