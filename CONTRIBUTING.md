# Contributing to Neural Playground

Welcome! The goal is to make neural-network concepts more understandable through experiments people can inspect, reproduce, and discuss.

## Good first contributions

- Improve an explanation in the learning guide.
- Add a test for an edge case in the MLP engine.
- Improve keyboard navigation or screen-reader labels.
- Add a synthetic dataset with documentation and deterministic tests.
- Improve numerical stability or clarify optimiser behaviour.
- Add a reproducible experiment that records its configuration and limitations.

## Before opening a pull request

1. Open an issue first for large features or mathematical changes.
2. Keep changes focused and explain the problem they solve.
3. Run `npm test` and `npm run check`.
4. Run `python -m unittest discover -s tests -v`.
5. If changing the model math, add or update a finite-difference gradient check.
6. Update docs and note any differences between the JavaScript and NumPy paths.

## Project principles

- **Explainable over magical:** no hidden training service or opaque model API.
- **Reproducible by default:** record seeds, settings and clear metrics.
- **Honest evaluation:** label training metrics as training metrics; don't claim generalisation without held-out evaluation.
- **Accessible:** support keyboard interaction, readable contrasts, reduced motion and responsive layouts.
- **Privacy-conscious:** synthetic sample data only; no tracking or remote storage.

## Code style

Use small, named functions, straightforward control flow, comments for non-obvious mathematical choices, and dependency-light implementations. Avoid adding a framework or dependency unless the reason is documented.

By participating, please keep discussion respectful, constructive, and focused on the work.
