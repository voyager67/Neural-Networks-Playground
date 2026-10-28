# Getting started

## The fastest route

1. Extract the repository archive.
2. Start a local static server from the project root (`python -m http.server 8080`).
3. Open `http://127.0.0.1:8080`.
4. Select a dataset, set a seed and architecture, then train.

Don't open `index.html` through a `file://` URL. JavaScript modules and service workers behave more reliably from localhost or a deployed HTTPS origin.

## Browser-only users

The web app needs no `npm install`, bundler, backend, API key, or model download. Browser local storage keeps the session on that device. Export JSON to carry a model elsewhere, or export CSV to inspect its synthetic examples.

## Python users

```bash
python -m pip install -r requirements.txt
python scripts/train_numpy.py --dataset xor --epochs 150 --seed 42
```

Use a virtual environment if you do not want to install packages into your system Python.

## GitHub Pages

Enable GitHub Actions as the Pages source in repository settings. `.github/workflows/static.yml` publishes the repository's static files when `main` changes. The Python CLI remains a local tool; Pages does not execute Python server processes.
