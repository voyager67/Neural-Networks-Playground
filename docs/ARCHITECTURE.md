# Architecture overview

Neural Playground has two intentionally separate execution paths that share the same teaching goals.

## Browser application

```text
index.html + styles.css
       │
       ▼
      app.js ───────► DOM controls, SVG network, Canvas plots, localStorage
       │
       ▼
src/ml-engine.js ──► synthetic datasets, MLP forward/backprop, optimisers, metrics
```

The static site is suitable for GitHub Pages. Training runs in the browser and is not sent to a server. The app persists a serialised model, dataset, history, controls, and experiment results in browser local storage. Model JSON and synthetic dataset CSV export are user-initiated downloads.

### Model layout

- Input: 2 numeric features.
- Hidden layers: 1–4 configurable dense layers, 2–32 neurons per layer.
- Hidden activation: Tanh, ReLU, Leaky ReLU or Sigmoid.
- Output: one sigmoid probability.
- Loss: binary cross-entropy.
- Optimisers: SGD, Momentum, Adam.
- Mini-batch size: 24 in the interactive controls.

Weights are stored as one matrix per layer. Each matrix has one row per neuron and one column per incoming connection plus a final bias column. Backpropagation computes averaged minibatch gradients; optimiser state is reset when a fresh model is constructed or a different optimiser is selected.

### Visualisation

- The decision-boundary canvas evaluates a grid of feature coordinates under the current model and overlays the generated examples.
- The SVG topology draws each connection with sign-sensitive colour and magnitude-sensitive opacity/width. Its alternate modes display the last minibatch's gradients or neuron activations.
- The inspector reads the selected node's activation and associated row of weights/gradients.
- The loss chart draws the recorded training loss across completed epochs.

## NumPy reference implementation

`neural_playground/engine.py` implements the same MLP concepts using NumPy arrays. `scripts/train_numpy.py` provides a local CLI to generate datasets, train the model, save a JSON model and CSV artifacts, and render a loss curve and a decision-boundary figure with Matplotlib.

The JS and NumPy engines are kept intentionally readable rather than highly optimised. They use different random number generators, so same-seed results are reproducible within each implementation but not expected to match across implementations.

## Reliability and limits

The supplied datasets are synthetic and labels are binary. The current app does not implement validation/test splits, multi-class output, convolutional or recurrent layers, GPU training, or a general model import format. Training accuracy is not a measure of generalisation. This is an educational tool—not a production ML framework or a scientific benchmark by itself.
