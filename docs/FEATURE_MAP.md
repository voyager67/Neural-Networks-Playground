# Feature map

| Feature | Entry point | Notes |
|---|---|---|
| Synthetic dataset generation | `src/ml-engine.js` → `createDataset` | XOR, circles, spiral, moons, blobs; seeded RNG |
| Network forward pass | `src/ml-engine.js` → `MLP.forward` | Dense matrices, bias stored in last column |
| Backpropagation | `src/ml-engine.js` → `MLP.gradientsFor` | BCE + sigmoid output derivative; hidden activation derivatives |
| Optimisers | `src/ml-engine.js` → `MLP.trainBatch` | SGD, Momentum, Adam with bias correction |
| Epoch loop and live updates | `app.js` → `startTraining` | Pause/resume and one-epoch step |
| Decision boundary | `app.js` → `renderBoundary` | Grid of probability predictions plus labelled points |
| Network topology | `app.js` → `renderNetwork` | SVG edges; weight / gradient / activation modes |
| Neuron inspector | `app.js` → `renderInspector` | Activation, bias, incoming row and gradients |
| Controlled comparison | `app.js` → `runComparison` | Four fixed configurations trained on current data |
| Model JSON | `MLP.toJSON`, `MLP.fromJSON` | Validates declared architecture and weight shapes |
| Dataset CSV | `csvForDataset` | x1, x2, label |
| NumPy reference | `neural_playground/engine.py` | Matrix-based equivalent for CLI experiments |
| Matplotlib report | `scripts/train_numpy.py` | Loss curve, boundary plot, experiment JSON and CSVs |
