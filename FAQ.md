# FAQ

### Does this use a hosted AI model or API key?

No. The browser's small MLP engine is implemented in JavaScript and runs on your device. The Python reference engine runs locally with NumPy and Matplotlib.

### Why doesn't my model match the NumPy output exactly?

The engines use separate random number generators and may initialise data and weights differently. Same seeds make each path repeatable within that path, not bit-for-bit equivalent across paths.

### Why is accuracy high but my model still looks wrong?

Accuracy depends on the current set and class balance. Inspect the loss and boundary, raise noise, or try a harder dataset. Training accuracy alone does not tell you how the model generalises to unseen examples.

### Why does training feel slow with a large network?

This is an intentionally inspectable CPU demo. More hidden units and more examples increase the cost, and live visualisation also has work to do. Start with 8–8 hidden layers and 200–300 examples before scaling up.

### Does the app upload my data?

No training data or model weights are sent to a server by the application. The stylesheet requests optional Google Fonts; the app works with local fallback fonts when that request is unavailable. The optional service worker caches local app assets.

### Can I train an image classifier or large language model here?

No. The current learner focuses on small two-dimensional binary classification to make forward passes and gradients inspectable. It is not a general deep-learning framework.

### I cleared my browser storage. Can I recover my session?

Only if you previously exported a model or kept a backup of the browser's saved data. Export JSON and CSV periodically if you need to preserve results.
