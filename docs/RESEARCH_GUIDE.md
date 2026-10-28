# A small experiment protocol

Neural Playground makes it easy to run a demo. Turning the demo into a useful experiment requires more care.

## Write a question first

Example: *On the two-moons dataset, how does hidden activation affect the final training loss for a fixed architecture and budget?*

## Record the setup

- Dataset type, sample count, noise, and seed.
- Architecture and parameter count.
- Activation, optimiser, learning rate, batch size, and epoch budget.
- Software version, runtime, and whether the result is from JavaScript or NumPy.
- Exact metric definitions and which split they describe.

## Keep the comparison controlled

Change one variable at a time when possible. Give every configuration its own initial model, but keep data and training budgets consistent. Repeat several seeds; avoid presenting one lucky run as the general result. Compare both convergence and final score, not just one metric.

## Avoid common mistakes

- **Training accuracy is not test accuracy.** This app's built-in interface reports training-set metrics only.
- **Lower loss is not always the full story.** Inspect decision boundaries, stability, and errors in different regions.
- **A 2D toy dataset is not a realistic benchmark.** The result teaches a mechanism, not necessarily how a model will behave on natural data.
- **Random initialisation matters.** Use repeated seeds to estimate variability.
- **Time results are hardware/browser dependent.** Do not compare raw milliseconds across different machines as a model-quality metric.

## Example NumPy run

```bash
python scripts/train_numpy.py --dataset circles --samples 400 --epochs 200 --seed 13 --activation tanh --optimizer adam --output outputs/circles-seed-13
```

The output folder contains a configuration summary, exact generated samples, training history, model weights, and plots. Share the configuration and outputs alongside any claim so others can reproduce it.

## Suggested report template

1. Research question and hypothesis.
2. Dataset generation details.
3. Model and parameter count.
4. Training procedure and hyperparameters.
5. Results across seeds (mean, standard deviation, failure cases).
6. Visual boundary and loss plots.
7. Limitations and what would change your conclusion.
