# Concepts: a small neural network from input to update

This guide pairs with the interactive Playground. The browser model is a fully connected multilayer perceptron (MLP) for two features and a binary label.

## 1. Forward pass

For an input vector \(x\), each layer calculates a weighted sum plus bias and applies a nonlinear activation:

\[
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \phi(z^{(l)})
\]

The output layer uses sigmoid, \(\sigma(z)=1/(1+e^{-z})\), so the single output can be interpreted as a class-1 probability. A threshold of 0.5 maps that probability to a class prediction.

## 2. Why a hidden non-linearity matters

A linear transformation followed by another linear transformation is still linear. A network needs nonlinear activations to represent decision boundaries such as XOR and concentric circles. Compare Tanh, ReLU, Leaky ReLU and Sigmoid in the control panel.

- **Tanh:** smooth, centred near zero, but can saturate.
- **ReLU:** returns zero for negative inputs and keeps positive inputs; simple and often effective, but a unit can become inactive.
- **Leaky ReLU:** gives negative inputs a small slope.
- **Sigmoid:** maps any real number to (0,1); it saturates for large magnitudes and is used in this lab's output layer.

## 3. Loss

Binary cross-entropy penalises a probability that disagrees with the binary target:

\[
\mathcal L = -\left(y\log p + (1-y)\log(1-p)\right)
\]

Loss is evaluated over examples. The interactive view shows training loss, not a held-out score.

## 4. Backpropagation and gradients

Backpropagation applies the chain rule backwards through the network. It computes how the objective changes with each weight and bias. For a sigmoid output and binary cross-entropy, the output logit delta is \(p-y\). Hidden-layer deltas multiply the downstream signal by that layer's activation derivative.

The Gradients view displays the last minibatch's gradient values. A large gradient means the current loss is locally sensitive to that parameter; it does not automatically mean that a weight should have a large value.

## 5. Optimisers

- **SGD:** updates each parameter using the current averaged minibatch gradient.
- **Momentum:** keeps a moving direction that can smooth updates and build speed along consistent directions.
- **Adam:** tracks first and second moments of gradients and uses bias-corrected estimates for adaptive parameter updates.

The learning rate controls the scale of updates. Too small may make learning slow; too large can make loss oscillate or diverge. There is no single optimizer or setting that wins every problem.

## 6. Decision boundaries

Every pixel or small region in the plot is evaluated by the current network. The colour represents the predicted probability, and the labelled points are training examples. A boundary that separates training points well can still perform poorly on new examples, especially when a network is overfit to noisy data.

## 7. Parameter count

A fully connected layer with `fan_in` inputs and `fan_out` neurons has `(fan_in + 1) * fan_out` trainable parameters: the extra column is the bias. Neural Playground uses this exact count for its metric.

## 8. Reproducibility

Record the dataset, sample count, noise, seed, architecture, activation, optimiser, learning rate, and epoch target. Repeat experiments with multiple seeds. For defensible generalisation claims, create separate train/validation/test sets and avoid choosing a model based on test-set results.
