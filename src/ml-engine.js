/**
 * Neural Playground — small, inspectable MLP implementation.
 * No framework, model API, or external runtime required.
 */

export const DATASETS = [
  { id: "xor", name: "XOR", description: "A classic non-linear decision problem.", recommended: "tanh · Adam" },
  { id: "circles", name: "Concentric circles", description: "Separate an inner disk from an outer ring.", recommended: "tanh · Adam" },
  { id: "spiral", name: "Two spirals", description: "A harder, tightly interleaved classification task.", recommended: "tanh · Adam" },
  { id: "moons", name: "Interleaving moons", description: "Find a curved boundary between two half-moons.", recommended: "ReLU · Adam" },
  { id: "blobs", name: "Gaussian blobs", description: "A simple baseline with two clustered classes.", recommended: "ReLU · SGD" },
];

export function makeRng(seed = 42) {
  let state = (Number(seed) >>> 0) || 1;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function gaussian(rng) {
  const u = Math.max(1e-10, rng());
  const v = Math.max(1e-10, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function shuffledIndices(length, rng) {
  const indices = Array.from({ length }, (_, i) => i);
  for (let i = length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}

export function createDataset(kind = "xor", count = 240, seed = 42, noise = 0.13) {
  const rng = makeRng(seed);
  const X = [];
  const y = [];
  const n = Math.max(20, Math.min(1600, Math.floor(count)));
  const sigma = Math.max(0.005, Math.min(0.5, Number(noise)));
  const push = (x1, x2, label) => {
    X.push([Math.max(-1.25, Math.min(1.25, x1)), Math.max(-1.25, Math.min(1.25, x2))]);
    y.push(label);
  };

  for (let i = 0; i < n; i++) {
    if (kind === "xor") {
      const x = rng() * 2 - 1;
      const z = rng() * 2 - 1;
      const label = (x < 0) !== (z < 0) ? 1 : 0;
      push(x + gaussian(rng) * sigma * 0.35, z + gaussian(rng) * sigma * 0.35, label);
    } else if (kind === "circles") {
      const label = rng() < 0.5 ? 0 : 1;
      const radius = label ? 0.69 : 0.31;
      const angle = rng() * Math.PI * 2;
      const r = radius + gaussian(rng) * sigma * 0.48;
      push(Math.cos(angle) * r, Math.sin(angle) * r, label);
    } else if (kind === "spiral") {
      const label = i % 2;
      const localIndex = Math.floor(i / 2);
      const pointsPerArm = Math.ceil(n / 2);
      const t = pointsPerArm <= 1 ? 0 : localIndex / (pointsPerArm - 1);
      const angle = t * Math.PI * 3.5 + (label ? Math.PI : 0);
      const radius = 0.08 + t * 0.92;
      push(Math.cos(angle) * radius + gaussian(rng) * sigma * 0.34,
           Math.sin(angle) * radius + gaussian(rng) * sigma * 0.34, label);
    } else if (kind === "moons") {
      const label = i % 2;
      const angle = rng() * Math.PI;
      let x, z;
      if (label === 0) {
        x = Math.cos(angle) * 0.68 - 0.26;
        z = Math.sin(angle) * 0.58 + 0.08;
      } else {
        x = 1 - Math.cos(angle) * 0.68 + 0.1;
        z = -Math.sin(angle) * 0.58 - 0.12;
      }
      push(x + gaussian(rng) * sigma * 0.55, z + gaussian(rng) * sigma * 0.55, label);
    } else { // blobs
      const label = rng() < 0.5 ? 0 : 1;
      const center = label ? 0.46 : -0.46;
      push(center + gaussian(rng) * (0.16 + sigma * 0.35),
           center + gaussian(rng) * (0.20 + sigma * 0.35), label);
    }
  }

  // Shuffle deterministically so classes do not arrive grouped by label.
  const order = shuffledIndices(X.length, rng);
  return { X: order.map(i => X[i]), y: order.map(i => y[i]), kind, seed: Number(seed), noise: sigma };
}

export function activate(z, name) {
  if (name === "relu") return Math.max(0, z);
  if (name === "leaky-relu") return z > 0 ? z : 0.01 * z;
  if (name === "tanh") return Math.tanh(z);
  if (name === "sigmoid") return z >= 0 ? 1 / (1 + Math.exp(-Math.min(z, 60))) : Math.exp(Math.max(z, -60)) / (1 + Math.exp(Math.max(z, -60)));
  return z;
}

export function activationDerivative(z, a, name) {
  if (name === "relu") return z > 0 ? 1 : 0;
  if (name === "leaky-relu") return z > 0 ? 1 : 0.01;
  if (name === "tanh") return 1 - a * a;
  if (name === "sigmoid") return a * (1 - a);
  return 1;
}

function zerosLikeWeights(weights) {
  return weights.map(matrix => matrix.map(row => row.map(() => 0)));
}

function cloneWeights(weights) {
  return weights.map(matrix => matrix.map(row => row.slice()));
}

export class MLP {
  constructor({ hidden = [8, 8], activation = "tanh", optimizer = "adam", learningRate = 0.025, seed = 42 } = {}) {
    if (!Array.isArray(hidden) || hidden.length < 1 || hidden.length > 4 || hidden.some(v => !Number.isInteger(Number(v)) || Number(v) < 2 || Number(v) > 32)) {
      throw new Error('Use 1–4 hidden layers, each with 2–32 neurons.');
    }
    if (!['sigmoid', 'tanh', 'relu', 'leaky-relu'].includes(activation)) throw new Error(`Unsupported activation: ${activation}`);
    if (!['sgd', 'momentum', 'adam'].includes(optimizer)) throw new Error(`Unsupported optimizer: ${optimizer}`);
    if (!Number.isFinite(Number(learningRate)) || Number(learningRate) < 0.0001 || Number(learningRate) > 1) throw new Error('Learning rate must be between 0.0001 and 1.');
    this.inputSize = 2;
    this.outputSize = 1;
    this.hidden = hidden.map(v => Math.max(2, Math.min(32, Math.floor(v))));
    this.layerSizes = [2, ...this.hidden, 1];
    this.activation = activation;
    this.optimizer = optimizer;
    this.learningRate = Math.max(0.0001, Math.min(1, Number(learningRate)));
    this.seed = Number(seed) || 1;
    this.rng = makeRng(this.seed);
    this.weights = [];
    this.velocity = [];
    this.m = [];
    this.v = [];
    this.optimizerSteps = 0;
    this.lastGradients = [];
    this.lastCache = null;
    this.initialize();
  }

  initialize() {
    this.weights = [];
    for (let layer = 1; layer < this.layerSizes.length; layer++) {
      const fanIn = this.layerSizes[layer - 1];
      const fanOut = this.layerSizes[layer];
      const scale = this.activation === "relu" || this.activation === "leaky-relu"
        ? Math.sqrt(2 / fanIn) : Math.sqrt(2 / (fanIn + fanOut));
      const mat = Array.from({ length: fanOut }, () => Array.from({ length: fanIn + 1 }, () => gaussian(this.rng) * scale));
      this.weights.push(mat);
    }
    this.velocity = zerosLikeWeights(this.weights);
    this.m = zerosLikeWeights(this.weights);
    this.v = zerosLikeWeights(this.weights);
    this.lastGradients = zerosLikeWeights(this.weights);
    this.optimizerSteps = 0;
  }

  setOptimizer(optimizer, learningRate = this.learningRate) {
    this.optimizer = optimizer;
    this.learningRate = Math.max(0.0001, Math.min(1, Number(learningRate)));
    this.velocity = zerosLikeWeights(this.weights);
    this.m = zerosLikeWeights(this.weights);
    this.v = zerosLikeWeights(this.weights);
    this.optimizerSteps = 0;
  }

  forward(input) {
    const activations = [input.slice()];
    const zs = [];
    let a = input.slice();
    for (let layer = 0; layer < this.weights.length; layer++) {
      const matrix = this.weights[layer];
      const z = matrix.map(row => {
        let total = row[row.length - 1];
        for (let j = 0; j < a.length; j++) total += row[j] * a[j];
        return total;
      });
      const isOutput = layer === this.weights.length - 1;
      a = z.map(value => isOutput ? activate(value, "sigmoid") : activate(value, this.activation));
      zs.push(z);
      activations.push(a);
    }
    this.lastCache = { activations, zs };
    return a[0];
  }

  gradientsFor(input, target) {
    const prediction = this.forward(input);
    const { activations, zs } = this.lastCache;
    const deltas = Array(this.weights.length);
    const last = this.weights.length - 1;
    // BCE + sigmoid has the compact derivative (prediction - target).
    deltas[last] = [prediction - target];
    for (let layer = last - 1; layer >= 0; layer--) {
      deltas[layer] = this.weights[layer + 1][0]
        .slice(0, this.layerSizes[layer + 1])
        .map((_, unit) => {
          let upstream = 0;
          for (let nextUnit = 0; nextUnit < deltas[layer + 1].length; nextUnit++) {
            upstream += this.weights[layer + 1][nextUnit][unit] * deltas[layer + 1][nextUnit];
          }
          return upstream * activationDerivative(zs[layer][unit], activations[layer + 1][unit], this.activation);
        });
    }
    const gradients = this.weights.map((matrix, layer) => matrix.map((row, unit) => {
      const grad = activations[layer].map(value => deltas[layer][unit] * value);
      grad.push(deltas[layer][unit]); // bias gradient
      return grad;
    }));
    return { gradients, prediction, loss: binaryCrossEntropy(prediction, target) };
  }

  trainBatch(X, y, indices, { momentum = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 } = {}) {
    const grads = zerosLikeWeights(this.weights);
    let totalLoss = 0;
    let correct = 0;
    for (const index of indices) {
      const result = this.gradientsFor(X[index], y[index]);
      totalLoss += result.loss;
      correct += (result.prediction >= 0.5 ? 1 : 0) === y[index] ? 1 : 0;
      result.gradients.forEach((matrix, l) => matrix.forEach((row, i) => row.forEach((value, j) => { grads[l][i][j] += value; })));
    }
    const count = Math.max(1, indices.length);
    grads.forEach(matrix => matrix.forEach(row => row.forEach((_, j, arr) => { arr[j] /= count; })));
    this.lastGradients = cloneWeights(grads);
    this.optimizerSteps += 1;

    this.weights.forEach((matrix, l) => matrix.forEach((row, i) => row.forEach((weight, j) => {
      const g = grads[l][i][j];
      if (this.optimizer === "momentum") {
        this.velocity[l][i][j] = momentum * this.velocity[l][i][j] + g;
        this.weights[l][i][j] = weight - this.learningRate * this.velocity[l][i][j];
      } else if (this.optimizer === "adam") {
        this.m[l][i][j] = beta1 * this.m[l][i][j] + (1 - beta1) * g;
        this.v[l][i][j] = beta2 * this.v[l][i][j] + (1 - beta2) * g * g;
        const mHat = this.m[l][i][j] / (1 - Math.pow(beta1, this.optimizerSteps));
        const vHat = this.v[l][i][j] / (1 - Math.pow(beta2, this.optimizerSteps));
        this.weights[l][i][j] = weight - this.learningRate * mHat / (Math.sqrt(vHat) + epsilon);
      } else {
        this.weights[l][i][j] = weight - this.learningRate * g;
      }
    })));
    return { loss: totalLoss / count, accuracy: correct / count };
  }

  trainEpoch(X, y, { batchSize = 24, shuffle = true } = {}) {
    const rng = this.rng;
    const indices = shuffle ? shuffledIndices(X.length, rng) : Array.from({ length: X.length }, (_, i) => i);
    let loss = 0;
    for (let start = 0; start < indices.length; start += batchSize) {
      const batch = indices.slice(start, start + batchSize);
      loss += this.trainBatch(X, y, batch).loss * batch.length;
    }
    return evaluate(this, X, y);
  }

  toJSON() {
    return {
      format: "neural-playground-model",
      formatVersion: 1,
      layerSizes: this.layerSizes,
      activation: this.activation,
      optimizer: this.optimizer,
      learningRate: this.learningRate,
      seed: this.seed,
      weights: this.weights,
      note: "Educational binary classifier; output activation is sigmoid and loss is binary cross-entropy."
    };
  }

  static fromJSON(obj) {
    if (!obj || obj.format !== "neural-playground-model" || !Array.isArray(obj.layerSizes) || !Array.isArray(obj.weights)) {
      throw new Error("This file is not a supported Neural Playground model export.");
    }
    if (obj.layerSizes.length < 3 || obj.layerSizes.length > 6 || obj.layerSizes[0] !== 2 || obj.layerSizes.at(-1) !== 1) {
      throw new Error("Model architecture is not supported (expected 2 inputs and 1 output).");
    }
    const model = new MLP({ hidden: obj.layerSizes.slice(1, -1), activation: obj.activation, optimizer: obj.optimizer, learningRate: obj.learningRate, seed: obj.seed });
    if (obj.weights.length !== model.weights.length || obj.weights.some((matrix, l) => matrix.length !== model.weights[l].length || matrix.some((row, i) => row.length !== model.weights[l][i].length))) {
      throw new Error("Model weight dimensions do not match the declared architecture.");
    }
    model.weights = cloneWeights(obj.weights);
    model.lastGradients = zerosLikeWeights(model.weights);
    model.velocity = zerosLikeWeights(model.weights);
    model.m = zerosLikeWeights(model.weights);
    model.v = zerosLikeWeights(model.weights);
    model.optimizerSteps = 0;
    return model;
  }
}

export function binaryCrossEntropy(probability, target) {
  const p = Math.max(1e-7, Math.min(1 - 1e-7, probability));
  return -(target * Math.log(p) + (1 - target) * Math.log(1 - p));
}

export function evaluate(model, X, y, knownLoss = null) {
  let loss = 0;
  let correct = 0;
  for (let i = 0; i < X.length; i++) {
    const p = model.forward(X[i]);
    loss += binaryCrossEntropy(p, y[i]);
    if ((p >= 0.5 ? 1 : 0) === y[i]) correct++;
  }
  return { loss: knownLoss == null ? loss / Math.max(1, X.length) : knownLoss, accuracy: correct / Math.max(1, X.length) };
}

export function parameterCount(model) {
  return model.weights.reduce((total, layer) => total + layer.reduce((sum, row) => sum + row.length, 0), 0);
}

export function csvForDataset(dataset) {
  return ["x1,x2,label", ...dataset.X.map((p, i) => `${p[0].toFixed(6)},${p[1].toFixed(6)},${dataset.y[i]}`)].join("\n");
}
