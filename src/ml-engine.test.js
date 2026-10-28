import test from 'node:test';
import assert from 'node:assert/strict';
import { MLP, createDataset, evaluate, parameterCount, activate, activationDerivative, binaryCrossEntropy, csvForDataset } from './ml-engine.js';

test('synthetic datasets are deterministic and labels are binary', () => {
  for (const kind of ['xor', 'circles', 'spiral', 'moons', 'blobs']) {
    const a = createDataset(kind, 100, 123, 0.1);
    const b = createDataset(kind, 100, 123, 0.1);
    assert.equal(a.X.length, 100);
    assert.equal(a.y.length, 100);
    assert.deepEqual(a, b);
    assert.ok(a.y.every(v => v === 0 || v === 1));
    assert.ok(a.X.every(p => p.length === 2 && p.every(Number.isFinite)));
  }
});

test('MLP shapes, parameter count and predictions are valid', () => {
  const model = new MLP({ hidden: [6, 4], seed: 7 });
  assert.deepEqual(model.layerSizes, [2, 6, 4, 1]);
  assert.equal(parameterCount(model), (2 + 1) * 6 + (6 + 1) * 4 + (4 + 1));
  const p = model.forward([0.1, -0.3]);
  assert.ok(p >= 0 && p <= 1);
  assert.ok(Number.isFinite(p));
});

test('all supported activation functions and derivatives are finite', () => {
  for (const name of ['sigmoid', 'tanh', 'relu', 'leaky-relu']) {
    const a = activate(-0.5, name);
    const d = activationDerivative(-0.5, a, name);
    assert.ok(Number.isFinite(a));
    assert.ok(Number.isFinite(d));
  }
  assert.ok(Math.abs(binaryCrossEntropy(0.9, 1) - 0.1053605) < 1e-5);
});

test('one epoch updates parameters for every optimizer without non-finite values', () => {
  const data = createDataset('xor', 80, 42, 0.1);
  for (const optimizer of ['sgd', 'momentum', 'adam']) {
    const model = new MLP({ hidden: [8, 8], optimizer, seed: 12 });
    const before = JSON.stringify(model.weights);
    const metrics = model.trainEpoch(data.X, data.y, { batchSize: 20 });
    assert.notEqual(JSON.stringify(model.weights), before);
    assert.ok(Number.isFinite(metrics.loss));
    assert.ok(metrics.loss >= 0);
    assert.ok(metrics.accuracy >= 0 && metrics.accuracy <= 1);
    assert.ok(model.weights.flat(2).every(Number.isFinite));
    assert.ok(model.lastGradients.flat(2).every(Number.isFinite));
  }
});

test('training learns a simple XOR boundary better than its initial state', () => {
  const data = createDataset('xor', 240, 42, 0.08);
  const model = new MLP({ hidden: [8, 8], activation: 'tanh', optimizer: 'adam', learningRate: 0.025, seed: 42 });
  const before = evaluate(model, data.X, data.y);
  for (let i = 0; i < 100; i++) model.trainEpoch(data.X, data.y, { batchSize: 24 });
  const after = evaluate(model, data.X, data.y);
  assert.ok(after.loss < before.loss * 0.72, `Expected loss to improve: ${before.loss} -> ${after.loss}`);
  assert.ok(after.accuracy > before.accuracy);
});

test('model JSON export/import preserves predictions', () => {
  const data = createDataset('moons', 40, 17, 0.1);
  const model = new MLP({ hidden: [5, 3], seed: 17 });
  for (let i = 0; i < 3; i++) model.trainEpoch(data.X, data.y);
  const restored = MLP.fromJSON(JSON.parse(JSON.stringify(model.toJSON())));
  for (const point of data.X.slice(0, 10)) assert.ok(Math.abs(model.forward(point) - restored.forward(point)) < 1e-10);
  assert.throws(() => MLP.fromJSON({ format: 'other' }), /not a supported/);
});

test('dataset CSV has a stable header and one row per example', () => {
  const data = createDataset('blobs', 32, 9, 0.1);
  const csv = csvForDataset(data).split('\n');
  assert.equal(csv[0], 'x1,x2,label');
  assert.equal(csv.length, 33);
});


test('backpropagated gradient matches a finite-difference estimate', () => {
  const model = new MLP({ hidden: [3], activation: 'tanh', optimizer: 'sgd', seed: 3 });
  const x = [0.21, -0.34], target = 1, epsilon = 1e-5;
  const analytic = model.gradientsFor(x, target).gradients[0][0][0];
  const original = model.weights[0][0][0];
  model.weights[0][0][0] = original + epsilon;
  const high = binaryCrossEntropy(model.forward(x), target);
  model.weights[0][0][0] = original - epsilon;
  const low = binaryCrossEntropy(model.forward(x), target);
  model.weights[0][0][0] = original;
  const numeric = (high - low) / (2 * epsilon);
  assert.ok(Math.abs(analytic - numeric) < 1e-5, `analytic ${analytic}, numeric ${numeric}`);
});

test('invalid model configuration is rejected early', () => {
  assert.throws(() => new MLP({ hidden: [1, 4] }), /1–4 hidden layers/);
  assert.throws(() => new MLP({ hidden: [4], activation: 'mystery' }), /Unsupported activation/);
});
