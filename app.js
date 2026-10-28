import { MLP, DATASETS, createDataset, evaluate, parameterCount, csvForDataset } from './src/ml-engine.js';

const $ = (id) => document.getElementById(id);
const storageKey = 'neural-playground-session-v1';
function storageGet(key) { try { return window.localStorage.getItem(key); } catch { return null; } }
function storageSet(key, value) { try { window.localStorage.setItem(key, value); } catch { /* Private mode / restricted origins can disable persistence. */ } }
const concepts = [
  { title: 'What is a decision boundary?', body: 'A decision boundary is the line or surface a classifier uses to separate classes. A neural network creates it by composing simple mathematical transformations. Watch this map evolve as the loss decreases.', equation: 'ŷ = σ(W₂ · φ(W₁x + b₁) + b₂)' },
  { title: 'What does the loss measure?', body: 'Loss gives the optimiser a signal about how wrong a prediction is. Binary cross-entropy strongly penalises confident predictions that disagree with the label. Accuracy is easier to interpret, but it can hide how confident the model was.', equation: 'L = −[y log(p) + (1−y) log(1−p)]' },
  { title: 'What is backpropagation?', body: 'Backpropagation applies the chain rule from the output back through each layer. It estimates how a tiny change to every weight would affect the loss. The Gradients view shows the last minibatch’s weight gradients.', equation: '∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂w' },
  { title: 'Why do optimisers differ?', body: 'SGD uses the current gradient, Momentum accumulates a direction over time, and Adam adapts a step for each parameter using running averages. A faster start does not guarantee a better final solution on every dataset.', equation: 'SGD: w ← w − ηg' },
];
const state = {
  model: null, dataset: null, epoch: 0, history: [], targetEpochs: 120,
  isTraining: false, paused: false, runId: 0, viz: 'weights', selectedNode: { layer: 1, unit: 0 },
  comparison: [], conceptIndex: 0, probeIndex: 0, lastMetrics: { loss: null, accuracy: null },
};

function hiddenFromInput() {
  const raw = $('architectureInput').value.trim();
  if (!raw) throw new Error('Enter hidden-layer sizes, such as 8,8.');
  const parts = raw.split(/[ ,x]+/i).filter(Boolean).map(Number);
  if (parts.length < 1 || parts.length > 4 || parts.some(n => !Number.isInteger(n) || n < 2 || n > 32)) {
    throw new Error('Use 1–4 hidden layers, each with 2–32 neurons. Example: 8,8.');
  }
  return parts;
}
function architectureString(model = state.model) { return model.layerSizes.join('–'); }
function currentSeed() { return Number($('seedInput').value) || 42; }
function currentDatasetOptions() { return { kind: $('datasetSelect').value, count: Number($('sampleCount').value), seed: currentSeed(), noise: Number($('noiseSlider').value) }; }
function createModelFromControls(hidden = null) {
  const hiddenLayers = hidden ?? hiddenFromInput();
  return new MLP({ hidden: hiddenLayers, activation: $('activationSelect').value, optimizer: $('optimizerSelect').value, learningRate: Number($('learningRate').value), seed: currentSeed() });
}
function setStatus(text, mode = '') {
  $('runStatus').textContent = text;
  $('runStatus').className = `status-tag ${mode}`;
  $('miniStatus').classList.toggle('trained', state.epoch > 0);
}
function toast(message, kind = 'success') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  $('toastRegion').append(el);
  setTimeout(() => el.remove(), 3400);
}
function delay(ms = 0) { return new Promise(resolve => setTimeout(resolve, ms)); }
function esc(value) { return String(value).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c])); }
function fmt(value, digits = 4) { return Number.isFinite(value) ? value.toFixed(digits) : '—'; }
function downloadFile(name, content, type = 'text/plain;charset=utf-8') {
  const blob = new Blob([content], { type }); const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function saveSession() {
  try {
    localStorage.setItem(storageKey, JSON.stringify({
      dataset: state.dataset, model: state.model?.toJSON(), epoch: state.epoch, history: state.history,
      settings: { targetEpochs: state.targetEpochs, sampleCount: $('sampleCount').value, noise: $('noiseSlider').value, seed: $('seedInput').value, datasetKind: $('datasetSelect').value, architecture: $('architectureInput').value, activation: $('activationSelect').value, optimizer: $('optimizerSelect').value, learningRate: $('learningRate').value },
      comparison: state.comparison, savedAt: new Date().toISOString(),
    }));
  } catch (e) { /* Storage can be disabled or full; the in-memory session still works. */ }
}
function tryRestore() {
  try {
    const raw = localStorage.getItem(storageKey); if (!raw) return false;
    const saved = JSON.parse(raw);
    if (!saved?.model || !saved?.dataset?.X || !saved?.dataset?.y) return false;
    if (saved.settings) {
      $('sampleCount').value = saved.settings.sampleCount ?? 240;
      $('sampleCountOutput').value = $('sampleCount').value;
      $('noiseSlider').value = saved.settings.noise ?? 0.13; $('noiseOutput').value = Number($('noiseSlider').value).toFixed(2);
      $('seedInput').value = saved.settings.seed ?? 42;
      $('datasetSelect').value = saved.settings.datasetKind ?? 'xor';
      $('architectureInput').value = saved.settings.architecture ?? '8,8';
      $('activationSelect').value = saved.settings.activation ?? 'tanh';
      $('optimizerSelect').value = saved.settings.optimizer ?? 'adam';
      $('learningRate').value = saved.settings.learningRate ?? 0.025;
      $('learningRateOutput').value = Number($('learningRate').value).toFixed(3);
      state.targetEpochs = Number(saved.settings.targetEpochs ?? 120);
      $('epochTarget').value = state.targetEpochs; $('epochTargetOutput').value = `${state.targetEpochs} epochs`;
    }
    state.dataset = saved.dataset; state.model = MLP.fromJSON(saved.model);
    state.epoch = Number(saved.epoch) || 0; state.history = Array.isArray(saved.history) ? saved.history : [];
    state.comparison = Array.isArray(saved.comparison) ? saved.comparison : [];
    return true;
  } catch (e) { return false; }
}
function initialise() {
  if (!tryRestore()) resetExperiment({ initial: true, quiet: true });
  syncControlOutputs();
  renderAll(true);
  renderComparison();
  updateTopLabels();
}
function syncControlOutputs() {
  $('sampleCountOutput').value = $('sampleCount').value;
  $('noiseOutput').value = Number($('noiseSlider').value).toFixed(2);
  $('learningRateOutput').value = Number($('learningRate').value).toFixed(3);
  $('epochTarget').value = state.targetEpochs;
  $('epochTargetOutput').value = `${state.targetEpochs} epochs`;
  $('targetEpochs').textContent = `Target: ${state.targetEpochs}`;
  $('architectureMetric').textContent = state.model ? architectureString() : '2–8–8–1';
}
function resetExperiment({ initial = false, quiet = false, useCurrentDataset = false } = {}) {
  stopRun();
  try {
    state.model = createModelFromControls();
  } catch (error) {
    if (!quiet) toast(error.message, 'error');
    return;
  }
  if (!useCurrentDataset || !state.dataset) state.dataset = createDataset(...Object.values(currentDatasetOptions()));
  state.epoch = 0; state.history = []; state.probeIndex = 0;
  state.selectedNode = { layer: Math.min(1, state.model.layerSizes.length - 2), unit: 0 };
  state.lastMetrics = evaluate(state.model, state.dataset.X, state.dataset.y);
  setStatus('READY');
  $('progressText').textContent = 'Ready to learn'; $('progressPercent').textContent = '0%'; $('progressBar').style.width = '0%';
  if (!initial && !quiet) toast('Model reset. The current dataset is ready for a fresh run.');
  updateTopLabels(); renderAll(true); saveSession();
}
function updateTopLabels() {
  $('miniModel').textContent = state.model ? architectureString() : '2–8–8–1';
  const datasetMeta = DATASETS.find(d => d.id === state.dataset?.kind);
  $('miniDataset').textContent = `${datasetMeta?.name ?? 'Synthetic data'} · ${state.epoch ? `${state.epoch} epochs` : 'Not trained'}`;
  $('architectureMetric').textContent = architectureString();
  $('parameterMetric').textContent = parameterCount(state.model).toLocaleString();
  $('targetEpochs').textContent = `Target: ${state.targetEpochs}`;
  $('compareDatasetTitle').textContent = `${datasetMeta?.name ?? 'Synthetic'} classification`;
  $('compareDatasetCopy').textContent = datasetMeta?.description ?? 'Synthetic binary classification.';
  $('compareArchitecture').textContent = `${architectureString()} network`;
  $('compareSamples').textContent = `${state.dataset.X.length} examples`;
}
function renderAll(includeBoundary = true) {
  if (!state.model || !state.dataset) return;
  renderMetrics();
  renderNetwork();
  renderLossChart();
  if (includeBoundary) renderBoundary();
  updateTopLabels();
}
function renderMetrics() {
  const metrics = evaluate(state.model, state.dataset.X, state.dataset.y);
  state.lastMetrics = metrics;
  $('lossMetric').textContent = fmt(metrics.loss, 4);
  $('accuracyMetric').innerHTML = `${(metrics.accuracy * 100).toFixed(1)}<small>%</small>`;
  $('epochMetric').textContent = state.epoch.toLocaleString();
  const last = state.history.at(-1);
  const prev = state.history.at(-2);
  $('lossTrend').textContent = !last ? 'Baseline' : prev && last.loss < prev.loss ? '↓ Improving' : prev && last.loss > prev.loss + 0.0001 ? '↑ Increasing' : 'Stabilising';
  $('lossTrend').className = `metric-chip ${!last ? '' : prev && last.loss <= prev.loss ? 'good' : 'bad'}`;
  $('accuracyTrend').textContent = !last ? 'Baseline' : `${Math.round(metrics.accuracy * 100)}% now`;
  $('accuracyTrend').className = `metric-chip ${metrics.accuracy >= .8 ? 'good' : ''}`;
  $('miniDataset').textContent = `${DATASETS.find(d => d.id === state.dataset.kind)?.name ?? state.dataset.kind} · ${state.epoch ? `${state.epoch} epochs` : 'Not trained'}`;
  $('progressPercent').textContent = `${Math.min(100, Math.round(state.epoch / state.targetEpochs * 100))}%`;
  $('progressBar').style.width = `${Math.min(100, state.epoch / state.targetEpochs * 100)}%`;
  $('progressText').textContent = state.isTraining ? (state.paused ? 'Training paused' : 'Optimising parameters…') : state.epoch >= state.targetEpochs ? 'Target completed' : state.epoch ? 'Training checkpoint saved' : 'Ready to learn';
  const minLoss = state.history.length ? Math.min(...state.history.map(item => item.loss)) : null;
  $('bestLossText').textContent = `Best loss: ${minLoss == null ? '—' : fmt(minLoss, 4)}`;
  if (state.history.length > 1) {
    const delta = state.history.at(-1).loss - state.history[0].loss;
    $('lossDeltaText').textContent = `${delta <= 0 ? '↓' : '↑'} ${Math.abs(delta).toFixed(4)} since start`;
  } else $('lossDeltaText').textContent = state.history.length ? 'First epoch recorded' : 'No training steps yet';
}
function renderBoundary() {
  const canvas = $('boundaryCanvas'); if (!canvas || !state.model || !state.dataset) return;
  const rect = canvas.getBoundingClientRect(); if (rect.width < 20 || rect.height < 20) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(rect.width), h = Math.round(rect.height);
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const pad = { left: 24, right: 14, top: 12, bottom: 22 };
  const pw = w - pad.left - pad.right; const ph = h - pad.top - pad.bottom;
  const xmin = -1.3, xmax = 1.3, ymin = -1.3, ymax = 1.3;
  const sx = x => pad.left + (x - xmin) / (xmax - xmin) * pw;
  const sy = y => pad.top + (ymax - y) / (ymax - ymin) * ph;
  ctx.fillStyle = '#0c1422'; ctx.fillRect(0, 0, w, h);
  const params = parameterCount(state.model);
  const grid = params > 1600 ? 8 : params > 500 ? 6 : 5;
  for (let px = 0; px < pw; px += grid) {
    for (let py = 0; py < ph; py += grid) {
      const x = xmin + (px / pw) * (xmax - xmin);
      const y = ymax - (py / ph) * (ymax - ymin);
      const p = state.model.forward([x, y]);
      const confidence = Math.min(1, Math.abs(p - .5) * 2);
      const a = .04 + confidence * .17;
      const c0 = [100, 145, 246], c1 = [244, 164, 108];
      const color = c0.map((v, i) => Math.round(v * (1 - p) + c1[i] * p));
      ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${a})`;
      ctx.fillRect(pad.left + px, pad.top + py, grid + .4, grid + .4);
    }
  }
  ctx.strokeStyle = 'rgba(142,164,203,.1)'; ctx.lineWidth = 1;
  for (let tick = -1; tick <= 1.001; tick += .5) {
    ctx.beginPath(); ctx.moveTo(sx(tick), pad.top); ctx.lineTo(sx(tick), pad.top + ph); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, sy(tick)); ctx.lineTo(pad.left + pw, sy(tick)); ctx.stroke();
  }
  ctx.strokeStyle = 'rgba(164,184,217,.26)'; ctx.setLineDash([3, 4]);
  ctx.beginPath(); ctx.moveTo(sx(0), pad.top); ctx.lineTo(sx(0), pad.top + ph); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.left, sy(0)); ctx.lineTo(pad.left + pw, sy(0)); ctx.stroke(); ctx.setLineDash([]);
  const radius = state.dataset.X.length > 400 ? 2.15 : state.dataset.X.length > 280 ? 2.7 : 3.2;
  state.dataset.X.forEach((point, i) => {
    const x = sx(point[0]), y = sy(point[1]);
    ctx.beginPath(); ctx.arc(x, y, radius + .8, 0, Math.PI * 2); ctx.fillStyle = 'rgba(5,10,19,.74)'; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fillStyle = state.dataset.y[i] ? '#f5ad76' : '#80adff'; ctx.fill();
    ctx.strokeStyle = state.dataset.y[i] ? 'rgba(255,226,201,.65)' : 'rgba(219,233,255,.65)'; ctx.lineWidth = .6; ctx.stroke();
  });
  ctx.fillStyle = '#8292ac'; ctx.font = '9px "DM Mono",monospace';
  ctx.fillText('−1', sx(-1) - 5, pad.top + ph + 14); ctx.fillText('0', sx(0) - 2, pad.top + ph + 14); ctx.fillText('+1', sx(1) - 5, pad.top + ph + 14);
  $('plotOverlay').classList.add('hidden');
}
function rgbaFromSign(value, max, positive = '#7cf0c5', negative = '#ff8fa7') {
  const magnitude = max > 0 ? Math.min(1, Math.abs(value) / max) : 0;
  const color = value >= 0 ? positive : negative;
  return { color, opacity: .16 + magnitude * .76, width: .65 + magnitude * 2.8, magnitude };
}
function renderNetwork() {
  if (!state.model) return;
  const svg = $('networkSvg'); const W = 760, H = 310;
  const nodes = [];
  const layerCount = state.model.layerSizes.length;
  const left = 55, right = W - 42, top = 34, bottom = H - 32;
  state.model.forward(state.dataset.X[state.probeIndex % state.dataset.X.length]);
  const cache = state.model.lastCache;
  const allWeight = state.model.weights.flat(2).map(Math.abs); const maxWeight = Math.max(1e-9, ...allWeight);
  const allGrad = state.model.lastGradients.flat(2).map(Math.abs); const maxGrad = Math.max(1e-9, ...allGrad);
  state.model.layerSizes.forEach((size, layer) => {
    const x = left + (right - left) * layer / (layerCount - 1);
    const count = size; const usable = bottom - top; const gap = Math.min(usable / Math.max(1, count - 1), 31);
    const full = gap * (count - 1); const y0 = (H - full) / 2;
    for (let unit = 0; unit < size; unit++) nodes.push({ x, y: y0 + unit * gap, layer, unit, activation: cache.activations[layer][unit] ?? 0 });
  });
  const byKey = (layer, unit) => nodes.find(n => n.layer === layer && n.unit === unit);
  const edgeParts = [];
  state.model.weights.forEach((matrix, layerIndex) => {
    matrix.forEach((row, unit) => {
      row.slice(0, state.model.layerSizes[layerIndex]).forEach((weight, prevUnit) => {
        const a = byKey(layerIndex, prevUnit), b = byKey(layerIndex + 1, unit);
        const grad = state.model.lastGradients[layerIndex]?.[unit]?.[prevUnit] ?? 0;
        let value = state.viz === 'gradients' ? grad : weight;
        let style = state.viz === 'activations' ? { color: '#71819a', opacity: .24, width: .8 } : rgbaFromSign(value, state.viz === 'gradients' ? maxGrad : maxWeight);
        edgeParts.push(`<line class="network-edge" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${style.color}" stroke-opacity="${style.opacity}" stroke-width="${style.width}" data-value="${value}" data-weight="${weight}" data-gradient="${grad}"><title>${`Layer ${layerIndex} → ${layerIndex + 1} · ${weight >= 0 ? '+' : ''}${weight.toFixed(5)} weight · ${grad >= 0 ? '+' : ''}${grad.toFixed(5)} gradient`}</title></line>`);
      });
    });
  });
  const nodeParts = nodes.map(n => {
    const isSelected = state.selectedNode.layer === n.layer && state.selectedNode.unit === n.unit;
    let fill = n.layer === 0 ? '#344b70' : n.layer === layerCount - 1 ? '#1f705c' : '#243448';
    let stroke = n.layer === 0 ? '#80adff' : n.layer === layerCount - 1 ? '#7cf0c5' : '#64809b';
    if (state.viz === 'activations') {
      const value = n.activation;
      const intensity = Math.max(0, Math.min(1, (value + 1) / 2));
      fill = n.layer === layerCount - 1 ? (value >= .5 ? '#ad744d' : '#3e5f91') : `rgba(${Math.round(65 + intensity * 65)},${Math.round(76 + intensity * 102)},${Math.round(106 + intensity * 68)},.95)`;
    }
    const r = n.layer === 0 || n.layer === layerCount - 1 ? 15 : 11.5;
    const label = n.layer === 0 ? (n.unit === 0 ? 'x₁' : 'x₂') : n.layer === layerCount - 1 ? 'ŷ' : String(n.unit + 1);
    const desc = n.layer === 0 ? `Input feature ${n.unit + 1}` : n.layer === layerCount - 1 ? 'Output probability' : `Hidden layer ${n.layer}, neuron ${n.unit + 1}`;
    return `<g class="network-selectable" data-layer="${n.layer}" data-unit="${n.unit}" tabindex="0" role="button" aria-label="Inspect ${desc}"><circle class="network-node${isSelected ? ' selected' : ''}" cx="${n.x}" cy="${n.y}" r="${r}" fill="${fill}" stroke="${isSelected ? '#fff' : stroke}" stroke-width="${isSelected ? 2.4 : 1.15}"/><text class="network-node-label" x="${n.x}" y="${n.y + 3}" text-anchor="middle">${label}</text><title>${desc} · activation ${Number(n.activation).toFixed(4)}</title></g>`;
  }).join('');
  const layerLabels = state.model.layerSizes.map((size, layer) => {
    const x = left + (right - left) * layer / (layerCount - 1);
    const label = layer === 0 ? 'INPUTS' : layer === layerCount - 1 ? 'OUTPUT' : `HIDDEN ${layer}`;
    return `<text x="${x}" y="17" text-anchor="middle" fill="#6f8099" style="font:9px 'DM Mono',monospace;letter-spacing:.9px">${label}</text><text x="${x}" y="${H - 7}" text-anchor="middle" fill="#9aabc1" style="font:9px 'DM Mono',monospace">${size} ${size === 1 ? 'unit' : 'units'}</text>`;
  }).join('');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = `<defs><filter id="soft-glow"><feGaussianBlur stdDeviation="2.5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>${layerLabels}${edgeParts.join('')}${nodeParts}`;
  svg.querySelectorAll('.network-selectable').forEach(el => {
    const select = () => { state.selectedNode = { layer: Number(el.dataset.layer), unit: Number(el.dataset.unit) }; renderNetwork(); renderInspector(); };
    el.addEventListener('click', select);
    el.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } });
  });
  $('networkEmpty').classList.add('hidden');
  $('networkModeText').textContent = state.viz === 'weights' ? 'Weight values' : state.viz === 'gradients' ? 'Last minibatch gradients' : 'Neuron activations';
  $('networkSummary')?.classList.remove('hidden');
  renderInspector();
}
function renderInspector() {
  if (!state.model || !state.dataset) return;
  const { layer, unit } = state.selectedNode;
  const cache = state.model.lastCache ?? (state.model.forward(state.dataset.X[0]), state.model.lastCache);
  if (!cache || layer < 0 || layer >= state.model.layerSizes.length || unit >= state.model.layerSizes[layer]) return;
  const activationValue = cache.activations[layer][unit];
  let weights = [], gradients = [], bias = null, role = '';
  if (layer === 0) {
    role = `INPUT FEATURE ${unit + 1}`;
    const first = state.model.weights[0];
    weights = first.map((row, i) => ({ label: `H1 · ${i + 1}`, value: row[unit], gradient: state.model.lastGradients[0]?.[i]?.[unit] ?? 0 }));
    bias = activationValue;
  } else {
    role = layer === state.model.layerSizes.length - 1 ? 'OUTPUT NEURON' : `HIDDEN LAYER ${layer}`;
    const matrix = state.model.weights[layer - 1];
    const row = matrix[unit];
    const grad = state.model.lastGradients[layer - 1][unit];
    weights = row.slice(0, -1).map((value, i) => ({ label: `Prev · ${i + 1}`, value, gradient: grad[i] }));
    bias = row.at(-1);
    weights.push({ label: 'Bias', value: bias, gradient: grad.at(-1) });
  }
  const max = Math.max(1e-9, ...weights.map(w => Math.abs(state.viz === 'gradients' ? w.gradient : w.value)));
  const list = weights.map(w => {
    const displayed = state.viz === 'gradients' ? w.gradient : w.value;
    const width = Math.max(2, Math.abs(displayed) / max * 100);
    const color = displayed >= 0 ? 'var(--accent-strong)' : 'var(--rose)';
    return `<div class="weight-row"><span>${esc(w.label)}</span><div class="weight-bar"><span style="width:${width}%;background:${color}"></span></div><output>${displayed >= 0 ? '+' : ''}${fmt(displayed, 4)}</output></div>`;
  }).join('');
  $('inspectorContent').innerHTML = `<div class="inspector-title-row"><div class="inspect-node-glyph">${layer === 0 ? `x${unit + 1}` : layer === state.model.layerSizes.length - 1 ? 'ŷ' : `h${layer}.${unit + 1}`}</div><div><strong>${layer === 0 ? `Input ${unit + 1}` : layer === state.model.layerSizes.length - 1 ? 'Output probability' : `Hidden ${layer} · Neuron ${unit + 1}`} </strong><small>${role}</small></div></div><div class="inspector-stat-grid"><div class="inspector-stat"><span>${layer === state.model.layerSizes.length - 1 ? 'PREDICTION' : 'ACTIVATION'}</span><strong>${fmt(activationValue, 5)}</strong></div><div class="inspector-stat"><span>${layer === 0 ? 'FEATURE VALUE' : 'BIAS'}</span><strong>${fmt(bias, 5)}</strong></div></div><div class="weight-list-head"><span>${state.viz === 'gradients' ? 'LAST GRADIENTS' : 'INCOMING WEIGHTS'}</span><span>${weights.length} VALUES</span></div><div class="weight-list">${list}</div><div class="field-hint" style="margin-top:10px">Probe example ${state.probeIndex + 1} · label ${state.dataset.y[state.probeIndex]} · output p=${fmt(state.model.forward(state.dataset.X[state.probeIndex]), 4)}</div>`;
}
function renderLossChart() {
  const canvas = $('lossCanvas'); const rect = canvas.getBoundingClientRect();
  if (rect.width < 20 || rect.height < 20) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2), w = rect.width, h = rect.height;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#111827'; ctx.fillRect(0, 0, w, h);
  const hist = state.history;
  if (!hist.length) { $('lossEmpty').classList.remove('hidden'); return; }
  $('lossEmpty').classList.add('hidden');
  const pad = { l: 35, r: 13, t: 12, b: 25 }, pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const maxLoss = Math.max(.5, ...hist.map(h => h.loss)) * 1.08;
  ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(137,157,193,.13)'; ctx.fillStyle = '#71819a'; ctx.font = '8px "DM Mono",monospace';
  for (let i = 0; i <= 4; i++) { const y = pad.t + ph * i / 4; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.fillText((maxLoss * (1 - i / 4)).toFixed(2), 2, y + 3); }
  const pointX = i => pad.l + (hist.length <= 1 ? pw : (i / (Math.max(state.targetEpochs - 1, hist.length - 1))) * pw);
  const pointY = value => pad.t + ph * (1 - Math.max(0, Math.min(maxLoss, value)) / maxLoss);
  const gradient = ctx.createLinearGradient(0, pad.t, 0, pad.t + ph); gradient.addColorStop(0, 'rgba(124,240,197,.22)'); gradient.addColorStop(1, 'rgba(124,240,197,0)');
  ctx.beginPath(); hist.forEach((point, i) => i ? ctx.lineTo(pointX(i), pointY(point.loss)) : ctx.moveTo(pointX(i), pointY(point.loss))); ctx.lineTo(pointX(hist.length - 1), pad.t + ph); ctx.lineTo(pointX(0), pad.t + ph); ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();
  ctx.beginPath(); hist.forEach((point, i) => i ? ctx.lineTo(pointX(i), pointY(point.loss)) : ctx.moveTo(pointX(i), pointY(point.loss))); ctx.strokeStyle = '#7cf0c5'; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();
  const last = hist.at(-1); ctx.beginPath(); ctx.arc(pointX(hist.length - 1), pointY(last.loss), 3.5, 0, Math.PI * 2); ctx.fillStyle = '#d8ffed'; ctx.fill();
  ctx.fillStyle = '#71819a'; ctx.fillText('1', pad.l, h - 6); ctx.fillText(String(hist.length), w - pad.r - 18, h - 6);
}
function stopRun() { state.runId++; state.isTraining = false; state.paused = false; $('pauseButton').disabled = true; $('pauseButton').textContent = 'Ⅱ'; $('trainButtonLabel').textContent = 'Train model'; }
async function startTraining() {
  if (state.isTraining) { if (state.paused) { state.paused = false; $('pauseButton').textContent = 'Ⅱ'; setStatus('TRAINING', 'training'); } return; }
  if (state.epoch >= state.targetEpochs) { toast('Target already completed. Increase the target or reset the model for a fresh run.', 'error'); return; }
  state.isTraining = true; state.paused = false; const runId = ++state.runId;
  $('pauseButton').disabled = false; $('trainButtonLabel').textContent = 'Training…'; setStatus('TRAINING', 'training');
  const target = state.targetEpochs;
  while (state.isTraining && runId === state.runId && state.epoch < target) {
    if (state.paused) { await delay(80); continue; }
    const metrics = state.model.trainEpoch(state.dataset.X, state.dataset.y, { batchSize: 24 });
    state.epoch++;
    state.history.push({ epoch: state.epoch, loss: metrics.loss, accuracy: metrics.accuracy });
    if (state.history.length > 500) state.history.shift();
    renderMetrics(); renderLossChart();
    const params = parameterCount(state.model);
    const networkInterval = params > 1600 ? 5 : params > 500 ? 3 : 1;
    const boundaryInterval = params > 1600 ? 8 : params > 500 ? 4 : 2;
    if (state.epoch % networkInterval === 0 || state.epoch === 1 || state.epoch === target) renderNetwork();
    if (state.epoch % boundaryInterval === 0 || state.epoch === 1 || state.epoch === target) renderBoundary();
    if (state.epoch % 5 === 0 || state.epoch === target) saveSession();
    await delay(7);
  }
  if (runId === state.runId) {
    state.isTraining = false; state.paused = false; $('pauseButton').disabled = true; $('trainButtonLabel').textContent = 'Train model';
    setStatus(state.epoch >= target ? 'COMPLETE' : 'PAUSED', state.epoch >= target ? 'done' : '');
    renderAll(true); saveSession();
    if (state.epoch >= target) toast(`Training complete · loss ${fmt(state.lastMetrics.loss)} · accuracy ${(state.lastMetrics.accuracy * 100).toFixed(1)}%`);
  }
}
function pauseTraining() {
  if (!state.isTraining) return;
  state.paused = !state.paused; $('pauseButton').textContent = state.paused ? '▶' : 'Ⅱ';
  setStatus(state.paused ? 'PAUSED' : 'TRAINING', state.paused ? '' : 'training');
  $('progressText').textContent = state.paused ? 'Training paused' : 'Optimising parameters…';
}
function stepEpoch() {
  if (state.isTraining) { toast('Pause or finish the current run before stepping manually.', 'error'); return; }
  const metrics = state.model.trainEpoch(state.dataset.X, state.dataset.y, { batchSize: 24 });
  state.epoch++; state.history.push({ epoch: state.epoch, loss: metrics.loss, accuracy: metrics.accuracy });
  renderAll(true); setStatus('STEPPED', 'done'); saveSession();
}
function downloadCurrentModel() { downloadFile('neural-playground-model.json', JSON.stringify(state.model.toJSON(), null, 2), 'application/json'); toast('Model JSON exported.'); }
function exportCurrentDataset() { downloadFile(`neural-playground-${state.dataset.kind}-dataset.csv`, csvForDataset(state.dataset), 'text/csv;charset=utf-8'); toast('Dataset CSV exported.'); }
function importModelFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const model = MLP.fromJSON(JSON.parse(reader.result));
      state.model = model; $('architectureInput').value = model.hidden.join(','); $('activationSelect').value = model.activation; $('optimizerSelect').value = model.optimizer; $('learningRate').value = model.learningRate;
      state.epoch = 0; state.history = []; state.selectedNode = { layer: 1, unit: 0 }; stopRun();
      renderAll(true); saveSession(); syncControlOutputs(); toast('Model imported. Training history is not included in model files.');
    } catch (error) { toast(error.message || 'Could not read this model file.', 'error'); }
  };
  reader.readAsText(file);
}
function renderComparison() {
  const cards = $('comparisonCards');
  if (!state.comparison.length) {
    $('comparisonTable').innerHTML = '<tr><td colspan="5" class="table-empty">No experiment results yet.</td></tr>';
    $('exportComparison').disabled = true; $('comparisonEmpty').classList.remove('hidden'); drawComparisonChart([]); return;
  }
  const items = state.comparison;
  const configs = [
    { title: 'Adam + Tanh', desc: 'Adaptive step sizes with smooth hidden activations.', tag: 'BASELINE' },
    { title: 'SGD + Tanh', desc: 'A simple gradient step with no adaptive moments.', tag: 'CLASSIC' },
    { title: 'Adam + ReLU', desc: 'Adaptive optimisation with a piecewise-linear activation.', tag: 'SPARSE' },
    { title: 'Momentum + Tanh', desc: 'Gradient descent with a moving average of updates.', tag: 'INERTIA' },
  ];
  cards.innerHTML = configs.map((cfg, i) => {
    const result = items[i];
    return `<article class="comparison-card"><div class="comparison-top"><span class="comparison-index">${String(i + 1).padStart(2,'0')}</span><span class="small-pill">${cfg.tag}</span></div><h3>${cfg.title}</h3><p>${cfg.desc}</p>${result ? `<div class="comparison-result"><div class="result-accuracy">${(result.accuracy * 100).toFixed(1)}<small>%</small></div><div class="result-label">TRAIN ACCURACY</div><div class="result-meta"><span>Loss ${fmt(result.loss, 4)}</span><span>${Math.round(result.ms)} ms</span></div></div>` : '<div class="comparison-placeholder">Results appear here after a run.</div>'}</article>`;
  }).join('');
  $('comparisonTable').innerHTML = items.map(item => `<tr><td>${esc(item.name)}</td><td>${fmt(item.loss, 4)}</td><td>${(item.accuracy * 100).toFixed(1)}%</td><td>${item.params.toLocaleString()}</td><td>${Math.round(item.ms)} ms</td></tr>`).join('');
  $('exportComparison').disabled = false; $('comparisonEmpty').classList.add('hidden'); drawComparisonChart(items);
}
function drawComparisonChart(items) {
  const canvas = $('comparisonCanvas'), rect = canvas.getBoundingClientRect(); if (rect.width < 20 || rect.height < 20) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2), w = rect.width, h = rect.height; canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h); ctx.fillStyle = '#101726'; ctx.fillRect(0,0,w,h);
  if (!items.length) return;
  const pad={l:42,r:18,t:17,b:37}, pw=w-pad.l-pad.r, ph=h-pad.t-pad.b; ctx.strokeStyle='rgba(137,157,193,.14)';ctx.fillStyle='#7888a0';ctx.font='9px "DM Mono",monospace';
  for(let i=0;i<=4;i++){const y=pad.t+ph*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.fillText((1-i/4).toFixed(1),8,y+3);}
  const gap=13, bw=Math.min(55,(pw-gap*(items.length-1))/items.length), total=bw*items.length+gap*(items.length-1), start=pad.l+(pw-total)/2;
  const cols=['#7cf0c5','#8cb8ff','#b7a0ff','#f5bd79'];
  items.forEach((item,i)=>{const x=start+i*(bw+gap);const bh=ph*Math.max(0,Math.min(1,item.accuracy));ctx.fillStyle=cols[i];ctx.globalAlpha=.78;ctx.fillRect(x,pad.t+ph-bh,bw,bh);ctx.globalAlpha=1;ctx.fillStyle='#e0e8f4';ctx.textAlign='center';ctx.font='9px "DM Mono",monospace';ctx.fillText(`${(item.accuracy*100).toFixed(0)}%`,x+bw/2,pad.t+ph-bh-6);ctx.fillStyle='#8493ab';ctx.fillText(['Adam','SGD','ReLU','Momentum'][i],x+bw/2,h-17);});ctx.textAlign='left';
}
async function runComparison() {
  if (state.isTraining) { toast('Finish or pause main training before running comparisons.', 'error'); return; }
  const button=$('runComparison'); button.disabled=true; button.innerHTML='<span class="play-symbol">◌</span> Running…'; $('comparisonStatus').textContent='Training four independent models…';
  const configs=[{name:'Adam + Tanh',activation:'tanh',optimizer:'adam',lr:.025},{name:'SGD + Tanh',activation:'tanh',optimizer:'sgd',lr:.06},{name:'Adam + ReLU',activation:'relu',optimizer:'adam',lr:.025},{name:'Momentum + Tanh',activation:'tanh',optimizer:'momentum',lr:.035}];
  const results=[]; const epochs=Math.min(100,Math.max(40,Math.round(state.targetEpochs*.6)));
  for(let i=0;i<configs.length;i++){
    const cfg=configs[i], started=performance.now();
    const model=new MLP({hidden:state.model.hidden,activation:cfg.activation,optimizer:cfg.optimizer,learningRate:cfg.lr,seed:currentSeed()+i*137+100});
    for(let ep=0;ep<epochs;ep++){ model.trainEpoch(state.dataset.X,state.dataset.y,{batchSize:24}); if(ep%10===0){$('comparisonStatus').textContent=`Training ${i+1}/4 · ${cfg.name} · epoch ${ep+1}/${epochs}`;await delay(0);} }
    const metrics=evaluate(model,state.dataset.X,state.dataset.y);results.push({name:cfg.name,loss:metrics.loss,accuracy:metrics.accuracy,params:parameterCount(model),ms:performance.now()-started,epochs,activation:cfg.activation,optimizer:cfg.optimizer,learningRate:cfg.lr,seed:currentSeed()+i*137+100});
    state.comparison=results.slice();renderComparison();await delay(0);
  }
  state.comparison=results; $('comparisonStatus').textContent=`Complete · ${epochs} epochs per model`; button.disabled=false;button.innerHTML='<span class="play-symbol">▶</span> Run comparison';renderComparison();saveSession();
  const winner=[...results].sort((a,b)=>a.loss-b.loss)[0];toast(`Comparison complete. Lowest final loss: ${winner.name} (${fmt(winner.loss,4)}).`);
}
function exportComparisonCSV() {
  if (!state.comparison.length) return;
  const rows=[['Configuration','Loss','Accuracy','Parameters','Milliseconds','Epochs','Activation','Optimizer','Learning rate','Seed'],...state.comparison.map(r=>[r.name,r.loss,r.accuracy,r.params,r.ms,r.epochs,r.activation,r.optimizer,r.learningRate,r.seed])];
  downloadFile('neural-playground-comparison.csv',rows.map(row=>row.map(v=>`"${String(v).replaceAll('"','""')}"`).join(',')).join('\n'),'text/csv;charset=utf-8');
}
function setPage(name) {
  const valid=['playground','compare','learn','guide']; if(!valid.includes(name))name='playground';
  document.querySelectorAll('.page').forEach(page=>{const active=page.id===`page-${name}`;page.hidden=!active;page.classList.toggle('active',active);});
  document.querySelectorAll('.nav-item').forEach(item=>item.classList.toggle('active',item.dataset.page===name));
  $('breadcrumbPage').textContent={playground:'Playground',compare:'Experiment lab',learn:'Learn the basics',guide:'Model guide'}[name];
  $('sidebar').classList.remove('open');
  if(name==='compare'){updateTopLabels();renderComparison();drawComparisonChart(state.comparison);}
  if(name==='playground')requestAnimationFrame(()=>renderAll(true));
  window.location.hash=name;
}
function updateConcept() {
  const c=concepts[state.conceptIndex];$('conceptTitle').textContent=c.title;$('conceptBody').textContent=c.body;$('conceptEquation').textContent=c.equation;$('conceptCount').textContent=`${String(state.conceptIndex+1).padStart(2,'0')} / ${String(concepts.length).padStart(2,'0')}`;
  $('conceptDots').innerHTML=concepts.map((_,i)=>`<i class="${i===state.conceptIndex?'active':''}"></i>`).join('');$('prevConcept').disabled=state.conceptIndex===0;$('nextConcept').disabled=state.conceptIndex===concepts.length-1;
}
function attachEvents() {
  document.querySelectorAll('.nav-item').forEach(button=>button.addEventListener('click',()=>setPage(button.dataset.page)));
  document.querySelectorAll('[data-go-playground]').forEach(button=>button.addEventListener('click',()=>setPage('playground')));
  $('menuToggle').addEventListener('click',()=>$('sidebar').classList.toggle('open'));
  $('themeToggle').addEventListener('click',()=>{document.body.classList.toggle('light');storageSet('neural-playground-theme',document.body.classList.contains('light')?'light':'dark');requestAnimationFrame(()=>renderAll(true));});
  if(storageGet('neural-playground-theme')==='light')document.body.classList.add('light');
  $('datasetSelect').addEventListener('change',()=>{const info=DATASETS.find(d=>d.id===$('datasetSelect').value);$('datasetDescription').textContent=info?.description??'';resetExperiment({useCurrentDataset:false});});
  $('sampleCount').addEventListener('input',()=>{$('sampleCountOutput').value=$('sampleCount').value;});
  $('noiseSlider').addEventListener('input',()=>{$('noiseOutput').value=Number($('noiseSlider').value).toFixed(2);});
  $('learningRate').addEventListener('input',()=>{$('learningRateOutput').value=Number($('learningRate').value).toFixed(3);});
  $('epochTarget').addEventListener('input',()=>{$('epochTargetOutput').value=`${$('epochTarget').value} epochs`;state.targetEpochs=Number($('epochTarget').value);$('targetEpochs').textContent=`Target: ${state.targetEpochs}`;saveSession();});
  $('epochTarget').addEventListener('change',()=>{if(state.epoch<state.targetEpochs){$('progressPercent').textContent=`${Math.round(state.epoch/state.targetEpochs*100)}%`;}});
  $('applyArchitecture').addEventListener('click',()=>{try{hiddenFromInput();resetExperiment({useCurrentDataset:true});}catch(e){toast(e.message,'error');}});
  $('architectureInput').addEventListener('keydown',e=>{if(e.key==='Enter')$('applyArchitecture').click();});
  ['activationSelect','optimizerSelect','learningRate'].forEach(id=>$(id).addEventListener('change',()=>resetExperiment({useCurrentDataset:true})));
  $('resampleButton').addEventListener('click',()=>{$('seedInput').value=String(currentSeed()+1);resetExperiment({useCurrentDataset:false});$('datasetDescription').textContent=DATASETS.find(d=>d.id===$('datasetSelect').value)?.description??'';});
  $('resetSidebar').addEventListener('click',()=>resetExperiment({useCurrentDataset:true}));
  $('trainButton').addEventListener('click',startTraining);$('startTrainingTop').addEventListener('click',()=>{setPage('playground');startTraining();});$('pauseButton').addEventListener('click',pauseTraining);$('stepButton').addEventListener('click',stepEpoch);
  $('exportModel').addEventListener('click',downloadCurrentModel);$('guideExportModel').addEventListener('click',downloadCurrentModel);$('guideExportDataset').addEventListener('click',exportCurrentDataset);
  $('refreshBoundary').addEventListener('click',()=>{renderBoundary();toast('Decision boundary refreshed.');});
  document.querySelectorAll('[data-viz]').forEach(btn=>btn.addEventListener('click',()=>{state.viz=btn.dataset.viz;document.querySelectorAll('[data-viz]').forEach(b=>b.classList.toggle('active',b===btn));renderNetwork();}));
  $('prevConcept').addEventListener('click',()=>{state.conceptIndex=Math.max(0,state.conceptIndex-1);updateConcept();});$('nextConcept').addEventListener('click',()=>{state.conceptIndex=Math.min(concepts.length-1,state.conceptIndex+1);updateConcept();});
  $('runComparison').addEventListener('click',runComparison);$('clearComparisons').addEventListener('click',()=>{state.comparison=[];renderComparison();$('comparisonStatus').textContent='Cleared';saveSession();});$('exportComparison').addEventListener('click',exportComparisonCSV);
  $('openHelp').addEventListener('click',()=>$('helpModal').hidden=false);
  $('closeHelp').addEventListener('click',()=>$('helpModal').hidden=true);$('gotIt').addEventListener('click',()=>{if($('dontShowHelp').checked)storageSet('neural-playground-suppress-help','1');$('helpModal').hidden=true;});
  const file=$('modelFileInput');file?.addEventListener('change',e=>{if(e.target.files?.[0])importModelFile(e.target.files[0]);file.value='';});
  document.addEventListener('keydown',e=>{
    if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName))return;
    if(e.key==='1')setPage('playground');if(e.key==='2')setPage('compare');if(e.key==='3')setPage('learn');
    if(e.key===' '&&document.querySelector('#page-playground').classList.contains('active')){e.preventDefault();if(state.isTraining)pauseTraining();else startTraining();}
    if(e.key==='?'&&document.querySelector('#page-playground').classList.contains('active'))$('helpModal').hidden=false;
    if(e.key==='Escape'){$('sidebar').classList.remove('open');$('helpModal').hidden=true;}
  });
  window.addEventListener('resize',()=>{clearTimeout(window._npResize);window._npResize=setTimeout(()=>{renderAll(true);drawComparisonChart(state.comparison);},120);});
  window.addEventListener('hashchange',()=>{const name=window.location.hash.slice(1);if(name)setPage(name);});
}

// File picker is intentionally hidden: the interface has a compact Import button in the overflow menu.
const importButton = document.createElement('button');
importButton.className='btn btn-secondary'; importButton.id='importModel'; importButton.textContent='↥ Import';
importButton.addEventListener('click',()=> $('modelFileInput').click());
const topActions=document.querySelector('.intro-actions'); if(topActions) topActions.insertBefore(importButton, $('startTrainingTop'));
const fileInput=document.createElement('input');fileInput.type='file';fileInput.id='modelFileInput';fileInput.accept='.json,application/json';fileInput.hidden=true;document.body.append(fileInput);

attachEvents();
initialise();
updateConcept();
if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1')) navigator.serviceWorker.register('./sw.js').catch(() => {});
const initialDatasetMeta=DATASETS.find(d=>d.id===state.dataset.kind);$('datasetDescription').textContent=initialDatasetMeta?.description??'';
if(window.location.hash&&['playground','compare','learn','guide'].includes(window.location.hash.slice(1)))setPage(window.location.hash.slice(1));
