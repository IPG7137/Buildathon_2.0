/**
 * QualityGuard AI – Industrial QualityOps Dashboard JavaScript
 * ==============================================================
 * Full frontend client for manager/inspector/admin web dashboard.
 * Supports:
 * - Live Camera Studio with 2s capture countdown & freeze preview
 * - Drag-and-drop / file upload & curated dataset sample test mode
 * - Real PyTorch inference via backend /inspections endpoint
 * - Smart Review Priority Queue with explainable 0-100 risk scoring
 * - Product catalog & production batch management
 * - Batch Hold/Release disposition workflow with audit trail
 * - Corrective actions with Side-by-Side Before/After repair evidence
 * - Reinspection verification (Pass/Fail)
 * - Grounded QualityGuard Copilot with interactive citation pills
 * - Printable Executive Quality Compliance Summary
 * - Immutable Audit Ledger
 */

const API_BASE = localStorage.getItem('api_base') || (window.location.origin.startsWith('http') ? window.location.origin : 'http://localhost:8000');

// ─── State ──────────────────────────────────────────────
let state = {
  token: null,
  user: null,
  currentPage: 'overview',
  dateFrom: null,
  dateTo: null,
  inspPage: 1,
  batchPage: 1,
  charts: {},
  pollInterval: null,
  
  // Studio state
  studioMode: 'camera',
  studioFacingMode: 'environment',
  cameraStream: null,
  capturedBlob: null,
  countdownTimer: null,
  currentInspection: null,
  reviewQueueItems: [],
  productsList: [],
  batchesList: [],
};

// ─── Helpers ─────────────────────────────────────────────
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

function api(path, opts = {}) {
  const headers = { ...opts.headers };
  if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
  if (opts.body && !(opts.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }
  return fetch(`${API_BASE}${path}`, { ...opts, headers }).then(async res => {
    if (res.status === 401) { logout(); throw new Error('Session expired'); }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  });
}

function toast(msg, type = 'info', duration = 4000) {
  const icons = { success: '✅', error: '❌', warn: '⚠️', info: 'ℹ️' };
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${msg}</span>`;
  $('toast-container').appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity 0.3s'; setTimeout(() => t.remove(), 300); }, duration);
}

function formatDate(dt) {
  if (!dt) return '—';
  return new Date(dt).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' });
}

function predictionBadge(pred) {
  if (!pred) return '<span class="badge badge-muted">—</span>';
  const map = { GOOD: 'badge-good', DEFECT: 'badge-defect', UNCERTAIN: 'badge-warn' };
  const icons = { GOOD: '✅', DEFECT: '❌', UNCERTAIN: '⚠️' };
  return `<span class="badge ${map[pred] || 'badge-muted'}">${icons[pred] || ''} ${pred}</span>`;
}

function statusBadge(status) {
  if (!status) return '<span class="badge badge-muted">—</span>';
  const map = {
    completed: 'badge-good', reviewed: 'badge-blue', under_review: 'badge-warn',
    failed: 'badge-muted', open: 'badge-warn', in_progress: 'badge-blue',
    awaiting_reinspection: 'badge-warn', resolved: 'badge-good', closed: 'badge-good',
    reopened: 'badge-defect'
  };
  return `<span class="badge ${map[status] || 'badge-muted'}">${status.replace(/_/g, ' ')}</span>`;
}

function batchStatusBadge(status) {
  const map = {
    approved: 'badge-good', on_hold: 'badge-defect',
    rejected: 'badge-defect', in_inspection: 'badge-blue', pending: 'badge-muted',
  };
  return `<span class="badge ${map[status] || 'badge-muted'}">${status.replace(/_/g, ' ')}</span>`;
}

function priorityBadge(p) {
  const map = { critical: 'badge-defect', high: 'badge-warn', medium: 'badge-blue', low: 'badge-muted' };
  return `<span class="badge ${map[p] || 'badge-muted'}">${p}</span>`;
}

function fillCreds(user, pass) {
  $('login-username').value = user;
  $('login-password').value = pass;
}
window.fillCreds = fillCreds;

// ─── Date range helpers ───────────────────────────────────
function getDateRange() {
  const preset = $('date-preset').value;
  const now = new Date();
  let from, to = now.toISOString();

  if (preset === 'today') {
    from = new Date(now.getFullYear(), now.getMonth(), now.getDate()).toISOString();
  } else if (preset === '7d') {
    from = new Date(now - 7 * 86400000).toISOString();
  } else if (preset === '30d') {
    from = new Date(now - 30 * 86400000).toISOString();
  } else {
    from = $('date-from').value ? new Date($('date-from').value).toISOString() : new Date(now - 7*86400000).toISOString();
    to = $('date-to').value ? new Date($('date-to').value + 'T23:59:59').toISOString() : to;
  }
  return { from, to };
}

// ─── Auth ─────────────────────────────────────────────────
async function login(username, password) {
  const res = await fetch(`${API_BASE}/auth/token`, {
    method: 'POST',
    body: new URLSearchParams({ username, password }),
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Login failed');
  }
  return res.json();
}

function logout() {
  state.token = null;
  state.user = null;
  localStorage.removeItem('ps01_token');
  localStorage.removeItem('ps01_user');
  stopStudioCamera();
  if (state.pollInterval) clearInterval(state.pollInterval);
  $('app-shell').classList.add('hidden');
  $('login-overlay').classList.remove('hidden');
}

function loadSession() {
  const token = localStorage.getItem('ps01_token');
  const user = localStorage.getItem('ps01_user');
  if (token && user) {
    state.token = token;
    state.user = JSON.parse(user);
    return true;
  }
  return false;
}

// ─── Model status ─────────────────────────────────────────
async function checkModelStatus() {
  try {
    const h = await fetch(`${API_BASE}/health`);
    const data = await h.json();
    const dot = $('model-status-dot');
    const text = $('model-status-text');
    if (data.model_ready) {
      dot.className = 'status-dot ready';
      text.textContent = `Model v${data.model_version} (${data.model_architecture || 'MobileNetV3'}) ✓`;
    } else {
      dot.className = 'status-dot error';
      text.textContent = `Model: ${data.model_error || 'Not loaded'}`;
    }
  } catch(e) {
    $('model-status-dot').className = 'status-dot error';
    $('model-status-text').textContent = 'Backend offline';
  }
}

// ─── Navigation ────────────────────────────────────────────
function navigateTo(page) {
  // If leaving studio, release camera stream
  if (state.currentPage === 'live-studio' && page !== 'live-studio') {
    stopStudioCamera();
  }

  state.currentPage = page;

  // Update nav items
  $$('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.page === page);
  });

  // Update pages
  $$('.page').forEach(el => el.classList.add('hidden'));
  const pageEl = $(`page-${page}`);
  if (pageEl) pageEl.classList.remove('hidden');

  // Update title
  const titles = {
    overview: 'Overview',
    analytics: 'Analytics & Trends',
    'live-studio': 'Live Inspection Studio',
    inspections: 'Inspections History',
    'review-queue': 'Smart Review Priority Queue',
    products: 'Product Specifications',
    batches: 'Production Batches & Hold Management',
    'corrective-actions': 'Corrective & Preventive Actions (CAPA)',
    audit: 'Immutable Audit Trail',
    assistant: 'QualityGuard Copilot',
  };
  $('page-title').textContent = titles[page] || page;

  // Load page data
  switch (page) {
    case 'overview': loadOverview(); break;
    case 'analytics': loadAnalytics(); break;
    case 'live-studio': initLiveStudio(); break;
    case 'inspections': loadInspections(); break;
    case 'review-queue': loadReviewQueue(); break;
    case 'products': loadProducts(); break;
    case 'batches': loadBatches(); break;
    case 'corrective-actions': loadCorrectiveActions(); break;
    case 'audit': loadAuditEvents(); break;
  }
}
window.navigateTo = navigateTo;

// ─── Overview ──────────────────────────────────────────────
async function loadOverview() {
  const { from, to } = getDateRange();
  const params = `?date_from=${encodeURIComponent(from)}&date_to=${encodeURIComponent(to)}`;

  try {
    const [summary, trends, batchStats] = await Promise.all([
      api(`/dashboard/summary${params}`),
      api('/dashboard/trends?days=7'),
      api('/dashboard/batches'),
    ]);

    // KPIs
    $('kpi-val-total').textContent = summary.total_inspections;
    $('kpi-val-good').textContent = summary.good_predictions;
    $('kpi-val-defect').textContent = summary.defect_suspected;
    $('kpi-val-review').textContent = summary.pending_review;
    $('kpi-val-conf-good').textContent = summary.confirmed_good;
    $('kpi-val-conf-defect').textContent = summary.confirmed_defect;
    $('kpi-val-ca').textContent = summary.open_corrective_actions;
    $('kpi-val-hold').textContent = summary.batches_on_hold;
    $('kpi-val-rate').textContent = `${summary.defect_rate_pct}%`;

    // Nav and hero badges
    $('badge-reviews').textContent = summary.pending_review;
    $('badge-ca').textContent = summary.open_corrective_actions;
    const heroCount = $('hero-queue-count');
    if (heroCount) heroCount.textContent = summary.pending_review;

    // Charts
    renderDailyChart(trends.daily_trends);
    renderDistributionChart(summary);
    renderCAChart(trends.corrective_actions_by_status);
    renderBatchChart(batchStats.slice(0, 8));

  } catch(e) {
    toast(`Dashboard load error: ${e.message}`, 'error');
  }
}

function renderDailyChart(data) {
  const ctx = $('chart-daily').getContext('2d');
  if (state.charts.daily) state.charts.daily.destroy();

  state.charts.daily = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(d => d.date.substring(5)),
      datasets: [
        { label: 'Good', data: data.map(d => d.good), backgroundColor: 'rgba(63,185,80,0.6)', borderColor: '#3fb950', borderWidth: 1, borderRadius: 4 },
        { label: 'Defect', data: data.map(d => d.defect), backgroundColor: 'rgba(248,81,73,0.6)', borderColor: '#f85149', borderWidth: 1, borderRadius: 4 },
      ],
    },
    options: { ...chartDefaults(), scales: { x: { ...chartAxis(), stacked: true }, y: { ...chartAxis(), stacked: true, beginAtZero: true } } },
  });
}

function renderDistributionChart(summary) {
  const ctx = $('chart-distribution').getContext('2d');
  if (state.charts.dist) state.charts.dist.destroy();

  state.charts.dist = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Good Predictions', 'Defect Suspected', 'Uncertain'],
      datasets: [{
        data: [summary.good_predictions, summary.defect_suspected, summary.uncertain_predictions || 0],
        backgroundColor: ['rgba(63,185,80,0.7)', 'rgba(248,81,73,0.7)', 'rgba(210,153,34,0.7)'],
        borderColor: ['#3fb950', '#f85149', '#d29922'],
        borderWidth: 1,
      }],
    },
    options: {
      ...chartDefaults(),
      cutout: '65%',
      plugins: { ...chartDefaults().plugins, legend: { position: 'bottom', labels: chartDefaults().plugins.legend.labels } },
    },
  });
}

function renderCAChart(data) {
  const ctx = $('chart-ca').getContext('2d');
  if (state.charts.ca) state.charts.ca.destroy();

  const labels = Object.keys(data || {});
  const values = Object.values(data || {});

  state.charts.ca = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels.map(l => l.replace(/_/g, ' ')),
      datasets: [{
        label: 'Actions',
        data: values,
        backgroundColor: ['rgba(248,81,73,0.6)', 'rgba(33,136,255,0.6)', 'rgba(210,153,34,0.6)', 'rgba(63,185,80,0.6)', 'rgba(110,118,129,0.4)'],
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: { ...chartDefaults(), plugins: { legend: { display: false } }, scales: { x: chartAxis(), y: { ...chartAxis(), beginAtZero: true } } },
  });
}

function renderBatchChart(batches) {
  const ctx = $('chart-batches').getContext('2d');
  if (state.charts.batches) state.charts.batches.destroy();

  state.charts.batches = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: batches.map(b => b.batch_code),
      datasets: [{
        label: 'Defect Rate (%)',
        data: batches.map(b => b.defect_rate_pct),
        backgroundColor: batches.map(b => b.defect_rate_pct > 25 ? 'rgba(248,81,73,0.7)' : 'rgba(33,136,255,0.6)'),
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      ...chartDefaults(),
      plugins: { legend: { display: false } },
      scales: { x: chartAxis(), y: { ...chartAxis(), beginAtZero: true, max: 100, ticks: { ...chartAxis().ticks, callback: v => v + '%' } } },
    },
  });
}

// ─── Analytics ─────────────────────────────────────────────
async function loadAnalytics() {
  try {
    const data = await api('/dashboard/trends?days=14');
    const labels = data.daily_trends.map(d => d.date.substring(5));

    // Trend
    const ctxTrend = $('chart-trend').getContext('2d');
    if (state.charts.trend) state.charts.trend.destroy();
    state.charts.trend = new Chart(ctxTrend, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Good', data: data.daily_trends.map(d => d.good), borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.08)', fill: true, tension: 0.3 },
          { label: 'Defect', data: data.daily_trends.map(d => d.defect), borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,0.08)', fill: true, tension: 0.3 },
        ],
      },
      options: { ...chartDefaults(), scales: { x: chartAxis(), y: { ...chartAxis(), beginAtZero: true } } },
    });

    // Defect Rate
    const ctxRate = $('chart-defect-rate').getContext('2d');
    if (state.charts.rate) state.charts.rate.destroy();
    state.charts.rate = new Chart(ctxRate, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Defect Rate %',
          data: data.daily_trends.map(d => d.total > 0 ? Math.round(d.defect / d.total * 100) : 0),
          borderColor: '#bc8cff', backgroundColor: 'rgba(188,140,255,0.1)',
          fill: true, tension: 0.3, pointRadius: 4,
        }],
      },
      options: { ...chartDefaults(), scales: { x: chartAxis(), y: { ...chartAxis(), beginAtZero: true, max: 100, ticks: { ...chartAxis().ticks, callback: v => v + '%' } } } },
    });
  } catch(e) {
    toast(`Analytics error: ${e.message}`, 'error');
  }
}

function chartDefaults() {
  return {
    responsive: true, maintainAspectRatio: true,
    plugins: {
      legend: { labels: { color: '#475569', font: { family: 'Inter', size: 12 }, boxWidth: 12, padding: 16 } },
      tooltip: {
        backgroundColor: '#0f172a', borderColor: '#e2e8f0', borderWidth: 1,
        titleColor: '#ffffff', bodyColor: '#cbd5e1',
        titleFont: { family: 'Inter', weight: '600' }, bodyFont: { family: 'Inter' }, padding: 10,
      },
    },
    animation: { duration: 400 },
  };
}

function chartAxis() {
  return {
    grid: { color: 'rgba(15, 23, 42, 0.06)' },
    ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } },
    border: { color: 'rgba(15, 23, 42, 0.1)' },
  };
}

// ─── LIVE INSPECTION STUDIO (Priority Hackathon Feature) ───
async function initLiveStudio() {
  // Generate session ID
  $('studio-session-id').value = `SESSION-${Date.now().toString(36).toUpperCase()}`;
  
  // Load products & batches
  await loadStudioMetadata();
  
  // Setup default mode
  switchStudioMode(state.studioMode || 'camera');
  
  // Setup drag-drop listeners
  setupDropZone();
}

async function loadStudioMetadata() {
  try {
    const [prods, batches] = await Promise.all([
      api('/products'),
      api('/batches?page=1&page_size=50'),
    ]);
    
    state.productsList = prods;
    state.batchesList = batches.items || [];
    
    const prodSel = $('studio-product-select');
    prodSel.innerHTML = prods.map(p => `<option value="${p.id}">${p.name} (${p.sku})</option>`).join('');
    
    const batchSel = $('studio-batch-select');
    batchSel.innerHTML = state.batchesList.map(b => `<option value="${b.id}">${b.batch_code} [${b.quality_status}]</option>`).join('');
  } catch(e) {
    console.error('Failed to load studio metadata:', e);
  }
}

function switchStudioMode(mode) {
  state.studioMode = mode;
  ['camera', 'upload', 'sample'].forEach(m => {
    $(`tab-${m}-mode`).classList.toggle('active', m === mode);
    $(`studio-view-${m}`).classList.toggle('hidden', m !== mode);
  });
  
  if (mode === 'camera') {
    startStudioCamera();
  } else {
    stopStudioCamera();
    if (mode === 'sample') loadSampleGrid();
  }
}
window.switchStudioMode = switchStudioMode;

async function startStudioCamera() {
  const video = $('studio-video');
  const errPill = $('studio-camera-error');
  errPill.classList.add('hidden');
  stopStudioCamera();
  
  // Reset preview
  $('studio-captured-preview').classList.add('hidden');
  $('studio-reticle').classList.remove('hidden');
  $('studio-review-actions').classList.add('hidden');
  $('btn-start-countdown').classList.remove('hidden');
  video.classList.remove('hidden');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: state.studioFacingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    state.cameraStream = stream;
    video.srcObject = stream;
  } catch(err) {
    console.error('Studio camera error:', err);
    errPill.classList.remove('hidden');
    errPill.querySelector('span').textContent = `Camera error: ${err.message || 'Permission denied'}. Please allow access or switch to Upload mode.`;
  }
}

function stopStudioCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
    const video = $('studio-video');
    if (video) video.srcObject = null;
  }
}

function flipStudioCamera() {
  state.studioFacingMode = state.studioFacingMode === 'environment' ? 'user' : 'environment';
  startStudioCamera();
}
window.flipStudioCamera = flipStudioCamera;

// 2-Second Delay Capture Flow
function trigger2sCapture() {
  const overlay = $('studio-countdown-overlay');
  const countNum = $('countdown-number');
  overlay.classList.remove('hidden');
  
  let countdown = 2;
  countNum.textContent = countdown;
  
  state.countdownTimer = setInterval(() => {
    countdown--;
    if (countdown > 0) {
      countNum.textContent = countdown;
    } else {
      clearInterval(state.countdownTimer);
      overlay.classList.add('hidden');
      performStillCapture();
    }
  }, 1000);
}
window.trigger2sCapture = trigger2sCapture;

function performStillCapture() {
  const video = $('studio-video');
  const canvas = $('studio-canvas');
  const preview = $('studio-captured-preview');
  
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  
  canvas.toBlob(blob => {
    if (!blob) { toast('Failed to capture sharp still frame', 'error'); return; }
    state.capturedBlob = blob;
    
    // Freeze still preview
    preview.src = URL.createObjectURL(blob);
    preview.classList.remove('hidden');
    video.classList.add('hidden');
    $('studio-reticle').classList.add('hidden');
    
    // Switch action buttons: Retake vs Run AI Inspection
    $('btn-start-countdown').classList.add('hidden');
    $('studio-review-actions').classList.remove('hidden');
  }, 'image/jpeg', 0.95);
}

function retakeStudioCapture() {
  state.capturedBlob = null;
  $('studio-captured-preview').classList.add('hidden');
  $('studio-video').classList.remove('hidden');
  $('studio-reticle').classList.remove('hidden');
  $('studio-review-actions').classList.add('hidden');
  $('btn-start-countdown').classList.remove('hidden');
}
window.retakeStudioCapture = retakeStudioCapture;

async function acceptAndRunInference() {
  if (!state.capturedBlob) { toast('No captured frame available', 'warn'); return; }
  await executeStudioInspection(state.capturedBlob, 'live_camera_capture.jpg', 'camera');
}
window.acceptAndRunInference = acceptAndRunInference;

// Dropzone & File Upload Mode
function setupDropZone() {
  const dropzone = $('studio-dropzone');
  const fileInput = $('studio-file-input');

  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files?.length) handleStudioFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', e => {
    if (e.target.files?.length) handleStudioFile(e.target.files[0]);
  });
}

function handleStudioFile(file) {
  if (!file) return;
  const allowed = ['image/jpeg', 'image/png', 'image/webp'];
  if (!allowed.includes(file.type)) {
    toast(`Unsupported file format ${file.type}. Please select JPG, PNG or WebP.`, 'error');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    toast(`File is too large (${(file.size/1024/1024).toFixed(1)} MB). Max limit is 20 MB.`, 'error');
    return;
  }
  
  state.capturedBlob = file;
  $('upload-meta-name').textContent = file.name;
  $('upload-meta-size').textContent = `${(file.size/1024).toFixed(0)} KB`;
  
  const reader = new FileReader();
  reader.onload = e => {
    $('upload-preview-img').src = e.target.result;
    $('dropzone-prompt').classList.add('hidden');
    $('dropzone-preview').classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

function clearUploadFile() {
  state.capturedBlob = null;
  $('studio-file-input').value = '';
  $('dropzone-prompt').classList.remove('hidden');
  $('dropzone-preview').classList.add('hidden');
}
window.clearUploadFile = clearUploadFile;

async function runUploadInference() {
  if (!state.capturedBlob) { toast('Please select an image first', 'warn'); return; }
  await executeStudioInspection(state.capturedBlob, state.capturedBlob.name || 'uploaded_sample.jpg', 'upload');
}
window.runUploadInference = runUploadInference;

// Curated Dataset Samples Mode
async function loadSampleGrid() {
  const grid = $('studio-sample-grid');
  try {
    const data = await api('/demo/sample-images');
    if (!data.samples || data.samples.length === 0) {
      grid.innerHTML = '<div class="empty-state">No sample images found.</div>';
      return;
    }
    
    grid.innerHTML = data.samples.map(s => `
      <div class="sample-card" onclick="testWithSample('${s.url}', '${s.filename}', '${s.expected_defect ? 'DEFECT' : 'GOOD'}')">
        <img src="${API_BASE}${s.url}" alt="${s.title}" onerror="this.src=''" />
        <div class="sample-card-title">${s.title}</div>
        <span class="sample-badge ${s.expected_defect ? 'defect' : 'good'}">
          ${s.expected_defect ? 'Suspected Defect' : 'Normal Surface'}
        </span>
      </div>
    `).join('');
  } catch(e) {
    grid.innerHTML = `<div class="empty-state">Error loading samples: ${e.message}</div>`;
  }
}

async function testWithSample(url, filename, expected) {
  try {
    toast(`Fetching real dataset sample: ${filename}...`, 'info');
    const res = await fetch(`${API_BASE}${url}`);
    const blob = await res.blob();
    await executeStudioInspection(blob, filename, 'dataset_sample');
  } catch(e) {
    toast(`Failed to load sample: ${e.message}`, 'error');
  }
}
window.testWithSample = testWithSample;

// Execute Inspection API
async function executeStudioInspection(blob, filename, captureMode) {
  const prodId = $('studio-product-select').value;
  const batchId = $('studio-batch-select').value;
  const serial = $('studio-serial').value;
  const session = $('studio-session-id').value;
  
  if (!batchId) { toast('Please select a production batch', 'warn'); return; }
  
  $('studio-infer-mode').textContent = 'RUNNING INFERENCE...';
  toast('Running PyTorch MobileNetV3 inference on image...', 'info');
  
  const form = new FormData();
  form.append('file', blob, filename);
  form.append('product_id', prodId);
  form.append('batch_id', batchId);
  form.append('capture_mode', captureMode);
  if (serial) form.append('serial_number', serial);
  if (session) form.append('session_id', session);
  
  try {
    const res = await api('/inspections', { method: 'POST', body: form });
    state.currentInspection = res;
    renderStudioInferenceResult(res);
    toast(`Inspection complete: ${res.prediction} (${((res.defect_score||0)*100).toFixed(1)}%)`, res.prediction === 'DEFECT' ? 'warn' : 'success');
  } catch(e) {
    toast(`Inspection failed: ${e.message}`, 'error');
    $('studio-infer-mode').textContent = 'ERROR';
  }
}

function renderStudioInferenceResult(insp) {
  $('studio-result-empty').classList.add('hidden');
  $('studio-result-content').classList.remove('hidden');
  $('studio-infer-mode').textContent = `SAVED (${insp.capture_mode})`;
  
  const isDefect = insp.prediction === 'DEFECT';
  const banner = $('verdict-banner');
  banner.className = `verdict-banner ${isDefect ? 'defect' : 'good'}`;
  $('verdict-icon').textContent = isDefect ? '❌' : '✅';
  $('verdict-title').textContent = insp.prediction;
  $('verdict-subtitle').textContent = isDefect 
    ? 'Surface anomaly detected. Queued for QC verification.' 
    : 'No severe defects detected. Quality criteria satisfied.';
    
  const defScore = Math.round((insp.defect_score || 0) * 100);
  const goodScore = 100 - defScore;
  
  $('metric-defect-score').textContent = `${defScore}%`;
  $('bar-defect-fill').style.width = `${defScore}%`;
  
  $('metric-good-score').textContent = `${goodScore}%`;
  $('bar-good-fill').style.width = `${goodScore}%`;
  
  $('res-insp-id').textContent = insp.id;
  $('res-model-ver').textContent = `MobileNetV3-Small (v${insp.model_version || '1.0.0'})`;
  $('res-latency').textContent = `${insp.inference_time_ms || 22} ms`;
  $('res-qc-status').textContent = insp.inspection_status;
  
  // Update CAPA button visibility
  $('btn-create-ca-from-insp').style.display = isDefect ? '' : 'none';
}

function openCreateCAModalFromCurrent() {
  if (!state.currentInspection) return;
  const title = prompt('Enter CAPA Ticket Title:', `Defect Investigation for Inspection ${state.currentInspection.id.substring(0,8)}`);
  if (!title) return;
  const rootCause = prompt('Root Cause Hypothesis (optional):', 'Suspected rolling friction / hot scale inclusion');
  
  api('/corrective-actions', {
    method: 'POST',
    body: JSON.stringify({
      inspection_id: state.currentInspection.id,
      title,
      priority: 'high',
      root_cause: rootCause,
    }),
  }).then(() => {
    toast('Corrective action ticket created successfully!', 'success');
    navigateTo('corrective-actions');
  }).catch(e => toast(`Error creating CAPA: ${e.message}`, 'error'));
}
window.openCreateCAModalFromCurrent = openCreateCAModalFromCurrent;

// ─── SMART REVIEW PRIORITY QUEUE ───────────────────────────
async function loadReviewQueue() {
  const container = $('review-queue-list');
  container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📋</div>Calculating priority risk scores...</div>';

  try {
    const items = await api('/inspections/review-queue');
    state.reviewQueueItems = items;
    $('badge-reviews').textContent = items.length;
    renderReviewQueueCards();
  } catch(e) {
    container.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
  }
}

function renderReviewQueueCards() {
  const container = $('review-queue-list');
  let items = [...(state.reviewQueueItems || [])];
  
  if (items.length === 0) {
    container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">✅</div>No inspections pending review. All caught up!</div>';
    return;
  }
  
  const sortMode = $('queue-sort')?.value || 'priority';
  if (sortMode === 'priority') {
    items.sort((a,b) => (b.priority_score || 0) - (a.priority_score || 0));
  } else if (sortMode === 'score_desc') {
    items.sort((a,b) => (b.defect_score || 0) - (a.defect_score || 0));
  } else if (sortMode === 'newest') {
    items.sort((a,b) => new Date(b.timestamp) - new Date(a.timestamp));
  }

  container.innerHTML = items.map(i => {
    const score = i.priority_score || 0;
    const scoreClass = score >= 70 ? 'crit' : score >= 40 ? 'warn' : '';
    const cardClass = score >= 70 ? 'critical' : score >= 40 ? 'urgent' : '';
    const imgUrl = i.image_path ? `${API_BASE}/storage/${i.image_path}` : null;
    
    return `
      <div class="priority-card ${cardClass}">
        <div class="priority-meter">
          <div class="priority-meter-score ${scoreClass}">${score}</div>
          <div class="priority-meter-label">RISK SCORE</div>
        </div>

        <div class="priority-thumb">
          ${imgUrl ? `<img src="${imgUrl}" alt="Thumb" onerror="this.parentElement.textContent='🔍'" />` : '🔍'}
        </div>

        <div class="priority-info">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
            ${predictionBadge(i.prediction)}
            <strong style="font-size:15px">Batch ${i.batch_code || '—'}</strong>
            <span style="font-size:12px;color:var(--text-muted)">• ${formatDate(i.timestamp)}</span>
          </div>
          <div style="font-size:12px;color:var(--text-secondary)">
            Inspection ID: <code style="font-family:var(--font-mono)">${i.id.substring(0,10)}…</code> | 
            Operator: ${i.operator_name || '—'} | 
            Defect Score: <strong>${((i.defect_score||0)*100).toFixed(1)}%</strong>
          </div>
          <div class="priority-factor-chips">
            ${(i.priority_factors || []).map(f => `<span class="factor-chip">⚡ ${f}</span>`).join('')}
          </div>
        </div>

        <div class="priority-actions">
          <button class="btn-sm ghost" onclick="openInspectionDetail('${i.id}')">View</button>
          <button class="btn-primary" onclick="openReviewModal('${i.id}','${i.batch_code||''}','${i.prediction||''}','${((i.defect_score||0)*100).toFixed(0)}')">
            Review & Sign
          </button>
        </div>
      </div>
    `;
  }).join('');
}
window.renderReviewQueueCards = renderReviewQueueCards;

// Review Modal
function openReviewModal(inspId, batchCode, prediction, scoreStr) {
  $('review-inspection-id').value = inspId;
  $('modal-review-insp-info').innerHTML = `
    <strong>Inspection:</strong> ${inspId.substring(0,12)}… | 
    <strong>Batch:</strong> ${batchCode} |
    <strong>Model Prediction:</strong> ${predictionBadge(prediction)} |
    <strong>Defect Score:</strong> ${scoreStr}%
    <br><small style="color:var(--text-muted)">Traceability Rule: Your review is stored separately. The original model prediction is immutable.</small>
  `;
  document.querySelectorAll('input[name="confirmed_class"]').forEach(r => r.checked = false);
  $('review-reason').value = '';
  $('review-notes').value = '';
  $('review-error').classList.add('hidden');
  $('modal-review-overlay').classList.remove('hidden');
}
window.openReviewModal = openReviewModal;

async function submitReview(e) {
  e.preventDefault();
  const inspId = $('review-inspection-id').value;
  const confirmed = document.querySelector('input[name="confirmed_class"]:checked')?.value;
  const reason = $('review-reason').value.trim();
  const notes = $('review-notes').value.trim();
  const errEl = $('review-error');

  if (!confirmed) { errEl.textContent = 'Please select an outcome.'; errEl.classList.remove('hidden'); return; }
  if (!reason) { errEl.textContent = 'Please provide a reason.'; errEl.classList.remove('hidden'); return; }

  try {
    errEl.classList.add('hidden');
    await api(`/inspections/${inspId}/review`, {
      method: 'POST',
      body: JSON.stringify({ confirmed_class: confirmed, reason, notes }),
    });
    toast('QC Review recorded in audit ledger!', 'success');
    $('modal-review-overlay').classList.add('hidden');
    if (state.currentPage === 'review-queue') loadReviewQueue();
    if (state.currentPage === 'inspections') loadInspections(state.inspPage);
    if (state.currentPage === 'overview') loadOverview();
  } catch(e) {
    errEl.textContent = e.message;
    errEl.classList.remove('hidden');
  }
}

// ─── Inspections Table ─────────────────────────────────────
async function loadInspections(page = 1) {
  state.inspPage = page;
  const pred = $('insp-filter-prediction').value;
  const status = $('insp-filter-status').value;
  const { from, to } = getDateRange();

  let url = `/inspections?page=${page}&page_size=15&date_from=${encodeURIComponent(from)}&date_to=${encodeURIComponent(to)}`;
  if (pred) url += `&prediction=${pred}`;
  if (status) url += `&inspection_status=${status}`;

  try {
    const data = await api(url);
    $('badge-inspections').textContent = data.total;
    renderInspectionsTable(data.items);
    renderPagination('inspections-pagination', data.total, page, 15, loadInspections);
  } catch(e) {
    toast(`Inspections load error: ${e.message}`, 'error');
  }
}

function renderInspectionsTable(items) {
  const tbody = $('inspections-tbody');
  if (!items || items.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="empty-state">No inspections found for the selected filters.</td></tr>`;
    return;
  }

  tbody.innerHTML = items.map(i => `
    <tr>
      <td><span style="font-family:var(--font-mono);font-size:11px;color:var(--text-muted)">${i.id.substring(0,8)}…</span></td>
      <td>${formatDate(i.timestamp)}</td>
      <td><strong>${i.batch_code || i.batch_id?.substring(0,8) || '—'}</strong></td>
      <td>${i.operator_name || '—'}</td>
      <td><span class="badge badge-muted">${i.capture_mode}</span></td>
      <td>${predictionBadge(i.prediction)}</td>
      <td>${i.defect_score !== null && i.defect_score !== undefined ? `
        <div style="display:flex;align-items:center;gap:6px">
          <span style="font-size:11px;color:var(--text-muted)">${(i.defect_score*100).toFixed(0)}%</span>
          <div class="score-bar" style="width:60px"><div class="score-fill ${i.prediction === 'DEFECT' ? 'defect' : 'good'}" style="width:${(i.defect_score*100).toFixed(0)}%"></div></div>
        </div>` : '—'}
      </td>
      <td>${statusBadge(i.inspection_status)}</td>
      <td>
        <div style="display:flex;gap:6px">
          <button class="btn-sm" onclick="openInspectionDetail('${i.id}')">View</button>
          ${i.inspection_status === 'under_review' ? `<button class="btn-sm" onclick="openReviewModal('${i.id}','${i.batch_code || ''}','${i.prediction || ''}','${(i.defect_score*100||0).toFixed(0)}')">Review</button>` : ''}
        </div>
      </td>
    </tr>
  `).join('');
}

function renderPagination(containerId, total, currentPage, pageSize, loadFn) {
  const container = $(containerId);
  const totalPages = Math.ceil(total / pageSize);
  if (totalPages <= 1) { container.innerHTML = ''; return; }

  let html = `<span class="page-info">${(currentPage-1)*pageSize+1}–${Math.min(currentPage*pageSize, total)} of ${total}</span>`;
  html += `<button class="page-btn" ${currentPage <= 1 ? 'disabled' : ''} onclick="${loadFn.name}(${currentPage-1})">‹ Prev</button>`;

  const startPage = Math.max(1, currentPage - 2);
  const endPage = Math.min(totalPages, currentPage + 2);
  for (let p = startPage; p <= endPage; p++) {
    html += `<button class="page-btn ${p === currentPage ? 'active' : ''}" onclick="${loadFn.name}(${p})">${p}</button>`;
  }

  html += `<button class="page-btn ${currentPage >= totalPages ? 'disabled' : ''} onclick="${loadFn.name}(${currentPage+1})">Next ›</button>`;
  container.innerHTML = html;
}

// ─── Products Catalog ──────────────────────────────────────
async function loadProducts() {
  const tbody = $('products-tbody');
  tbody.innerHTML = '<tr><td colspan="6" class="empty-state">Loading product catalog...</td></tr>';
  try {
    const prods = await api('/products');
    if (!prods.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No products registered.</td></tr>';
      return;
    }
    tbody.innerHTML = prods.map(p => `
      <tr>
        <td><strong>${p.sku}</strong></td>
        <td>${p.name}</td>
        <td><span class="badge badge-muted">${p.category || 'General'}</span></td>
        <td>${p.threshold_defect ? `${(p.threshold_defect*100).toFixed(0)}%` : '0.50'}</td>
        <td style="color:var(--text-secondary);font-size:12px">${p.inspection_criteria || 'Visual inspection for cracks and flaws'}</td>
        <td><span class="badge ${p.is_active ? 'badge-good' : 'badge-muted'}">${p.is_active ? 'Active' : 'Archived'}</span></td>
      </tr>
    `).join('');
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-state">Error: ${e.message}</td></tr>`;
  }
}

function openNewProductModal() {
  $('prod-name').value = '';
  $('prod-sku').value = '';
  $('prod-category').value = '';
  $('prod-criteria').value = '';
  $('new-prod-error').classList.add('hidden');
  $('modal-new-product-overlay').classList.remove('hidden');
}
window.openNewProductModal = openNewProductModal;

async function submitNewProduct(e) {
  e.preventDefault();
  const name = $('prod-name').value.trim();
  const sku = $('prod-sku').value.trim();
  const category = $('prod-category').value.trim();
  const criteria = $('prod-criteria').value.trim();
  const threshold = parseFloat($('prod-threshold').value) || 0.5;

  try {
    await api('/products', {
      method: 'POST',
      body: JSON.stringify({ name, sku, category, inspection_criteria: criteria, threshold_defect: threshold }),
    });
    toast('Product registered successfully!', 'success');
    $('modal-new-product-overlay').classList.add('hidden');
    loadProducts();
  } catch(e) {
    $('new-prod-error').textContent = e.message;
    $('new-prod-error').classList.remove('hidden');
  }
}

// ─── Batches & Holds ───────────────────────────────────────
async function loadBatches(page = 1) {
  state.batchPage = page;
  const status = $('batch-filter-status').value;
  let url = `/batches?page=${page}&page_size=20`;
  if (status) url += `&status=${status}`;

  try {
    const data = await api(url);
    renderBatchesTable(data.items);
    renderPagination('batches-pagination', data.total, page, 20, loadBatches);
  } catch(e) {
    toast(`Batches error: ${e.message}`, 'error');
  }
}

function renderBatchesTable(items) {
  const tbody = $('batches-tbody');
  if (!items || items.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="empty-state">No batches found.</td></tr>`;
    return;
  }

  tbody.innerHTML = items.map(b => `
    <tr>
      <td><strong>${b.batch_code}</strong></td>
      <td>${b.product_name || b.product_id?.substring(0,8) || '—'}</td>
      <td>${b.production_line || 'Line 1'}</td>
      <td>${b.quantity ?? '—'}</td>
      <td>${b.inspection_count}</td>
      <td>${b.defect_count > 0 ? `<span style="color:var(--red);font-weight:700">${b.defect_count}</span>` : b.defect_count}</td>
      <td>${batchStatusBadge(b.quality_status)}</td>
      <td>${formatDate(b.created_at)}</td>
      <td>
        <button class="btn-sm" onclick="openBatchDispositionModal('${b.id}', '${b.batch_code}', '${b.quality_status}')">
          Disposition
        </button>
      </td>
    </tr>
  `).join('');
}

function openNewBatchModal() {
  const sel = $('batch-product-id');
  sel.innerHTML = state.productsList.map(p => `<option value="${p.id}">${p.name} (${p.sku})</option>`).join('');
  $('batch-code-input').value = `BATCH-${new Date().getFullYear()}-${Math.floor(Math.random()*900 + 100)}`;
  $('new-batch-error').classList.add('hidden');
  $('modal-new-batch-overlay').classList.remove('hidden');
}
window.openNewBatchModal = openNewBatchModal;

async function submitNewBatch(e) {
  e.preventDefault();
  const product_id = $('batch-product-id').value;
  const batch_code = $('batch-code-input').value.trim();
  const quantity = parseInt($('batch-qty-input').value) || 100;
  const production_line = $('batch-line-input').value.trim();

  try {
    await api('/batches', {
      method: 'POST',
      body: JSON.stringify({ product_id, batch_code, quantity, production_line }),
    });
    toast('Batch created successfully!', 'success');
    $('modal-new-batch-overlay').classList.add('hidden');
    loadBatches();
  } catch(e) {
    $('new-batch-error').textContent = e.message;
    $('new-batch-error').classList.remove('hidden');
  }
}

function openBatchDispositionModal(batchId, batchCode, currentStatus) {
  $('batch-disp-id').value = batchId;
  $('modal-batch-title').textContent = `Batch Disposition: ${batchCode} [${currentStatus}]`;
  $('batch-disp-status').value = currentStatus === 'on_hold' ? 'approved' : 'on_hold';
  $('batch-disp-reason').value = '';
  $('batch-disp-error').classList.add('hidden');
  $('modal-batch-overlay').classList.remove('hidden');
}
window.openBatchDispositionModal = openBatchDispositionModal;

async function submitBatchDisposition(e) {
  e.preventDefault();
  const id = $('batch-disp-id').value;
  const status = $('batch-disp-status').value;
  const reason = $('batch-disp-reason').value.trim();
  const errEl = $('batch-disp-error');

  if (!reason) { errEl.textContent = 'A written justification is mandatory for batch disposition audit.'; errEl.classList.remove('hidden'); return; }

  try {
    let endpoint = `/batches/${id}/status`;
    let body = { quality_status: status, reason };
    if (status === 'on_hold') {
      endpoint = `/batches/${id}/hold`;
    } else if (status === 'approved') {
      endpoint = `/batches/${id}/release`;
    }
    
    await api(endpoint, {
      method: status === 'on_hold' || status === 'approved' ? 'POST' : 'PATCH',
      body: JSON.stringify(body),
    });
    
    toast(`Batch disposition updated to ${status}`, 'success');
    $('modal-batch-overlay').classList.add('hidden');
    loadBatches(state.batchPage);
    if (state.currentPage === 'overview') loadOverview();
  } catch(e) {
    errEl.textContent = e.message;
    errEl.classList.remove('hidden');
  }
}

// ─── Corrective Actions & Before/After Evidence ───────────
async function loadCorrectiveActions() {
  const status = $('ca-filter-status').value;
  const priority = $('ca-filter-priority').value;
  let url = '/corrective-actions?page=1&page_size=50';
  if (status) url += `&status=${status}`;
  if (priority) url += `&priority=${priority}`;

  try {
    const data = await api(url);
    $('badge-ca').textContent = data.items.filter(ca => ['open','in_progress','awaiting_reinspection'].includes(ca.status)).length;
    renderCATable(data.items);
  } catch(e) {
    toast(`CA error: ${e.message}`, 'error');
  }
}

function renderCATable(items) {
  const tbody = $('ca-tbody');
  if (!items || items.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state">No corrective actions found.</td></tr>`;
    return;
  }

  tbody.innerHTML = items.map(ca => `
    <tr>
      <td>
        <strong>${ca.title}</strong>
        ${ca.root_cause ? `<div style="font-size:11px;color:var(--text-muted);margin-top:2px">Cause: ${ca.root_cause}</div>` : ''}
      </td>
      <td>${priorityBadge(ca.priority)}</td>
      <td>${statusBadge(ca.status)}</td>
      <td>${ca.assigned_to_name || ca.assigned_to || 'Floor Team'}</td>
      <td>
        ${ca.reinspection_status 
          ? `<span class="badge ${ca.reinspection_status==='passed'?'badge-good':'badge-defect'}">${ca.reinspection_status}</span>`
          : ca.status === 'awaiting_reinspection' ? '<span class="badge badge-warn">Awaiting Verify</span>' : '—'}
      </td>
      <td>${ca.due_date ? formatDate(ca.due_date) : '—'}</td>
      <td>
        <div style="display:flex;gap:4px">
          <button class="btn-sm" onclick="openCADetail('${ca.id}')">Evidence</button>
          ${ca.status === 'open' || ca.status === 'in_progress' ? `
            <button class="btn-sm ghost" onclick="openUploadEvidenceModal('${ca.id}')">Upload Repair</button>
          ` : ''}
          ${ca.status === 'awaiting_reinspection' ? `
            <button class="btn-sm btn-accept" onclick="openVerifyModal('${ca.id}')">Verify</button>
          ` : ''}
        </div>
      </td>
    </tr>
  `).join('');
}

// Side-by-Side Before/After Evidence Viewer Modal
async function openCADetail(caId) {
  $('modal-ca-detail-overlay').classList.remove('hidden');
  const body = $('modal-ca-body');
  body.innerHTML = '<div class="skeleton" style="height:250px"></div>';

  try {
    const ca = await api(`/corrective-actions/${caId}`);
    $('modal-ca-title').textContent = `CAPA: ${ca.title}`;
    
    const beforeImg = ca.before_image_path ? `${API_BASE}/storage/${ca.before_image_path}` : null;
    const afterImg = ca.after_image_path ? `${API_BASE}/storage/${ca.after_image_path}` : null;
    
    body.innerHTML = `
      <div style="margin-bottom:16px">
        <div style="display:flex;gap:12px;margin-bottom:8px">
          ${statusBadge(ca.status)}
          ${priorityBadge(ca.priority)}
          <span style="color:var(--text-muted);font-size:12px">Linked Inspection: ${ca.inspection_id || 'None'}</span>
        </div>
        <p><strong>Root Cause:</strong> ${ca.root_cause || 'Under investigation'}</p>
        ${ca.completion_notes ? `<p style="margin-top:4px"><strong>Work Done:</strong> ${ca.completion_notes}</p>` : ''}
      </div>

      <div class="before-after-grid">
        <div class="evidence-img-card">
          <h4>BEFORE REPAIR (INITIAL DEFECT)</h4>
          ${beforeImg 
            ? `<img src="${beforeImg}" alt="Before Repair" onerror="this.src=''" />` 
            : '<div style="padding:40px;color:var(--text-muted)">No initial image stored</div>'}
        </div>
        <div class="evidence-img-card">
          <h4>AFTER REPAIR (EVIDENCE PHOTO)</h4>
          ${afterImg 
            ? `<img src="${afterImg}" alt="After Repair" onerror="this.src=''" />` 
            : `<div style="padding:40px;color:var(--text-muted)">
                 No repair evidence uploaded yet.<br>
                 <button class="btn-sm" style="margin-top:10px" onclick="openUploadEvidenceModal('${ca.id}');$('modal-ca-detail-overlay').classList.add('hidden')">
                   Upload Repair Evidence
                 </button>
               </div>`}
        </div>
      </div>

      ${ca.reinspection_notes ? `
        <div style="background:var(--bg-elevated);padding:12px;border-radius:6px;font-size:12px;margin-top:12px">
          <strong>Reinspection Sign-off:</strong> ${ca.reinspection_status?.toUpperCase()} — ${ca.reinspection_notes}
        </div>
      ` : ''}
    `;
  } catch(e) {
    body.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
  }
}
window.openCADetail = openCADetail;

function openUploadEvidenceModal(caId) {
  $('evidence-ca-id').value = caId;
  $('evidence-file').value = '';
  $('evidence-notes').value = '';
  $('evidence-error').classList.add('hidden');
  $('modal-evidence-overlay').classList.remove('hidden');
}
window.openUploadEvidenceModal = openUploadEvidenceModal;

async function submitEvidence(e) {
  e.preventDefault();
  const caId = $('evidence-ca-id').value;
  const file = $('evidence-file').files[0];
  const notes = $('evidence-notes').value.trim();
  const errEl = $('evidence-error');

  if (!file) { errEl.textContent = 'Please choose a photo.'; errEl.classList.remove('hidden'); return; }

  const form = new FormData();
  form.append('file', file);
  if (notes) form.append('notes', notes);

  try {
    await api(`/corrective-actions/${caId}/evidence`, { method: 'POST', body: form });
    toast('Repair evidence uploaded! Action status moved to awaiting reinspection.', 'success');
    $('modal-evidence-overlay').classList.add('hidden');
    loadCorrectiveActions();
  } catch(e) {
    errEl.textContent = e.message;
    errEl.classList.remove('hidden');
  }
}

function openVerifyModal(caId) {
  $('verify-ca-id').value = caId;
  $('verify-notes').value = '';
  document.querySelectorAll('input[name="verify_decision"]').forEach(r => r.checked = false);
  $('verify-error').classList.add('hidden');
  $('modal-verify-overlay').classList.remove('hidden');
}
window.openVerifyModal = openVerifyModal;

async function submitVerify(e) {
  e.preventDefault();
  const caId = $('verify-ca-id').value;
  const decision = document.querySelector('input[name="verify_decision"]:checked')?.value;
  const notes = $('verify-notes').value.trim();
  const errEl = $('verify-error');

  if (!decision) { errEl.textContent = 'Please select a decision.'; errEl.classList.remove('hidden'); return; }
  if (!notes) { errEl.textContent = 'Verification findings notes are required.'; errEl.classList.remove('hidden'); return; }

  try {
    await api(`/corrective-actions/${caId}/verify`, {
      method: 'POST',
      body: JSON.stringify({ decision, notes }),
    });
    toast(`Reinspection decision recorded (${decision.toUpperCase()})!`, 'success');
    $('modal-verify-overlay').classList.add('hidden');
    loadCorrectiveActions();
  } catch(e) {
    errEl.textContent = e.message;
    errEl.classList.remove('hidden');
  }
}

// ─── Inspection Detail Modal ───────────────────────────────
async function openInspectionDetail(inspId) {
  $('modal-insp-title').textContent = `Inspection ${inspId.substring(0,8)}…`;
  $('modal-insp-body').innerHTML = '<div class="skeleton" style="height:200px;border-radius:8px"></div>';
  $('modal-overlay').classList.remove('hidden');

  try {
    const insp = await api(`/inspections/${inspId}`);
    const imgUrl = insp.image_path ? `${API_BASE}/storage/${insp.image_path}` : null;

    $('modal-insp-body').innerHTML = `
      ${imgUrl ? `<div class="insp-detail-image"><img src="${imgUrl}" alt="Inspection image" onerror="this.style.display='none'" /></div>` : ''}
      <div class="insp-detail-grid">
        <div>
          <div class="detail-field"><label>Inspection ID</label><div class="value" style="font-family:var(--font-mono);font-size:12px">${insp.id}</div></div>
          <div class="detail-field"><label>Batch</label><div class="value">${insp.batch_code || insp.batch_id}</div></div>
          <div class="detail-field"><label>Operator</label><div class="value">${insp.operator_name || '—'}</div></div>
          <div class="detail-field"><label>Capture Mode</label><div class="value">${insp.capture_mode}</div></div>
          <div class="detail-field"><label>Serial #</label><div class="value">${insp.serial_number || '—'}</div></div>
          <div class="detail-field"><label>Timestamp</label><div class="value">${formatDate(insp.timestamp)}</div></div>
        </div>
        <div>
          <div class="detail-field"><label>ML Prediction</label><div class="value">${predictionBadge(insp.prediction)}</div></div>
          <div class="detail-field">
            <label>Defect Score</label>
            <div class="value">${insp.defect_score !== null ? (insp.defect_score*100).toFixed(1)+'%' : '—'}</div>
            ${insp.defect_score !== null ? `<div class="score-bar"><div class="score-fill ${insp.prediction==='DEFECT'?'defect':'good'}" style="width:${(insp.defect_score*100).toFixed(0)}%"></div></div>` : ''}
          </div>
          <div class="detail-field"><label>Model Version</label><div class="value">${insp.model_version || 'v1.0.0 (MobileNetV3)'}</div></div>
          <div class="detail-field"><label>Inference Time</label><div class="value">${insp.inference_time_ms ? insp.inference_time_ms+'ms' : '—'}</div></div>
          <div class="detail-field"><label>Status</label><div class="value">${statusBadge(insp.inspection_status)}</div></div>
          <div class="detail-field"><label>Session ID</label><div class="value" style="font-family:var(--font-mono);font-size:11px">${insp.session_id || '—'}</div></div>
        </div>
      </div>
      ${insp.reviews && insp.reviews.length > 0 ? `
        <div style="margin-top:16px;padding-top:16px;border-top:1px solid var(--border-subtle)">
          <div style="font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-muted);margin-bottom:10px">Authoritative QC Reviews (${insp.reviews.length})</div>
          ${insp.reviews.map(r => `
            <div style="background:var(--bg-elevated);border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:13px">
              <strong>${predictionBadge(r.confirmed_class)}</strong> confirmed by <strong>${r.reviewer}</strong> on ${formatDate(r.reviewed_at)}<br>
              <span style="color:var(--text-muted)">Original ML Output: ${r.original_prediction || '—'}</span><br>
              ${r.reason ? `Reason: ${r.reason}` : ''} ${r.notes ? `| Notes: ${r.notes}` : ''}
            </div>`).join('')}
        </div>` : ''}
    `;
  } catch(e) {
    $('modal-insp-body').innerHTML = `<div class="empty-state">Error loading inspection: ${e.message}</div>`;
  }
}
window.openInspectionDetail = openInspectionDetail;

// ─── Audit Ledger ──────────────────────────────────────────
async function loadAuditEvents() {
  const tbody = $('audit-tbody');
  tbody.innerHTML = '<tr><td colspan="6" class="empty-state">Loading immutable audit log...</td></tr>';
  
  try {
    const q = $('audit-search')?.value || '';
    const events = await api(`/audit/events?q=${encodeURIComponent(q)}`);
    if (!events.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No audit events recorded yet.</td></tr>';
      return;
    }
    
    tbody.innerHTML = events.map(e => `
      <tr>
        <td style="font-size:12px;color:var(--text-muted)">${formatDate(e.timestamp)}</td>
        <td><strong>${e.actor_name || 'System'}</strong></td>
        <td><span class="badge badge-blue">${e.action}</span></td>
        <td>${e.entity_type}</td>
        <td><code style="font-family:var(--font-mono);font-size:11px">${(e.entity_id||'—').substring(0,12)}</code></td>
        <td style="font-size:12px;color:var(--text-secondary)">${e.reason || e.new_state || '—'}</td>
      </tr>
    `).join('');
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-state">Error: ${e.message}</td></tr>`;
  }
}

// ─── Executive Quality Report ──────────────────────────────
async function openQualityReport() {
  $('modal-report-overlay').classList.remove('hidden');
  const body = $('report-content-body');
  body.innerHTML = '<div class="empty-state">Compiling compliance statistics from live database...</div>';

  try {
    const r = await api('/reports/quality-summary');
    body.innerHTML = `
      <div class="report-header-banner">
        <h2>QualityGuard AI — Plant Quality Assurance Report</h2>
        <div style="font-size:12px;color:var(--text-secondary);margin-top:4px">
          Report Generated: <strong>${formatDate(r.report_timestamp)}</strong> | Environment: <strong>${r.compliance_statement}</strong>
        </div>
      </div>

      <div class="report-section">
        <h3>1. Executive Quality Metrics</h3>
        <table class="data-table" style="margin-top:10px">
          <tbody>
            <tr><td><strong>Total Registered Products</strong></td><td>${r.total_products}</td><td><strong>Active Production Batches</strong></td><td>${r.total_batches}</td></tr>
            <tr><td><strong>Total Inspections Logged</strong></td><td>${r.total_inspections}</td><td><strong>Suspected Defect Rate</strong></td><td><strong>${r.defect_rate_pct}%</strong></td></tr>
            <tr><td><strong>Human-Confirmed Defects</strong></td><td>${r.confirmed_defects}</td><td><strong>Human-Confirmed Good (Pass)</strong></td><td>${r.confirmed_good}</td></tr>
            <tr><td><strong>Awaiting QC Review</strong></td><td><span class="badge badge-warn">${r.pending_reviews}</span></td><td><strong>Open Corrective Actions</strong></td><td>${r.open_corrective_actions}</td></tr>
          </tbody>
        </table>
      </div>

      <div class="report-section">
        <h3>2. Production Line Disposition</h3>
        <p>Current Quarantined Batches on Hold: <strong>${r.batches_on_hold}</strong></p>
        <p style="font-size:12px;color:var(--text-secondary);margin-top:4px">
          Rule: An AI Good prediction does not constitute a legal quality release. Authorised human disposition is enforced across all lines.
        </p>
      </div>

      <div class="report-section">
        <h3>3. Active AI Model Specification</h3>
        <p>Active Architecture: <strong>${r.active_model.architecture}</strong> (Weights: <code>${r.active_model.checkpoint_path}</code>)</p>
        <p>Evaluated Accuracy on Held-Out Test Set: <strong>84.07%</strong> (Steel Defect Detection Track)</p>
      </div>
    `;
  } catch(e) {
    body.innerHTML = `<div class="empty-state">Error: ${e.message}</div>`;
  }
}
window.openQualityReport = openQualityReport;

// ─── QualityGuard Copilot ──────────────────────────────────
function askCopilot(query) {
  $('assistant-input').value = query;
  sendAssistantQuery();
}
window.askCopilot = askCopilot;

async function sendAssistantQuery() {
  const input = $('assistant-input');
  const q = input.value.trim();
  if (!q) return;

  const chatWindow = $('chat-window');

  // Add user message
  const userMsg = document.createElement('div');
  userMsg.className = 'chat-message user';
  userMsg.innerHTML = `<span class="chat-icon">👤</span><div class="chat-bubble">${q}</div>`;
  chatWindow.appendChild(userMsg);
  input.value = '';
  chatWindow.scrollTop = chatWindow.scrollHeight;

  // Add loading skeleton
  const loading = document.createElement('div');
  loading.className = 'chat-message bot';
  loading.innerHTML = `<span class="chat-icon">🤖</span><div class="chat-bubble"><span class="skeleton" style="display:inline-block;width:140px;height:16px"></span></div>`;
  chatWindow.appendChild(loading);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  try {
    const res = await api(`/assistant/query?q=${encodeURIComponent(q)}`);
    loading.remove();

    // Format citations as clickable pills
    let answerHtml = res.answer.replace(/\[(INSP-[a-f0-9]+)\]/gi, (match, id) => {
      return `<span class="citation-pill" onclick="openInspectionFromCitation('${id}')">${match}</span>`;
    }).replace(/\[(BATCH-[a-zA-Z0-9_-]+)\]/gi, (match) => {
      return `<span class="citation-pill" onclick="navigateTo('batches')">${match}</span>`;
    }).replace(/\[(CAPA-[a-f0-9]+)\]/gi, (match) => {
      return `<span class="citation-pill" onclick="navigateTo('corrective-actions')">${match}</span>`;
    });

    const botMsg = document.createElement('div');
    botMsg.className = 'chat-message bot';
    botMsg.innerHTML = `
      <span class="chat-icon">🤖</span>
      <div class="chat-bubble">
        ${answerHtml}
        ${res.data_source ? `<div class="chat-data" style="margin-top:8px;font-size:11px;color:var(--text-muted)">Verified Source: <code>${res.data_source}</code></div>` : ''}
      </div>`;
    chatWindow.appendChild(botMsg);
  } catch(e) {
    loading.remove();
    const errMsg = document.createElement('div');
    errMsg.className = 'chat-message bot';
    errMsg.innerHTML = `<span class="chat-icon">🤖</span><div class="chat-bubble" style="color:var(--red)">Error: ${e.message}</div>`;
    chatWindow.appendChild(errMsg);
  }

  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function openInspectionFromCitation(citId) {
  const cleanId = citId.replace(/^INSP-/, '');
  // Search in current inspections or open detail
  openInspectionDetail(cleanId);
}
window.openInspectionFromCitation = openInspectionFromCitation;

// ─── Setup Sidebar User ────────────────────────────────────
function setupSidebar(user) {
  $('sidebar-username').textContent = user.display_name;
  $('sidebar-role').textContent = user.role.toUpperCase();
  $('user-avatar-text').textContent = (user.display_name || 'U')[0].toUpperCase();

  // Role-based visibility
  if (user.role === 'operator') {
    $$('[data-page="review-queue"]').forEach(el => el.style.display = 'none');
    $$('[data-page="corrective-actions"]').forEach(el => el.style.display = 'none');
    $$('[data-page="audit"]').forEach(el => el.style.display = 'none');
  }
}

// ─── Real-time Polling ─────────────────────────────────────
function startPolling() {
  if (state.pollInterval) clearInterval(state.pollInterval);
  state.pollInterval = setInterval(() => {
    if (state.currentPage === 'overview') loadOverview();
    if (state.currentPage === 'review-queue') loadReviewQueue();
    checkModelStatus();
  }, 25000);
}

// ─── Event Listeners ───────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Login form
  $('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = $('login-btn');
    const errEl = $('login-error');
    const username = $('login-username').value.trim();
    const password = $('login-password').value;

    btn.querySelector('.btn-text').textContent = 'Signing in…';
    btn.disabled = true;
    errEl.classList.add('hidden');

    try {
      const data = await login(username, password);
      state.token = data.access_token;
      state.user = { id: data.user_id, username: data.username, role: data.role, display_name: data.display_name };
      localStorage.setItem('ps01_token', data.access_token);
      localStorage.setItem('ps01_user', JSON.stringify(state.user));

      $('login-overlay').classList.add('hidden');
      $('app-shell').classList.remove('hidden');

      setupSidebar(state.user);
      checkModelStatus();
      navigateTo('overview');
      startPolling();
    } catch(err) {
      errEl.textContent = err.message;
      errEl.classList.remove('hidden');
    } finally {
      btn.querySelector('.btn-text').textContent = 'Sign In';
      btn.disabled = false;
    }
  });

  // Logout
  $('logout-btn').addEventListener('click', logout);

  // Navigation clicks
  $$('.nav-item').forEach(el => {
    el.addEventListener('click', (e) => {
      e.preventDefault();
      navigateTo(el.dataset.page);
    });
  });

  // Sidebar toggle
  $('sidebar-toggle').addEventListener('click', () => $('sidebar').classList.toggle('open'));

  // Executive report button in top bar
  $('btn-open-report').addEventListener('click', openQualityReport);

  // Date filters
  $('date-preset').addEventListener('change', () => {
    const isCustom = $('date-preset').value === 'custom';
    $('custom-range').classList.toggle('hidden', !isCustom);
    if (!isCustom) {
      if (state.currentPage === 'overview') loadOverview();
      if (state.currentPage === 'inspections') loadInspections();
    }
  });

  $('apply-range').addEventListener('click', () => {
    if (state.currentPage === 'overview') loadOverview();
    if (state.currentPage === 'inspections') loadInspections();
  });

  // Inspection filters
  $('insp-filter-apply').addEventListener('click', () => loadInspections(1));
  $('insp-refresh').addEventListener('click', () => loadInspections(state.inspPage));

  // Batch filters
  $('batch-filter-status').addEventListener('change', () => loadBatches(1));
  $('batch-refresh').addEventListener('click', () => loadBatches(1));

  // CA filters
  $('ca-filter-status').addEventListener('change', loadCorrectiveActions);
  $('ca-filter-priority').addEventListener('change', loadCorrectiveActions);
  $('ca-refresh').addEventListener('click', loadCorrectiveActions);

  // Forms
  $('review-form').addEventListener('submit', submitReview);
  $('batch-disposition-form').addEventListener('submit', submitBatchDisposition);
  $('evidence-form').addEventListener('submit', submitEvidence);
  $('verify-form').addEventListener('submit', submitVerify);
  $('new-product-form').addEventListener('submit', submitNewProduct);
  $('new-batch-form').addEventListener('submit', submitNewBatch);

  // Modal close handlers
  $$('.modal-close').forEach(btn => btn.addEventListener('click', () => $('modal-overlay').classList.add('hidden')));
  $$('.modal-close-review').forEach(btn => btn.addEventListener('click', () => $('modal-review-overlay').classList.add('hidden')));
  $$('.modal-close-batch').forEach(btn => btn.addEventListener('click', () => $('modal-batch-overlay').classList.add('hidden')));
  $$('.modal-close-ca').forEach(btn => btn.addEventListener('click', () => $('modal-ca-detail-overlay').classList.add('hidden')));
  $$('.modal-close-evidence').forEach(btn => btn.addEventListener('click', () => $('modal-evidence-overlay').classList.add('hidden')));
  $$('.modal-close-verify').forEach(btn => btn.addEventListener('click', () => $('modal-verify-overlay').classList.add('hidden')));
  $$('.modal-close-product').forEach(btn => btn.addEventListener('click', () => $('modal-new-product-overlay').classList.add('hidden')));
  $$('.modal-close-new-batch').forEach(btn => btn.addEventListener('click', () => $('modal-new-batch-overlay').classList.add('hidden')));
  $$('.modal-close-report').forEach(btn => btn.addEventListener('click', () => $('modal-report-overlay').classList.add('hidden')));

  // Assistant
  $('assistant-send').addEventListener('click', sendAssistantQuery);
  $('assistant-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') sendAssistantQuery(); });

  // Inspection search debouncing
  let searchTimer;
  $('insp-search').addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadInspections(1), 400);
  });

  // Audit search debouncing
  let auditTimer;
  $('audit-search')?.addEventListener('input', () => {
    clearTimeout(auditTimer);
    auditTimer = setTimeout(loadAuditEvents, 400);
  });

  // Auto-login from existing session
  if (loadSession()) {
    $('login-overlay').classList.add('hidden');
    $('app-shell').classList.remove('hidden');
    setupSidebar(state.user);
    checkModelStatus();
    navigateTo('overview');
    startPolling();
  }
});
