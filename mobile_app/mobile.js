/**
 * PS01 Product Quality Inspector – Mobile App JavaScript
 * ========================================================
 * PWA mobile app with:
 * - Authentication
 * - Batch & product selection
 * - Live camera capture (getUserMedia)
 * - Photo upload / gallery
 * - Real inference via backend /inspections endpoint
 * - Result display with safety-appropriate messaging
 * - Inspection history
 * - Evidence-Aware: multiple photos per session
 */

const API_BASE = localStorage.getItem('api_base') || (window.location.origin.startsWith('http') ? window.location.origin : 'http://localhost:8000');

// ─── State ──────────────────────────────────────────────
const state = {
  token: null,
  user: null,
  currentScreen: 'login',
  previousScreen: null,
  
  // Inspection session
  selectedProduct: null,
  selectedBatch: null,
  serialNumber: '',
  sessionId: null,
  photoIndex: 1,
  
  // Camera
  cameraStream: null,
  facingMode: 'environment',
  
  // Current photo
  capturedImageBytes: null,
  capturedFilename: null,
  
  // Last inspection
  lastInspection: null,
  
  // History
  historyPage: 1,
};

// ─── Utilities ───────────────────────────────────────────
const $ = id => document.getElementById(id);
const screen = id => document.getElementById(`screen-${id}`);

function api(path, opts = {}) {
  const headers = { ...opts.headers };
  if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
  if (opts.body && typeof opts.body === 'object' && !(opts.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(opts.body);
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

function toast(msg, type = 'info', duration = 3000) {
  const icons = { success: '✅', error: '❌', warn: '⚠️', info: 'ℹ️' };
  const t = document.createElement('div');
  t.className = `m-toast ${type}`;
  t.innerHTML = `<span>${icons[type]}</span><span>${msg}</span>`;
  $('m-toast-container').appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity 0.3s'; setTimeout(() => t.remove(), 300); }, duration);
}

function formatDate(dt) {
  if (!dt) return '—';
  return new Date(dt).toLocaleString('en-IN', { dateStyle: 'short', timeStyle: 'short' });
}

function newSessionId() {
  return 'sess-' + Date.now().toString(36) + '-' + Math.random().toString(36).substring(2, 7);
}

// ─── Screen navigation ───────────────────────────────────
function showScreen(id, slideBack = false) {
  const prev = state.currentScreen;
  
  // Remove active from all
  document.querySelectorAll('.screen').forEach(el => {
    if (el.id === `screen-${prev}`) {
      el.classList.remove('active');
      if (!slideBack) el.classList.add('slide-out');
      setTimeout(() => el.classList.remove('slide-out'), 300);
    } else {
      el.classList.remove('active', 'slide-out');
    }
  });
  
  state.previousScreen = prev;
  state.currentScreen = id;
  
  const nextEl = screen(id);
  if (nextEl) {
    nextEl.classList.remove('slide-out');
    nextEl.classList.add('active');
  }
}

// ─── Auth ─────────────────────────────────────────────────
async function doLogin(username, password) {
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
  stopCamera();
  showScreen('login');
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

function qFill(u, p) {
  $('m-username').value = u;
  $('m-password').value = p;
}
window.qFill = qFill;

// ─── Home ─────────────────────────────────────────────────
async function loadHome() {
  const user = state.user;
  if (!user) return;
  
  const hour = new Date().getHours();
  const greet = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';
  $('home-greeting').textContent = `${greet}, ${user.display_name.split(' ')[0]}`;
  
  // Today's stats
  try {
    const today = new Date().toISOString().split('T')[0];
    const data = await api(`/dashboard/summary?date_from=${today}T00:00:00&date_to=${new Date().toISOString()}`);
    $('stat-today-total').textContent = data.total_inspections;
    $('stat-today-good').textContent = data.good_predictions;
    $('stat-today-defect').textContent = data.defect_suspected;
  } catch(e) {
    // Non-operators may not have dashboard access
  }
  
  loadRecentHistory();
}

async function loadRecentHistory() {
  try {
    const data = await api('/inspections?page=1&page_size=10');
    renderHistoryList($('history-list'), data.items, true);
  } catch(e) {
    $('history-list').innerHTML = '<div class="history-empty">Could not load history.</div>';
  }
}

function renderHistoryList(container, items, mini = false) {
  if (!items || items.length === 0) {
    container.innerHTML = '<div class="history-empty">No inspections found.</div>';
    return;
  }
  
  container.innerHTML = items.map(i => {
    const pred = i.prediction || 'FAILED';
    const badgeClass = pred === 'GOOD' ? 'good' : pred === 'DEFECT' ? 'defect' : pred === 'UNCERTAIN' ? 'uncertain' : 'failed';
    const icon = pred === 'GOOD' ? '✅' : pred === 'DEFECT' ? '❌' : pred === 'UNCERTAIN' ? '⚠️' : '—';
    const imgSrc = i.image_path ? `${API_BASE}/storage/${i.image_path.replace('images/','')}` : null;
    
    return `
      <div class="history-item" onclick="viewInspectionHistory('${i.id}')">
        <div class="history-item-icon">
          ${imgSrc ? `<img src="${imgSrc}" alt="item" onerror="this.parentElement.textContent='🔍'" />` : '🔍'}
        </div>
        <div class="history-item-info">
          <div class="history-item-batch">${i.batch_code || i.batch_id?.substring(0,10) || '—'}</div>
          <div class="history-item-meta">${formatDate(i.timestamp)} · ${i.capture_mode}</div>
        </div>
        <span class="history-badge ${badgeClass}">${icon} ${pred}</span>
      </div>
    `;
  }).join('');
}

function viewInspectionHistory(id) {
  // Simple alert for now; production would go to a detail screen
  toast(`Viewing inspection ${id.substring(0,8)}… (open dashboard for full details)`, 'info');
}

window.viewInspectionHistory = viewInspectionHistory;

// ─── Batch selection ─────────────────────────────────────
async function loadBatchSelectScreen() {
  // New session ID
  state.sessionId = newSessionId();
  state.photoIndex = 1;
  $('session-id').value = state.sessionId;
  $('operator-display').value = state.user?.display_name || '';
  
  // Load products
  try {
    const products = await api('/products');
    const sel = $('product-select');
    sel.innerHTML = '<option value="">Select product…</option>';
    products.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.name;
      sel.appendChild(opt);
    });
  } catch(e) {
    toast(`Could not load products: ${e.message}`, 'error');
  }
  
  // Load batches
  try {
    const data = await api('/batches?page=1&page_size=50&status=in_inspection');
    const sel = $('batch-select');
    sel.innerHTML = '<option value="">Select batch…</option>';
    if (data.items) {
      data.items.forEach(b => {
        const opt = document.createElement('option');
        opt.value = b.id;
        opt.dataset.code = b.batch_code;
        opt.textContent = `${b.batch_code} (${b.quality_status})`;
        sel.appendChild(opt);
      });
    }
  } catch(e) {
    toast(`Could not load batches: ${e.message}`, 'warn');
  }
}

// ─── Capture method screen ────────────────────────────────
function setupCaptureMethodScreen() {
  const batchSel = $('batch-select');
  const selectedOpt = batchSel.options[batchSel.selectedIndex];
  const batchCode = selectedOpt?.dataset?.code || selectedOpt?.textContent || '';
  const productName = $('product-select').options[$('product-select').selectedIndex]?.textContent || '';
  
  $('capture-batch-info').innerHTML = `
    <div><strong>Product:</strong> ${productName}</div>
    <div><strong>Batch:</strong> ${batchCode || state.selectedBatch}</div>
    <div><strong>Session:</strong> <code style="font-size:12px">${state.sessionId}</code></div>
    ${$('serial-number').value ? `<div><strong>Serial:</strong> ${$('serial-number').value}</div>` : ''}
    <div style="margin-top:8px;font-size:12px;color:var(--text3)">Photo ${state.photoIndex} of this session</div>
  `;
}

// ─── Camera ──────────────────────────────────────────────
async function startCamera() {
  const video = $('camera-video');
  const errCard = $('camera-error');
  const hint = $('camera-hint');
  
  errCard.classList.add('hidden');
  
  // Stop any existing stream
  stopCamera();
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: state.facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    state.cameraStream = stream;
    video.srcObject = stream;
    hint.textContent = 'Position the item within the frame';
  } catch(err) {
    console.error('Camera error:', err);
    errCard.classList.remove('hidden');
    $('camera-error-msg').textContent = getCameraErrorMessage(err);
    hint.textContent = 'Camera unavailable — use Photo mode';
  }
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
    const video = $('camera-video');
    if (video) video.srcObject = null;
  }
}

function getCameraErrorMessage(err) {
  if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
    return 'Camera permission denied. Please allow camera access in browser settings and try again, or use Photo mode.';
  } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
    return 'No camera found on this device. Use Photo mode to upload an image.';
  } else if (err.name === 'NotReadableError') {
    return 'Camera is in use by another app. Close other apps and retry.';
  }
  return `Camera error: ${err.message}. Try Photo mode.`;
}

let mobileCountdownTimer = null;

function captureFrame() {
  const video = $('camera-video');
  const countEl = $('m-camera-countdown');
  
  if (!video.srcObject || !state.cameraStream) {
    toast('Camera not ready', 'error');
    return;
  }
  
  let countdown = 2;
  countEl.textContent = countdown;
  countEl.classList.remove('hidden');
  
  mobileCountdownTimer = setInterval(() => {
    countdown--;
    if (countdown > 0) {
      countEl.textContent = countdown;
    } else {
      clearInterval(mobileCountdownTimer);
      countEl.classList.add('hidden');
      freezeAndReviewFrame();
    }
  }, 1000);
}

function freezeAndReviewFrame() {
  const video = $('camera-video');
  const canvas = $('camera-canvas');
  const preview = $('m-camera-preview-frozen');
  const actions = $('m-camera-review-actions');
  const captureBtn = $('camera-capture-btn');
  
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  
  canvas.toBlob(async (blob) => {
    if (!blob) { toast('Failed to capture frame', 'error'); return; }
    
    const arrayBuffer = await blob.arrayBuffer();
    state.capturedImageBytes = new Uint8Array(arrayBuffer);
    state.capturedFilename = `camera_${Date.now()}.jpg`;
    
    // Freeze still preview on screen
    preview.src = URL.createObjectURL(blob);
    preview.classList.remove('hidden');
    captureBtn.classList.add('hidden');
    actions.classList.remove('hidden');
  }, 'image/jpeg', 0.95);
}

function retakeMobileFrame() {
  $('m-camera-preview-frozen').classList.add('hidden');
  $('m-camera-review-actions').classList.add('hidden');
  $('camera-capture-btn').classList.remove('hidden');
  state.capturedImageBytes = null;
}

function acceptMobileFrame() {
  $('m-camera-preview-frozen').classList.add('hidden');
  $('m-camera-review-actions').classList.add('hidden');
  $('camera-capture-btn').classList.remove('hidden');
  stopCamera();
  runInspection('camera');
}

function flipCamera() {
  state.facingMode = state.facingMode === 'environment' ? 'user' : 'environment';
  startCamera();
}

// ─── Photo capture ────────────────────────────────────────
function handlePhotoFile(file) {
  if (!file) return;
  
  const errEl = $('photo-error');
  errEl.classList.add('hidden');
  
  // Validate
  const allowed = ['image/jpeg','image/png','image/webp','image/bmp'];
  if (!allowed.includes(file.type)) {
    errEl.textContent = `File type not supported: ${file.type}. Use JPG, PNG or WebP.`;
    errEl.classList.remove('hidden');
    return;
  }
  
  const maxMB = 20;
  if (file.size > maxMB * 1024 * 1024) {
    errEl.textContent = `File too large (${(file.size/1024/1024).toFixed(1)} MB). Max ${maxMB} MB.`;
    errEl.classList.remove('hidden');
    return;
  }
  
  $('photo-meta').textContent = `${(file.size/1024).toFixed(0)} KB • ${file.type}`;
  $('photo-meta').classList.remove('hidden');
  
  const reader = new FileReader();
  reader.onload = (e) => {
    // Show preview
    const img = $('photo-preview-img');
    img.src = e.target.result;
    img.classList.remove('hidden');
    $('photo-placeholder').classList.add('hidden');
    
    // Store bytes
    const arr = new Uint8Array(e.target.result);
    state.capturedImageBytes = arr;
    state.capturedFilename = file.name;
    
    $('inspect-photo-btn').disabled = false;
  };
  reader.readAsArrayBuffer(file);
}

// ─── Inspection ───────────────────────────────────────────
async function runInspection(captureMode = 'photo') {
  showScreen('processing');
  
  const steps = {
    validate: $('step-validate'),
    infer: $('step-infer'),
    save: $('step-save'),
  };
  
  // Reset steps
  Object.values(steps).forEach(el => { el.className = 'step'; el.textContent = el.textContent.replace(/^[✓⟳○✕]\s/, '○ '); });
  
  // Step 1: Validate image
  steps.validate.className = 'step active';
  steps.validate.textContent = '⟳ Validating image…';
  
  if (!state.capturedImageBytes) {
    steps.validate.textContent = '✕ No image to inspect';
    steps.validate.className = 'step error';
    setTimeout(() => showScreen(state.previousScreen || 'home', true), 2000);
    return;
  }
  
  await sleep(400);
  steps.validate.textContent = '✓ Validating image';
  steps.validate.className = 'step done';
  steps.infer.className = 'step active';
  steps.infer.textContent = '⟳ Running inference…';
  
  let result = null;
  
  try {
    // Step 2: Submit inspection with image
    const formData = new FormData();
    const blob = new Blob([state.capturedImageBytes], { type: 'image/jpeg' });
    formData.append('file', blob, state.capturedFilename || 'inspection.jpg');
    formData.append('batch_id', state.selectedBatch);
    formData.append('capture_mode', captureMode);
    formData.append('session_id', state.sessionId);
    formData.append('photo_index', state.photoIndex.toString());
    formData.append('is_primary', state.photoIndex === 1 ? 'true' : 'false');
    if ($('serial-number')?.value) formData.append('serial_number', $('serial-number').value);
    
    const res = await fetch(`${API_BASE}/inspections`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${state.token}` },
      body: formData,
    });
    
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }
    
    result = await res.json();
    
    steps.infer.textContent = '✓ Running inference';
    steps.infer.className = 'step done';
    steps.save.className = 'step active';
    steps.save.textContent = '⟳ Saving result…';
    
    await sleep(300);
    steps.save.textContent = '✓ Saved';
    steps.save.className = 'step done';
    
    state.lastInspection = result;
    state.photoIndex++;
    
    await sleep(500);
    showResultScreen(result);
    
  } catch(e) {
    steps.infer.textContent = `✕ Failed: ${e.message}`;
    steps.infer.className = 'step';
    steps.infer.style.color = 'var(--red)';
    
    // Show failed result
    showResultScreen({
      success: false,
      error: e.message,
      inspection_status: 'INSPECTION_FAILED',
      prediction: 'FAILED',
      recommended_action: 'Check your network connection and backend status. Retry or capture a new image.',
    });
  }
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// ─── Result screen ────────────────────────────────────────
function showResultScreen(result) {
  const status = result.ml_status || result.inspection_status || 'INSPECTION_FAILED';
  const pred = result.prediction || 'FAILED';
  
  // Header styling
  const header = $('result-header');
  header.className = 'result-header';
  const iconMap = {
    GOOD_PREDICTION: '✅', DEFECT_SUSPECTED: '❌',
    NEEDS_MANUAL_REVIEW: '⚠️', INSPECTION_FAILED: '—',
    GOOD: '✅', DEFECT: '❌', UNCERTAIN: '⚠️', FAILED: '—',
  };
  const statusMap = {
    GOOD_PREDICTION: 'GOOD PREDICTION', DEFECT_SUSPECTED: 'DEFECT SUSPECTED',
    NEEDS_MANUAL_REVIEW: 'NEEDS MANUAL REVIEW', INSPECTION_FAILED: 'INSPECTION FAILED',
    GOOD: 'GOOD PREDICTION', DEFECT: 'DEFECT SUSPECTED', UNCERTAIN: 'NEEDS MANUAL REVIEW',
  };
  const classMap = {
    GOOD_PREDICTION: 'good', GOOD: 'good',
    DEFECT_SUSPECTED: 'defect', DEFECT: 'defect',
    NEEDS_MANUAL_REVIEW: 'uncertain', UNCERTAIN: 'uncertain',
    INSPECTION_FAILED: 'failed', FAILED: 'failed',
  };
  
  header.classList.add(classMap[status] || classMap[pred] || 'failed');
  $('result-icon').textContent = iconMap[status] || iconMap[pred] || '—';
  $('result-status').textContent = statusMap[status] || status;
  
  if (result.defect_score !== null && result.defect_score !== undefined) {
    $('result-confidence').textContent = `Defect score: ${(result.defect_score * 100).toFixed(1)}%`;
  } else {
    $('result-confidence').textContent = result.error ? 'Inference failed' : '';
  }
  
  // Image
  if (state.capturedImageBytes) {
    try {
      const blob = new Blob([state.capturedImageBytes], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      const img = $('result-image');
      img.src = url;
      img.classList.remove('hidden');
    } catch(e) {}
  }
  
  // Details
  $('res-batch').textContent = result.batch_id?.substring(0,12) || state.selectedBatch?.substring(0,12) || '—';
  $('res-id').textContent = result.inspection_id?.substring(0,16) || '—';
  $('res-operator').textContent = state.user?.display_name || '—';
  $('res-timestamp').textContent = result.timestamp ? formatDate(result.timestamp) : formatDate(new Date().toISOString());
  $('res-model-ver').textContent = result.model_version || '—';
  $('res-infer-time').textContent = result.inference_time_ms ? `${result.inference_time_ms}ms` : '—';
  
  // Scores
  const goodPct = result.good_score !== null && result.good_score !== undefined ? Math.round(result.good_score * 100) : 0;
  const defectPct = result.defect_score !== null && result.defect_score !== undefined ? Math.round(result.defect_score * 100) : 0;
  
  $('score-good-fill').style.width = `${goodPct}%`;
  $('score-defect-fill').style.width = `${defectPct}%`;
  $('score-good-pct').textContent = `${goodPct}%`;
  $('score-defect-pct').textContent = `${defectPct}%`;
  
  // Action
  $('action-text').textContent = result.recommended_action || 'Follow your authorised quality procedure.';
  
  // Show screen
  showScreen('result');
  
  // Refresh home stats
  if (result.success) {
    loadRecentHistory();
  }
}

// ─── History screen ───────────────────────────────────────
async function loadFullHistory() {
  const container = $('history-full-list');
  container.innerHTML = '<div class="history-empty">Loading…</div>';
  
  try {
    const data = await api('/inspections?page=1&page_size=30');
    renderHistoryList(container, data.items, false);
  } catch(e) {
    container.innerHTML = `<div class="history-empty">Error: ${e.message}</div>`;
  }
}

// ─── Settings ─────────────────────────────────────────────
function loadSettings() {
  $('api-url-setting').value = localStorage.getItem('api_base') || 'http://localhost:8000';
  $('settings-user').textContent = state.user ? `${state.user.display_name} (${state.user.role})` : '—';
  
  fetch(`${API_BASE}/health`).then(r => r.json()).then(data => {
    $('settings-model-status').textContent = data.model_ready 
      ? `Ready — v${data.model_version}` 
      : `Not ready: ${data.model_error}`;
  }).catch(() => {
    $('settings-model-status').textContent = 'Backend offline';
  });
}

// ─── Bootstrap ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  
  // ── Login ──
  $('mobile-login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = $('m-login-btn');
    const errEl = $('m-login-error');
    const username = $('m-username').value.trim();
    const password = $('m-password').value;
    
    btn.textContent = 'Signing in…';
    btn.disabled = true;
    errEl.classList.add('hidden');
    
    try {
      const data = await doLogin(username, password);
      state.token = data.access_token;
      state.user = { id: data.user_id, username: data.username, role: data.role, display_name: data.display_name };
      localStorage.setItem('ps01_token', data.access_token);
      localStorage.setItem('ps01_user', JSON.stringify(state.user));
      
      showScreen('home');
      await loadHome();
    } catch(err) {
      errEl.textContent = err.message;
      errEl.classList.remove('hidden');
    } finally {
      btn.textContent = 'Sign In';
      btn.disabled = false;
    }
  });
  
  // ── Logout buttons ──
  $('home-logout-btn').addEventListener('click', logout);
  $('settings-logout').addEventListener('click', logout);
  
  // ── New inspection ──
  $('new-inspection-btn').addEventListener('click', async () => {
    showScreen('batch-select');
    await loadBatchSelectScreen();
  });
  
  // ── Batch back ──
  $('batch-back').addEventListener('click', () => showScreen('home', true));
  
  // ── Proceed to capture ──
  $('proceed-to-capture').addEventListener('click', () => {
    const batchSel = $('batch-select');
    const errEl = $('batch-form-error');
    
    if (!batchSel.value) {
      errEl.textContent = 'Please select a batch.';
      errEl.classList.remove('hidden');
      return;
    }
    
    errEl.classList.add('hidden');
    state.selectedBatch = batchSel.value;
    state.selectedProduct = $('product-select').value;
    state.serialNumber = $('serial-number').value;
    
    // Update home batch display
    const batchCode = batchSel.options[batchSel.selectedIndex]?.textContent || '';
    $('selected-batch-display').innerHTML = `
      <div class="selected-batch">Batch: ${batchCode}</div>
      <div class="selected-batch-sub">Session: ${state.sessionId}</div>
    `;
    
    setupCaptureMethodScreen();
    showScreen('capture-select');
  });
  
  // ── Capture back ──
  $('capture-back').addEventListener('click', () => showScreen('batch-select', true));
  
  // ── Camera ──
  $('use-camera-btn').addEventListener('click', async () => {
    showScreen('camera');
    await startCamera();
  });
  
  $('camera-back').addEventListener('click', () => {
    stopCamera();
    showScreen('capture-select', true);
  });
  
  $('camera-flip').addEventListener('click', flipCamera);
  
  $('camera-capture-btn').addEventListener('click', captureFrame);
  $('m-btn-retake').addEventListener('click', retakeMobileFrame);
  $('m-btn-accept').addEventListener('click', acceptMobileFrame);
  
  $('camera-switch-photo').addEventListener('click', () => {
    stopCamera();
    showScreen('photo');
  });
  
  // ── Photo ──
  $('use-photo-btn').addEventListener('click', () => showScreen('photo'));
  $('photo-back').addEventListener('click', () => showScreen('capture-select', true));
  
  $('photo-file-take').addEventListener('change', (e) => handlePhotoFile(e.target.files[0]));
  $('photo-file-gallery').addEventListener('change', (e) => handlePhotoFile(e.target.files[0]));
  
  $('inspect-photo-btn').addEventListener('click', () => runInspection('photo'));
  
  $('add-more-photos-btn').addEventListener('click', () => {
    // Reset photo screen but keep session
    $('photo-preview-img').src = '';
    $('photo-preview-img').classList.add('hidden');
    $('photo-placeholder').classList.remove('hidden');
    $('photo-meta').classList.add('hidden');
    $('inspect-photo-btn').disabled = true;
    state.capturedImageBytes = null;
    setupCaptureMethodScreen();
    showScreen('photo');
  });
  
  // ── Result actions ──
  $('new-inspection-from-result').addEventListener('click', async () => {
    state.photoIndex = 1;
    state.sessionId = newSessionId();
    state.capturedImageBytes = null;
    showScreen('batch-select');
    await loadBatchSelectScreen();
  });
  
  $('add-photo-same-session').addEventListener('click', () => {
    state.capturedImageBytes = null;
    $('photo-preview-img').src = '';
    $('photo-preview-img').classList.add('hidden');
    $('photo-placeholder').classList.remove('hidden');
    $('photo-meta').classList.add('hidden');
    $('inspect-photo-btn').disabled = true;
    setupCaptureMethodScreen();
    showScreen('capture-select');
  });
  
  $('reinspect-btn').addEventListener('click', async () => {
    // Go back to capture with same session and batch
    state.capturedImageBytes = null;
    setupCaptureMethodScreen();
    showScreen('capture-select');
  });
  
  // ── Bottom nav ──
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      const target = btn.dataset.screen;
      if (target === 'home') {
        showScreen('home');
        loadHome();
      } else if (target === 'history') {
        showScreen('history');
        loadFullHistory();
      } else if (target === 'settings') {
        showScreen('settings');
        loadSettings();
      }
    });
  });
  
  // ── Settings ──
  $('save-api-url').addEventListener('click', () => {
    const url = $('api-url-setting').value.trim();
    if (url) {
      localStorage.setItem('api_base', url);
      toast('API URL saved. Restart needed.', 'success');
    }
  });
  
  // ── Auto-login from session ──
  if (loadSession()) {
    showScreen('home');
    loadHome();
  } else {
    showScreen('login');
  }
  
  // ── Prevent zoom on double tap on iOS ──
  document.addEventListener('touchend', (e) => {
    if (e.timeStamp - (document._lastTouch || 0) < 300) {
      e.preventDefault();
    }
    document._lastTouch = e.timeStamp;
  });
  
  // ── Stop camera when hidden ──
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.currentScreen === 'camera') {
      stopCamera();
    }
  });
});
