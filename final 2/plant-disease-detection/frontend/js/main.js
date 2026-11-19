/**
 * main.js — PlantGuard AI Frontend Logic
 * Handles: navbar scroll, image upload, prediction API calls,
 *          results display, dashboard stats, stat counter animations.
 * CVRT Engineering College | JNTUA B.Tech CSE 2024-25
 */

const API_BASE = '/api';

/* ── Navbar Scroll Effect ─────────────────────────────────── */
const navbar = document.getElementById('navbar');
if (navbar) {
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 40);
  });
}

/* ── Mobile Nav Toggle ────────────────────────────────────── */
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');
if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => navLinks.classList.toggle('open'));
  document.addEventListener('click', (e) => {
    if (!navbar.contains(e.target)) navLinks.classList.remove('open');
  });
}

/* ── Animated Stat Counters (Index Page) ────────────────────── */
function animateCounter(el) {
  const target = parseInt(el.getAttribute('data-target'), 10);
  const duration = 1500;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 4);
    el.textContent = Math.floor(eased * target);
    if (progress < 1) requestAnimationFrame(update);
    else el.textContent = target;
  }
  requestAnimationFrame(update);
}

const statNumbers = document.querySelectorAll('.stat-number[data-target]');
if (statNumbers.length > 0) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });
  statNumbers.forEach(el => observer.observe(el));
}

/* ════════════════════════════════════════════════════════════
   PREDICT PAGE — Image Upload & Prediction
   ════════════════════════════════════════════════════════════ */
const uploadZone     = document.getElementById('uploadZone');
const imageInput     = document.getElementById('imageInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview   = document.getElementById('imagePreview');
const removeImage    = document.getElementById('removeImage');
const predictBtn     = document.getElementById('predictBtn');
const predictBtnText = document.getElementById('predictBtnText');
const loadingContainer = document.getElementById('loadingContainer');
const errorAlert     = document.getElementById('errorAlert');
const errorMessage   = document.getElementById('errorMessage');

let selectedFile = null;

/* ── Drag & Drop Handling ──────────────────────────────────── */
if (uploadZone) {
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFileSelected(file);
  });
}

/* ── File Input Change ──────────────────────────────────────── */
if (imageInput) {
  imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelected(file);
  });
}

/** Preview selected image and enable the Predict button */
function handleFileSelected(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    previewContainer.style.display = 'block';
    uploadZone.style.display = 'none';
    predictBtn.disabled = false;
    predictBtnText.textContent = 'Analyze This Leaf';
    hideError();
  };
  reader.readAsDataURL(file);
}

/* ── Remove image ───────────────────────────────────────────── */
if (removeImage) {
  removeImage.addEventListener('click', () => {
    selectedFile = null;
    imageInput.value = '';
    previewContainer.style.display = 'none';
    uploadZone.style.display = '';
    predictBtn.disabled = true;
    predictBtnText.textContent = 'Select an Image First';
    hideError();
  });
}

/* ── Predict Button Click ───────────────────────────────────── */
if (predictBtn) {
  predictBtn.addEventListener('click', handlePredict);
}

/** Send the image to the Flask API and redirect to results page */
async function handlePredict() {
  if (!selectedFile) return;

  // Show loading state
  predictBtn.style.display = 'none';
  loadingContainer.style.display = 'block';
  hideError();

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Prediction failed. Please try again.');
    }

    // Store result in session storage, redirect to results page
    const resultData = {
      ...data.prediction,
      prediction_id: data.prediction_id,
      image_src: imagePreview.src   // base64 for display
    };
    sessionStorage.setItem('plantguard_result', JSON.stringify(resultData));
    window.location.href = '/results';

  } catch (err) {
    showError(err.message || 'An unexpected error occurred. Make sure the server is running.');
    predictBtn.style.display = 'flex';
    loadingContainer.style.display = 'none';
  }
}

function showError(msg) {
  if (errorAlert && errorMessage) {
    errorMessage.textContent = msg;
    errorAlert.style.display = 'flex';
  }
}
function hideError() {
  if (errorAlert) errorAlert.style.display = 'none';
}

/* ════════════════════════════════════════════════════════════
   RESULTS PAGE — Display Prediction Data
   ════════════════════════════════════════════════════════════ */

if (document.getElementById('resultsContent')) {
  const stored = sessionStorage.getItem('plantguard_result');

  if (!stored) {
    document.getElementById('noResults').style.display = 'block';
    document.getElementById('resultsContent').style.display = 'none';
  } else {
    const result = JSON.parse(stored);
    populateResults(result);
  }
}

function populateResults(result) {
  const noResults = document.getElementById('noResults');
  if (noResults) noResults.style.display = 'none';

  // Disease name & emoji
  const isHealthy = result.is_healthy;
  const emoji = isHealthy ? '✅' : '⚠️';
  const el = (id) => document.getElementById(id);

  if (el('resultEmoji')) el('resultEmoji').textContent = emoji;
  if (el('resultIndicator')) el('resultIndicator').classList.toggle('diseased', !isHealthy);
  if (el('diseaseName')) el('diseaseName').textContent = result.display_name || result.predicted_class;

  // Confidence badge
  const confidence = result.confidence || 0;
  const confPct = (confidence * 100).toFixed(1) + '%';
  if (el('confidenceBadge')) {
    el('confidenceBadge').textContent = confPct + ' Confidence';
    if (confidence < 0.6) el('confidenceBadge').classList.add('low');
  }
  if (el('confidencePercent')) el('confidencePercent').textContent = confPct;

  // Animated confidence bar
  setTimeout(() => {
    const fill = el('confidenceBarFill');
    if (fill) fill.style.width = (confidence * 100) + '%';
  }, 200);

  // Crop and severity chips
  if (el('cropChip')) el('cropChip').textContent = '🌱 ' + (result.crop_type || 'Unknown');
  if (el('severityChip')) {
    el('severityChip').textContent = '⚡ Severity: ' + (result.severity || 'Unknown');
    if (result.severity === 'Critical') el('severityChip').classList.add('critical');
  }

  // Description
  if (el('diseaseDescription')) el('diseaseDescription').textContent = result.description || '';

  // Treatment list
  const treatmentList = el('treatmentList');
  if (treatmentList && result.treatment) {
    treatmentList.innerHTML = result.treatment.map(t => `<li>${t}</li>`).join('');
  }

  // Prevention list
  const preventionList = el('preventionList');
  if (preventionList && result.prevention) {
    preventionList.innerHTML = result.prevention.map(p => `<li>${p}</li>`).join('');
  }

  // Top 3 predictions
  const top3List = el('top3List');
  if (top3List && result.top3) {
    top3List.innerHTML = result.top3.map((item, idx) => {
      const label = item.label.replace('___', ' — ').replace(/_/g, ' ');
      const pct = (item.confidence * 100).toFixed(1);
      return `
        <div class="top3-item">
          <span class="top3-rank">${idx + 1}</span>
          <span class="top3-label">${label}</span>
          <div class="top3-bar-bg">
            <div class="top3-bar-fill" style="width:${pct}%"></div>
          </div>
          <span class="top3-pct">${pct}%</span>
        </div>`;
    }).join('');
  }

  // Show analyzed image if available
  if (result.image_src && el('imageInfoCard') && el('analyzedImage')) {
    el('analyzedImage').src = result.image_src;
    el('imageInfoCard').style.display = 'block';
  }
}

/* ════════════════════════════════════════════════════════════
   DASHBOARD PAGE — Load Stats & History
   ════════════════════════════════════════════════════════════ */

if (document.getElementById('statsRow')) {
  loadDashboard();
}

async function loadDashboard() {
  // Load system stats
  try {
    const res = await fetch(`${API_BASE}/stats`);
    const data = await res.json();
    if (data.success) {
      const stats = data.stats;
      setInner('totalPredictions', stats.total_predictions ?? '—');
      setInner('healthyCount', stats.healthy_predictions ?? '—');
      setInner('diseasesDetected', stats.diseases_detected ?? '—');
      setInner('avgConfidence', stats.avg_confidence ? stats.avg_confidence + '%' : '—');
    }
  } catch (e) {
    console.warn('Could not load stats:', e);
  }

  // Load prediction history
  await loadHistory();
}

async function loadHistory() {
  const loading = document.getElementById('historyLoading');
  const empty = document.getElementById('historyEmpty');
  const tableWrapper = document.getElementById('historyTableWrapper');
  const tbody = document.getElementById('historyTableBody');

  try {
    const res = await fetch(`${API_BASE}/history?limit=20`);
    const data = await res.json();

    if (loading) loading.style.display = 'none';

    if (!data.success || data.count === 0) {
      if (empty) empty.style.display = 'block';
      return;
    }

    if (tableWrapper) tableWrapper.style.display = 'block';
    if (tbody) {
      tbody.innerHTML = data.predictions.map((p, i) => {
        const diseaseName = p.predicted_disease
          .replace('___', ' — ').replace(/_/g, ' ');
        const isHealthy = p.is_healthy === 1;
        const confPct = ((p.confidence_score || 0) * 100).toFixed(1) + '%';
        const date = new Date(p.predicted_at).toLocaleDateString('en-IN');
        const statusBadge = isHealthy
          ? '<span class="badge-healthy">✓ Healthy</span>'
          : '<span class="badge-disease">⚠ Disease Found</span>';

        return `
          <tr>
            <td>${data.count - i}</td>
            <td>${diseaseName}</td>
            <td>${p.crop_type || '—'}</td>
            <td>${confPct}</td>
            <td>${statusBadge}</td>
            <td>${date}</td>
          </tr>`;
      }).join('');
    }
  } catch (e) {
    if (loading) loading.style.display = 'none';
    if (empty) {
      empty.style.display = 'block';
      empty.innerHTML = '<div class="empty-icon">⚠️</div><p>Could not load history. Is the server running?</p>';
    }
  }
}

/* Refresh button on dashboard */
const refreshBtn = document.getElementById('refreshHistoryBtn');
if (refreshBtn) {
  refreshBtn.addEventListener('click', () => {
    const historyLoading = document.getElementById('historyLoading');
    const tableWrapper = document.getElementById('historyTableWrapper');
    const historyEmpty = document.getElementById('historyEmpty');
    if (historyLoading) historyLoading.style.display = 'block';
    if (tableWrapper) tableWrapper.style.display = 'none';
    if (historyEmpty) historyEmpty.style.display = 'none';
    loadHistory();
  });
}

/* ── Helper: set innerHTML safely ──────────────────────────── */
function setInner(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}
