document.addEventListener('DOMContentLoaded', () => {
  const imageInput = document.getElementById('imageInput');
  const processBtn = document.getElementById('processBtn');
  const techniqueSelect = document.getElementById('techniqueSelect');
  const resultCard = document.getElementById('resultCard');
  const origImg = document.getElementById('origImg');
  const procImg = document.getElementById('procImg');
  const origFeatures = document.getElementById('origFeatures');
  const procFeatures = document.getElementById('procFeatures');
  const downloadLink = document.getElementById('downloadLink');
  const claheClip = document.getElementById('claheClip');
  const claheVal = document.getElementById('claheVal');
  const gamma = document.getElementById('gamma');
  const gammaVal = document.getElementById('gammaVal');
  const templateInput = document.getElementById('templateInput');
  const origHistCanvas = document.getElementById('origHist');
  const procHistCanvas = document.getElementById('procHist');
  const resetBtn = document.getElementById('resetBtn');
  const fileDrop = document.getElementById('fileDrop');
  const notify = document.getElementById('notify');
  const notifyMessage = document.getElementById('notifyMessage');
  const notifyClose = document.getElementById('notifyClose');

  let origChart = null;
  let procChart = null;

  const kernelSize = document.getElementById('kernelSize');
  const kernelSizeVal = document.getElementById('kernelSizeVal');
  const sobelK = document.getElementById('sobelK');

  // safe guards: only attach listeners when elements exist
  if (claheClip && claheVal) claheClip.addEventListener('input', () => { claheVal.textContent = claheClip.value; });
  if (kernelSize && kernelSizeVal) kernelSize.addEventListener('input', ()=> { kernelSizeVal.textContent = kernelSize.value; });
  if (sobelK) sobelK.addEventListener('change', ()=>{});

// show/hide params based on technique
const params = document.getElementById('params');
function updateParamsVisibility(){
  const t = techniqueSelect.value;
  const kLabel = kernelSize.parentElement;
  const sLabel = sobelK.parentElement;
  if (['clahe','match','average','median','sobel'].includes(t)) {
    params.style.display = 'block';
    // show/hide kernel vs sobel specific controls
    if (['average','median'].includes(t)) { kLabel.style.display='flex'; sLabel.style.display='none'; }
    else if (t === 'sobel') { kLabel.style.display='none'; sLabel.style.display='flex'; }
    else { kLabel.style.display='none'; sLabel.style.display='none'; }
  } else {
    params.style.display = 'none';
    kLabel.style.display='none'; sLabel.style.display='none';
  }
}
if (techniqueSelect) {
  techniqueSelect.addEventListener('change', updateParamsVisibility);
  updateParamsVisibility();
}

// spinner overlay
const spinner = document.createElement('div');
spinner.className = 'spinner-overlay';
spinner.innerHTML = '<div class="spinner"></div>';
document.body.appendChild(spinner);

function showSpinner(){ spinner.classList.add('active'); }
function hideSpinner(){ spinner.classList.remove('active'); }


// quick backend health check (non-blocking) — try Flask first, then relative
let _backendLastState = null; // 'up' | 'down'
async function checkBackend(){
  const tryUrls = ['http://127.0.0.1:5000/api/health','/api/health'];
  let ok = false;
  for (const u of tryUrls) {
    try {
      const r = await fetch(u, { method: 'GET' });
      if (r.ok) { ok = true; break; }
    } catch(e) {
      // ignore and try next
    }
  }

  const el = document.getElementById('backendStatus');
  const dot = el ? el.querySelector('.status-dot') : null;
  if (ok) {
    if (dot) { dot.classList.remove('status-unknown','status-down'); dot.classList.add('status-up'); }
    if (_backendLastState !== 'up') {
      _backendLastState = 'up';
      // clear previous unreachable notification if present
      hideNotification();
    }
    return true;
  } else {
    if (dot) { dot.classList.remove('status-up','status-unknown'); dot.classList.add('status-down'); }
    if (_backendLastState !== 'down') {
      _backendLastState = 'down';
      showNotification('Backend not reachable. Start Flask: python app.py');
    }
    return false;
  }
}

// poll backend and update indicator periodically
checkBackend();
setInterval(checkBackend, 5000);



// auto-hide timer
let _notifyTimer = null;
function showNotification(msg){
  if(!notify) { alert(msg); return; }
  notifyMessage.textContent = msg;
  notify.style.display = 'block';
  if(_notifyTimer) clearTimeout(_notifyTimer);
  _notifyTimer = setTimeout(()=>{ hideNotification(); _notifyTimer = null; }, 6000);
}

function hideNotification(){ if(notify) { notify.style.display = 'none'; if(_notifyTimer){ clearTimeout(_notifyTimer); _notifyTimer = null; } } }

  notifyClose && notifyClose.addEventListener('click', ()=> hideNotification());

function drawHistogram(canvas, data, color){
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth;
  const h = canvas.height = canvas.clientHeight;
  ctx.clearRect(0,0,w,h);
  if (!data || data.length === 0) return;

  // normalize data
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;

  ctx.beginPath();
  ctx.moveTo(0, h);
  for (let i=0;i<data.length;i++){
    const x = (i / (data.length-1)) * w;
    const y = h - ((data[i]-min)/range) * (h*0.9) - h*0.05;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(w, h);
  ctx.closePath();
  ctx.fillStyle = color || 'rgba(100,150,255,0.12)';
  ctx.fill();

  // stroke
  ctx.beginPath();
  for (let i=0;i<data.length;i++){
    const x = (i / (data.length-1)) * w;
    const y = h - ((data[i]-min)/range) * (h*0.9) - h*0.05;
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.strokeStyle = color || '#60a5fa';
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

if (processBtn) processBtn.addEventListener('click', async () => {
  if (!imageInput.files || imageInput.files.length === 0) { showNotification('Please select an image first'); return; }

  const file = imageInput.files[0];
  const technique = techniqueSelect.value;

  processBtn.disabled = true; processBtn.textContent = 'Processing...'; showSpinner();

  const form = new FormData();
  form.append('image', file);
  form.append('technique', technique);
  form.append('clahe_clip', claheClip.value);
  if (kernelSize) form.append('kernel', kernelSize.value);
  if (sobelK) form.append('sobel_ksize', sobelK.value);
  if (templateInput.files && templateInput.files[0]) form.append('template', templateInput.files[0]);

  try {
    // prefer Flask origin first (when running backend separately), then try relative
    const tryUrls = ['http://127.0.0.1:5000/api/process','/api/process'];
    let res = null;
    for (const u of tryUrls) {
      try {
        res = await fetch(u, { method: 'POST', body: form });
        break;
      } catch (e) {
        // try next
      }
    }
    if (!res) {
      showNotification('Request failed: network or backend unavailable');
      hideSpinner(); processBtn.disabled = false; processBtn.textContent = 'Process Image';
      return;
    }

    // handle non-JSON or empty responses safely
    if (!res.ok) {
      const text = await res.text().catch(()=>null);
      const msg = text ? `Server error: ${text}` : `Server returned ${res.status} ${res.statusText}`;
      showNotification(msg);
      hideSpinner();
      processBtn.disabled = false; processBtn.textContent = 'Process Image';
      return;
    }

    const contentType = res.headers.get('content-type') || '';
    let data = null;
    if (contentType.includes('application/json')) {
      try { data = await res.json(); }
      catch (e) { showNotification('Failed to parse JSON response from server'); hideSpinner(); processBtn.disabled = false; processBtn.textContent = 'Process Image'; return; }
    } else {
      const text = await res.text().catch(()=>null);
      showNotification('Unexpected server response: ' + (text || 'empty')); hideSpinner(); processBtn.disabled = false; processBtn.textContent = 'Process Image'; return;
    }
    if (data.error) { showNotification('Error: ' + data.error); hideSpinner(); processBtn.disabled = false; processBtn.textContent = 'Process Image'; return; }

    // show original preview
    const reader = new FileReader();
    reader.onload = () => { origImg.src = reader.result; };
    reader.readAsDataURL(file);
    // processed image(s) and features
    if (Array.isArray(data.processed_images) && data.processed_images.length > 0) {
      const items = data.processed_images;
      const procList = document.getElementById('procList');
      const procLabel = document.getElementById('procLabel');

      // hide prev/next controls when stacking
      const prevBtn = document.getElementById('prevBtn');
      const nextBtn = document.getElementById('nextBtn');
      if (prevBtn) prevBtn.style.display = 'none';
      if (nextBtn) nextBtn.style.display = 'none';

      // clear list and append each processed image as a stacked block
      if (procList) procList.innerHTML = '';
      items.forEach((it, i) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'proc-item';
        const title = document.createElement('div');
        title.className = 'proc-item-title';
        title.textContent = it.name;
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + it.b64;
        img.alt = it.name;
        img.className = 'proc-item-img';
        wrapper.appendChild(title);
        wrapper.appendChild(img);
        if (procList) procList.appendChild(wrapper);
        // set download for first item
        if (i === 0 && downloadLink) { downloadLink.href = img.src; downloadLink.style.display = 'inline-block'; downloadLink.download = it.name.replace(/\s+/g, '_') + '.png'; }
      });

      // set label to first item's name
      if (procLabel && items[0]) { procLabel.style.display = 'inline-block'; procLabel.textContent = items[0].name; }

    } else {
      // single processed image (legacy)
      procImg.src = 'data:image/png;base64,' + data.processed_image_png_base64;
      if (downloadLink) { downloadLink.href = procImg.src; downloadLink.style.display = 'inline-block'; }
    }

    origFeatures.textContent = JSON.stringify(data.original_features, null, 2);
    procFeatures.textContent = JSON.stringify(data.processed_features, null, 2);
    resultCard.style.display = 'block';

    // histograms
    try{
      const odata = data.original_histogram || [];
      const pdata = data.processed_histogram || [];
      try{
        drawHistogram(origHistCanvas, odata, 'rgba(96,165,250,0.9)');
        drawHistogram(procHistCanvas, pdata, 'rgba(52,211,153,0.9)');
      }catch(e){console.warn('hist draw',e)}
    }catch(e){ console.warn('hist chart', e); }

    window.scrollTo({ top: resultCard.offsetTop - 10, behavior: 'smooth' });

  } catch (err) { showNotification('Request failed: ' + err.message); }
  finally { processBtn.disabled = false; processBtn.textContent = 'Process Image'; hideSpinner(); }
});

// drag & drop behavior
if (fileDrop) {
  fileDrop.addEventListener('dragover', (e) => { e.preventDefault(); fileDrop.classList.add('drag'); });
  fileDrop.addEventListener('dragleave', (e) => { fileDrop.classList.remove('drag'); });
  fileDrop.addEventListener('drop', (e) => { e.preventDefault(); fileDrop.classList.remove('drag'); if (e.dataTransfer.files[0] && imageInput) imageInput.files = e.dataTransfer.files; const label = fileDrop.querySelector('.file-drop-content p'); if (label && imageInput && imageInput.files[0]) label.textContent = imageInput.files[0].name; });
}

if (imageInput) {
  imageInput.addEventListener('change', () => { if (imageInput.files && imageInput.files[0] && fileDrop) { const label = fileDrop.querySelector('.file-drop-content p'); if (label) label.textContent = imageInput.files[0].name; } });
}

// sample button removed — no-op

if (resetBtn) {
  resetBtn.addEventListener('click', ()=>{
    if (imageInput) imageInput.value=''; if (templateInput) templateInput.value=''; if (origImg) origImg.src=''; if (procImg) procImg.src=''; if (origFeatures) origFeatures.textContent=''; if (procFeatures) procFeatures.textContent=''; if (resultCard) resultCard.style.display='none'; if(origChart){origChart.destroy(); origChart=null} if(procChart){procChart.destroy(); procChart=null} const label = fileDrop ? fileDrop.querySelector('.file-drop-content p') : null; if (label) label.textContent='Click or drop image here';
  });
}
});
