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
const sampleBtn = document.getElementById('sampleBtn');
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

claheClip.addEventListener('input', () => { claheVal.textContent = claheClip.value; });
kernelSize.addEventListener('input', ()=> { kernelSizeVal.textContent = kernelSize.value; });
sobelK.addEventListener('change', ()=>{});

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
techniqueSelect.addEventListener('change', updateParamsVisibility);
updateParamsVisibility();

// spinner overlay
const spinner = document.createElement('div');
spinner.className = 'spinner-overlay';
spinner.innerHTML = '<div class="spinner"></div>';
document.body.appendChild(spinner);

function showSpinner(){ spinner.classList.add('active'); }
function hideSpinner(){ spinner.classList.remove('active'); }



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

processBtn.addEventListener('click', async () => {
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
    const res = await fetch('/api/process', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) { showNotification('Error: ' + data.error); hideSpinner(); processBtn.disabled = false; processBtn.textContent = 'Process Image'; return; }

    // show original preview
    const reader = new FileReader();
    reader.onload = () => { origImg.src = reader.result; };
    reader.readAsDataURL(file);

    // processed image and features
    procImg.src = 'data:image/png;base64,' + data.processed_image_png_base64;
    origFeatures.textContent = JSON.stringify(data.original_features, null, 2);
    procFeatures.textContent = JSON.stringify(data.processed_features, null, 2);
    downloadLink.href = procImg.src; downloadLink.style.display = 'inline-block'; resultCard.style.display = 'block';

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
fileDrop.addEventListener('dragover', (e) => { e.preventDefault(); fileDrop.classList.add('drag'); });
fileDrop.addEventListener('dragleave', (e) => { fileDrop.classList.remove('drag'); });
fileDrop.addEventListener('drop', (e) => { e.preventDefault(); fileDrop.classList.remove('drag'); if (e.dataTransfer.files[0]) imageInput.files = e.dataTransfer.files; const label = fileDrop.querySelector('.file-drop-content p'); label.textContent = imageInput.files[0].name; });

imageInput.addEventListener('change', () => { if (imageInput.files && imageInput.files[0]) { const label = fileDrop.querySelector('.file-drop-content p'); label.textContent = imageInput.files[0].name; } });

sampleBtn.addEventListener('click', ()=>{
  fetch('/img/xray.png').then(r=>{ if(!r.ok) throw new Error('not found'); return r.blob(); }).then(blob=>{
    const f = new File([blob],'xray.png',{type:blob.type}); const dt = new DataTransfer(); dt.items.add(f); imageInput.files = dt.files; const label = fileDrop.querySelector('.file-drop-content p'); label.textContent = f.name;
    }).catch(()=>showNotification('Sample image not available in /img'));
});

// Hide sample button if file not present to avoid 404 spam
fetch('/img/xray.png', { method: 'GET' }).then(r => {
  if (!r.ok) sampleBtn.style.display = 'none';
}).catch(()=>{ sampleBtn.style.display = 'none'; });

resetBtn.addEventListener('click', ()=>{
  imageInput.value=''; templateInput.value=''; origImg.src=''; procImg.src=''; origFeatures.textContent=''; procFeatures.textContent=''; resultCard.style.display='none'; if(origChart){origChart.destroy(); origChart=null} if(procChart){procChart.destroy(); procChart=null} const label = fileDrop.querySelector('.file-drop-content p'); label.textContent='Click or drop image here';
});
