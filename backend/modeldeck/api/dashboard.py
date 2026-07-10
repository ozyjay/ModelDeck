# The embedded asset keeps the first dashboard build-free; line wrapping would alter CSS/JS.
# ruff: noqa: E501
DASHBOARD_HTML = """<!doctype html>
<html lang="en-AU">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ModelDeck</title>
  <style>
    :root{color-scheme:dark;--bg:#0c111b;--card:#151d2b;--line:#2d3a50;--text:#eef4ff;--muted:#a9b6ca;--accent:#72dfb5;--warn:#ffc66d}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#17243a,var(--bg) 40%);color:var(--text);font:15px system-ui,sans-serif}
    main{max-width:1180px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px}
    h1{margin:0;font-size:2.2rem}h2{font-size:1rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}p{color:var(--muted)}
    .pill{border:1px solid var(--line);border-radius:99px;padding:7px 12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}
    section,.worker{background:color-mix(in srgb,var(--card) 92%,transparent);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 14px 40px #0004}
    dl{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:0}dt{color:var(--muted)}dd{margin:0;text-align:right}.workers{display:grid;gap:12px}.worker h3{margin:0 0 4px}.worker.ready{border-color:var(--accent)}
    button{border:0;border-radius:8px;padding:9px 13px;margin:8px 6px 0 0;background:var(--accent);color:#071711;font-weight:700;cursor:pointer}button.secondary{background:#34435b;color:var(--text)}
    code{color:var(--accent)}.notice{color:var(--warn)}
  </style>
</head>
<body><main>
  <header><div><h1>ModelDeck</h1><p>Local runtime control and capability routing</p></div><span class="pill" id="gateway">Gateway checking…</span></header>
  <div class="grid">
    <section><h2>Machine</h2><dl id="machine"><dt>Status</dt><dd>Loading…</dd></dl></section>
    <section><h2>Resources</h2><dl id="resources"><dt>Status</dt><dd>Loading…</dd></dl></section>
    <section><h2>Policy</h2><p>One model load at a time. Local files only. No cloud fallback.</p><p class="notice">Mock workers prove lifecycle behaviour without using the GPU.</p></section>
  </div>
  <h2>Workers</h2><div class="workers" id="workers">Loading…</div>
  <h2>Cached model library</h2><section><p id="catalogue">Scanning local Hugging Face cache…</p></section>
</main>
<script>
const esc=s=>String(s??'unknown').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function action(id,op){await fetch(`/api/workers/${id}/${op}`,{method:'POST'});await refreshWorkers()}
async function refreshWorkers(){const rows=await fetch('/api/workers').then(r=>r.json());workers.innerHTML=rows.map(w=>`<article class="worker ${esc(w.state)}"><h3>${esc(w.id)}</h3><p>${esc(w.generation_family)} · <code>${esc(w.state)}</code> · ${esc(w.endpoint)}</p><button onclick="action('${esc(w.id)}','start')">Start</button><button class="secondary" onclick="action('${esc(w.id)}','stop')">Stop</button><button class="secondary" onclick="action('${esc(w.id)}','restart')">Restart</button></article>`).join('')}
async function refresh(){const h=await fetch('/api/hardware').then(r=>r.json());const d=h.detected;machine.innerHTML=`<dt>Configured target</dt><dd>${esc(h.configured.gpu_architecture)} / ROCm ${esc(h.configured.rocm_family)}</dd><dt>Fedora</dt><dd>${esc(d.fedora_release)}</dd><dt>Kernel</dt><dd>${esc(d.kernel)}</dd><dt>Python</dt><dd>${esc(d.python)}</dd>`;resources.innerHTML=`<dt>RAM available</dt><dd>${Math.round(d.memory.available_bytes/2**30)} GiB</dd><dt>Swap used</dt><dd>${Math.round(d.swap.used_bytes/2**20)} MiB</dd><dt>/mnt/work</dt><dd>${d.filesystems[1]?.available?'available':'missing'}</dd><dt>ROCm packages</dt><dd>${d.rocm_packages.length?'detected':'not detected'}</dd>`;const c=await fetch('/api/catalogue').then(r=>r.json());catalogue.textContent=`${c.models.length} cached model repositories found. Cache presence is not treated as runtime compatibility.`;await refreshWorkers();fetch('http://127.0.0.1:8600/v1/health').then(r=>r.ok?gateway.textContent='Gateway ready':0).catch(()=>gateway.textContent='Gateway unavailable')}
refresh();setInterval(refreshWorkers,3000);
</script></body></html>"""
