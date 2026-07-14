const ta = document.getElementById('input');
  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  });

  document.getElementById('sendBtn').addEventListener('click', () => {
    const query = ta.value.trim();
    if(query.length){
      const platform = attachTrigger.dataset.platform || null;
      startIngestion(query, platform);
      ta.value = '';
      ta.style.height = 'auto';
    }
  });

  ta.addEventListener('keydown', (e) => {
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault();
      document.getElementById('sendBtn').click();
    }
  });

  const attachTrigger = document.getElementById('attachTrigger');
const attachMenu = document.getElementById('attachMenu');

attachTrigger.addEventListener('click', (e) => {
  e.stopPropagation();
  attachMenu.classList.toggle('open');
  attachTrigger.classList.toggle('active');
});

const attachIcon = document.getElementById('attachIcon');
const attachLabel = document.getElementById('attachLabel');
const attachCancel = document.getElementById('attachCancel');

// remember the defaults so "cancel" can restore them exactly
const defaultIconHTML = attachIcon.innerHTML;
const defaultLabelText = attachLabel.textContent;
const defaultPlaceholder = ta.placeholder;

function resetAttachSelection(){
  attachIcon.innerHTML = defaultIconHTML;
  attachLabel.textContent = defaultLabelText;
  ta.placeholder = defaultPlaceholder;
  delete attachTrigger.dataset.platform;

  attachTrigger.classList.remove('selected');
  attachTrigger.classList.remove('pop');
}

attachCancel.addEventListener('click', (e) => {
  e.stopPropagation(); // don't let this also trigger the dropdown toggle
  resetAttachSelection();
});


attachMenu.querySelectorAll('.dropdown-item').forEach(item => {
  item.addEventListener('click', () => {
    const choice = item.dataset.value;
    const iconSvg = item.querySelector('svg');
    const title = item.querySelector('.dropdown-item-title').textContent;

    // swap the trigger's icon + label to match the selected platform
    attachIcon.innerHTML = iconSvg.outerHTML;
    attachLabel.textContent = title;
    attachTrigger.dataset.platform = choice;

    attachTrigger.classList.add('selected');

    // retrigger the pop animation even on repeated selections
    attachTrigger.classList.remove('pop');
    void attachTrigger.offsetWidth; // force reflow so the animation restarts
    attachTrigger.classList.add('pop');

    console.log('Attach option selected:', choice);
    attachMenu.classList.remove('open');
    attachTrigger.classList.remove('active');
  });
});

document.addEventListener('click', (e) => {
  if(!attachMenu.contains(e.target) && !attachTrigger.contains(e.target)){
    attachMenu.classList.remove('open');
    attachTrigger.classList.remove('active');
  }
});

const pipelineEl   = document.getElementById('pipeline');
const stepsEl      = document.getElementById('steps');
const pipelineSrc  = document.getElementById('pipelineSource');
const analyticsEl  = document.getElementById('analytics');
const analyticsGrid = document.getElementById('analyticsGrid');
const analyticsClose = document.getElementById('analyticsClose');
const hintEl = document.getElementById('hint');

const CHECK_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>';

// step recipes per source type. Each has a label, a sub-label (shown while
// pending/active), and a rough duration range in ms to simulate.
const STEP_LIBRARY = {
  youtube: [
    { id: 'meta',      title: 'Fetching video metadata',        sub: 'Title, description, upload date', range: [500, 900] },
    { id: 'channel',   title: 'Fetching channel data',           sub: 'Subscriber count, channel age', range: [500, 900] },
    { id: 'engagement',title: 'Fetching engagement metrics',     sub: 'Views and likes', range: [400, 800] },
    { id: 'comments',  title: 'Collecting comments',              sub: 'Paginating top-level comments', range: [900, 1800] },
    { id: 'audio',     title: 'Downloading audio track',          sub: 'Extracting for ASR fallback', range: [1200, 2400] },
    { id: 'transcript',title: 'Checking for a Bangla transcript', sub: 'Filtering out auto-generated (ASR) caption tracks', range: [700, 1300] },
    { id: 'compile',   title: 'Compiling channel & content insights', sub: 'Merging all sources', range: [500, 900] },
  ],
  social: [ // Facebook / Instagram links
    { id: 'meta',      title: 'Fetching post metadata',    sub: 'Caption, media type, post date', range: [500, 900] },
    { id: 'engagement',title: 'Fetching engagement metrics', sub: 'Reactions and shares', range: [400, 800] },
    { id: 'comments',  title: 'Collecting comments',        sub: 'Paginating top-level comments', range: [900, 1600] },
    { id: 'compile',   title: 'Compiling verification dataset', sub: 'Merging all sources', range: [500, 900] },
  ],
  text: [ // raw pasted text / no platform selected
    { id: 'parse',     title: 'Parsing submitted text',      sub: 'Language detection, claim extraction', range: [400, 800] },
    { id: 'crossref',  title: 'Cross-referencing known sources', sub: 'Matching against verified corpus', range: [900, 1600] },
    { id: 'compile',   title: 'Compiling verification dataset', sub: 'Merging all sources', range: [500, 900] },
  ],
};

function sourceKindFor(platform){
  if(platform === 'upload') return 'youtube';
  if(platform === 'photos' || platform === 'drive') return 'social';
  return 'text';
}

const BACKEND_URL = 'http://localhost:8000'; // point this at wherever you run the FastAPI backend

function rand(range){
  return Math.round(range[0] + Math.random() * (range[1] - range[0]));
}

function setStepState(li, state){
  li.dataset.state = state;
  const icon = li.querySelector('.step-icon');
  icon.innerHTML = state === 'done' ? CHECK_SVG : '';
}

function renderSteps(steps){
  stepsEl.innerHTML = '';
  const byId = {};
  steps.forEach(step => {
    const li = document.createElement('li');
    li.className = 'step';
    li.dataset.state = 'pending';
    li.innerHTML = `
      <span class="step-icon"></span>
      <div class="step-text">
        <div class="step-title">${step.title}</div>
        <div class="step-sub">${step.sub}</div>
      </div>
    `;
    stepsEl.appendChild(li);
    byId[step.id] = li;
  });
  return byId;
}

function startIngestion(query, platform){
  analyticsEl.classList.remove('open');
  hintEl.style.display = 'none';
  document.body.classList.add('session-active');

  const kind = sourceKindFor(platform);
  pipelineSrc.textContent = query.length > 60 ? query.slice(0, 60) + '\u2026' : query;
  pipelineEl.classList.add('open');

  if(kind === 'youtube'){
    runRealYoutubeIngestion(query);
  } else {
    runSimulatedIngestion(kind, query);
  }
}

// ---- real backend path (YouTube only) ----
async function runRealYoutubeIngestion(url){
  const byId = renderSteps(STEP_LIBRARY.youtube);

  let res;
  try{
    res = await fetch(`${BACKEND_URL}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    if(!res.ok) throw new Error(`Backend responded ${res.status}`);
  } catch(err){
    console.warn('Ingestion backend unreachable, falling back to simulation:', err);
    pipelineSrc.textContent += '  ·  backend unreachable, showing simulated demo';
    runSimulatedIngestion('youtube', url, byId);
    return;
  }

  const { job_id } = await res.json();
  const source = new EventSource(`${BACKEND_URL}/jobs/${job_id}/stream`);

  source.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    const li = byId[msg.step];
    if(!li) return;

    if(msg.status === 'error'){
      setStepState(li, 'error');
      li.querySelector('.step-sub').textContent = msg.message || 'Something went wrong on this step.';
      source.close();
      return;
    }

    setStepState(li, msg.status); // 'active' | 'done'

    if(msg.step === 'compile' && msg.status === 'done' && msg.result){
      source.close();
      buildAnalyticsFromResult(msg.result);
    }
  };

  source.onerror = () => {
    source.close();
  };
}

// ---- simulated path (Facebook/Instagram, raw text, or backend-unreachable fallback) ----
function runSimulatedIngestion(kind, query, existingById){
  const steps = STEP_LIBRARY[kind];
  const byId = existingById || renderSteps(steps);
  const ordered = steps.map(s => byId[s.id]);

  let i = 0;
  function runStep(){
    if(i > 0) setStepState(ordered[i - 1], 'done');
    if(i >= ordered.length){
      setTimeout(() => buildAnalyticsMock(kind, query), 350);
      return;
    }
    setStepState(ordered[i], 'active');
    setTimeout(runStep, rand(steps[i].range));
    i++;
  }
  runStep();
}

function buildAnalyticsFromResult(result){
  analyticsGrid.innerHTML = '';
  document.body.classList.add('results-active');
  const rows = [
    ['Title', result.title || '—'],
    ['Channel', result.channel || '—'],
    ['Subscribers', result.subscribers ?? '—'],
    ['Views', result.views ?? '—'],
    ['Likes', result.likes ?? '—'],
    ['Comments collected', result.comments_collected ?? '—'],
    ['Transcript source', result.transcript_source === 'manual' ? 'Manual upload' : 'None (auto-generated excluded)', result.transcript_source !== 'manual' ? 'flag' : null],
  ];
  rows.forEach(([label, value, flag]) => {
    const div = document.createElement('div');
    div.className = 'stat' + (flag ? ' flag' : '');
    div.innerHTML = `<div class="stat-label">${label}</div><div class="stat-value">${value}</div>`;
    analyticsGrid.appendChild(div);
  });
  analyticsEl.classList.add('open');
}

function buildAnalyticsMock(kind, query){
  analyticsGrid.innerHTML = '';
  document.body.classList.add('results-active');

  // Mock figures — only used for platforms/paths with no real backend
  // behind them yet (Facebook, Instagram, raw text) or if the backend
  // was unreachable.
  const statSets = {
    youtube: [
      ['Title', 'সংগৃহীত ভিডিও শিরোনাম (placeholder)'],
      ['Channel', 'Unverified channel name'],
      ['Subscribers', '128,400'],
      ['Views', '46,209'],
      ['Likes', '2,113'],
      ['Comments collected', '318'],
      ['Audio duration', '6m 42s'],
      ['Transcript', 'None found (auto-generated caption skipped)', 'flag'],
    ],
    social: [
      ['Caption language', 'Bangla'],
      ['Reactions', '1,204'],
      ['Shares', '87'],
      ['Comments collected', '96'],
      ['Media type', 'Video post'],
    ],
    text: [
      ['Detected language', 'Bangla'],
      ['Claim segments found', '3'],
      ['Closest known match', 'No strong match', 'flag'],
    ],
  };

  statSets[kind].forEach(([label, value, flag]) => {
    const div = document.createElement('div');
    div.className = 'stat' + (flag ? ' flag' : '');
    div.innerHTML = `<div class="stat-label">${label}</div><div class="stat-value">${value}</div>`;
    analyticsGrid.appendChild(div);
  });

  analyticsEl.classList.add('open');
}

analyticsClose.addEventListener('click', () => {
  analyticsEl.classList.remove('open');
  pipelineEl.classList.remove('open');
  hintEl.style.display = '';
  document.body.classList.remove('results-active');
  document.body.classList.remove('session-active');
  resetAttachSelection();
});