const $ = (sel) => document.querySelector(sel);

const form = $("#predict-form");
const predictBtn = $("#predict-btn");
const retrainBtn = $("#retrain-btn");
const formError = $("#form-error");
const resultsCard = $("#results");
const neutralCheck = $("#neutral-check");
const hostWrap = $("#host-wrap");
const teamList = $("#team-list");
const modelStatus = $("#model-status");
const retrainStatusEl = $("#retrain-status");

let retrainPoll = null;

function pct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

function showError(msg) {
  if (!msg) {
    formError.classList.add("hidden");
    formError.textContent = "";
    return;
  }
  formError.textContent = msg;
  formError.classList.remove("hidden");
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = body.detail ?? body.message ?? res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

async function loadTeams() {
  try {
    const { teams } = await api("/teams");
    teamList.innerHTML = teams.map((t) => `<option value="${escapeAttr(t)}">`).join("");
  } catch (e) {
    console.warn("Could not load team list:", e);
  }
}

function escapeAttr(s) {
  return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

function setDefaultDate() {
  const input = form.elements.date;
  if (!input.value) {
    const d = new Date();
    input.value = d.toISOString().slice(0, 10);
  }
}

neutralCheck.addEventListener("change", () => {
  hostWrap.classList.toggle("hidden", !neutralCheck.checked);
});

function renderResults(data, meta) {
  resultsCard.classList.remove("hidden");
  $("#fixture-line").textContent = `${meta.home_team} vs ${meta.away_team} · ${meta.date} · ${meta.tournament}`;

  const rows = [
    { key: "home", label: "Home", prob: data.home_win_prob, cls: "home" },
    { key: "draw", label: "Draw", prob: data.draw_prob, cls: "draw" },
    { key: "away", label: "Away", prob: data.away_win_prob, cls: "away" },
  ];

  $("#outcomes").innerHTML = rows
    .map(
      (r) => `
    <div class="outcome-row">
      <span>${r.label}</span>
      <div class="bar-track"><div class="bar-fill ${r.cls}" style="width:${r.prob * 100}%"></div></div>
      <span class="pct">${pct(r.prob)}</span>
    </div>`
    )
    .join("");

  $("#lambda-home").textContent = data.lambda_home.toFixed(2);
  $("#lambda-away").textContent = data.lambda_away.toFixed(2);
  $("#over25").textContent = pct(data.over_25_prob);
  $("#btts").textContent = pct(data.btts_prob);

  const lines = data.top_scorelines || [];
  $("#scorelines").innerHTML = lines
    .map((s) => `<li><span>${s.score}</span><span>${pct(s.prob)}</span></li>`)
    .join("");
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  showError("");
  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting…";

  const fd = new FormData(form);
  const payload = {
    home_team: fd.get("home_team").trim(),
    away_team: fd.get("away_team").trim(),
    date: fd.get("date"),
    tournament: fd.get("tournament"),
    neutral: fd.get("neutral") === "on",
    host_country: fd.get("host_country")?.trim() || null,
    prediction_mode: fd.get("prediction_mode"),
  };

  try {
    const data = await api("/predict", { method: "POST", body: JSON.stringify(payload) });
    renderResults(data, payload);
  } catch (e) {
    showError(e.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
    refreshModelStatus();
  }
});

function updateRetrainUI(status) {
  const running = status.running;
  retrainBtn.disabled = running;
  predictBtn.disabled = running;

  if (running) {
    retrainStatusEl.textContent = status.message || `Phase: ${status.phase}…`;
    retrainStatusEl.classList.remove("muted");
  } else if (status.error) {
    retrainStatusEl.textContent = `Error: ${status.error}`;
    retrainStatusEl.style.color = "var(--danger)";
  } else if (status.last_completed_at) {
    const rows = status.last_result?.processed_rows;
    const acc = status.last_result?.splits?.test?.accuracy;
    let msg = `Last retrain: ${new Date(status.last_completed_at).toLocaleString()}`;
    if (rows != null) msg += ` · ${rows.toLocaleString()} matches`;
    if (acc != null) msg += ` · test accuracy ${(acc * 100).toFixed(1)}%`;
    retrainStatusEl.textContent = msg;
    retrainStatusEl.style.color = "";
    retrainStatusEl.classList.add("muted");
  } else {
    retrainStatusEl.textContent = "";
  }
}

async function refreshModelStatus() {
  try {
    const health = await api("/health");
    if (health.artifacts_ready) {
      const info = await api("/model-info");
      modelStatus.textContent = `Ready · ${info.tabular_backend} · ${info.feature_count} features`;
    } else {
      modelStatus.textContent = "No trained models yet. Run retrain below.";
    }
  } catch {
    modelStatus.textContent = "Could not reach API.";
  }
}

async function pollRetrain() {
  try {
    const status = await api("/retrain/status");
    updateRetrainUI(status);
    if (status.running) {
      retrainPoll = setTimeout(pollRetrain, 1500);
    } else {
      retrainPoll = null;
      await refreshModelStatus();
    }
  } catch (e) {
    retrainStatusEl.textContent = e.message;
  }
}

retrainBtn.addEventListener("click", async () => {
  if (!confirm("Retrain rebuilds the dataset and retrains all models. This can take several minutes. Continue?")) {
    return;
  }
  retrainBtn.disabled = true;
  try {
    await api("/retrain", { method: "POST" });
    pollRetrain();
  } catch (e) {
    retrainStatusEl.textContent = e.message;
    retrainBtn.disabled = false;
  }
});

async function init() {
  setDefaultDate();
  await loadTeams();
  await refreshModelStatus();
  const status = await api("/retrain/status");
  updateRetrainUI(status);
  if (status.running) pollRetrain();
}

init();
