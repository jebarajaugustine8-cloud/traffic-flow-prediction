/* ==========================================================
   Traffic Flow Prediction — NEON NIGHT DRIVE frontend
   - AJAX prediction with animated signal + counter + forecast
   - Route planner: up to 3 alternative routes from OSRM
     (synthetic curved fallbacks guarantee 3 are always shown)
   - THE COMMUTE RACE: 5 vehicles animate along the ACTUAL
     map route, speeds proportional to predicted ETAs
   - 3D interactions: tilt cards, scroll reveal, cursor glow
   - Weather effects: live rain / fog overlays
   ========================================================== */

const btn = document.getElementById("predictBtn");
const errorMsg = document.getElementById("errorMsg");

const lamps = {
  red: document.getElementById("lampRed"),
  amber: document.getElementById("lampAmber"),
  green: document.getElementById("lampGreen"),
};

const SIGNAL_COLORS = { green: "#37d67a", amber: "#ffb020", red: "#ff5a5a" };
const ALT_COLORS = ["#00e5ff", "#ff2d95", "#9aa3c7"]; // route 1/2/3 accents

/* ---------------- single-point prediction ---------------- */

function setSignal(color) {
  Object.values(lamps).forEach((l) => l.classList.remove("on"));
  if (lamps[color]) lamps[color].classList.add("on");
}

function animateCount(el, target, duration = 700) {
  const start = performance.now();
  function frame(now) {
    const p = Math.min((now - start) / duration, 1);
    el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function hourLabel(h) {
  const ampm = h < 12 ? "AM" : "PM";
  const hh = h % 12 === 0 ? 12 : h % 12;
  return `${hh} ${ampm}`;
}

function drawForecast(forecast, selectedHour, bestHour, worstHour) {
  const wrap = document.getElementById("forecastBars");
  wrap.innerHTML = "";
  const max = Math.max(...forecast, 1);
  forecast.forEach((v, h) => {
    const bar = document.createElement("div");
    bar.className = "bar";
    if (h === worstHour) bar.classList.add("worst");
    if (h === bestHour) bar.classList.add("best");
    if (h === selectedHour) bar.classList.add("selected");
    const tip = document.createElement("span");
    tip.className = "tip";
    tip.textContent = `${hourLabel(h)} · ${v} veh/hr`;
    bar.appendChild(tip);
    wrap.appendChild(bar);
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        bar.style.height = `${Math.max((v / max) * 100, 3)}%`;
      })
    );
  });
  document.getElementById("forecastHint").textContent =
    `Best time to travel: ${hourLabel(bestHour)} (lightest traffic). ` +
    `Busiest hour: ${hourLabel(worstHour)}. Your selected hour is shown in amber.`;
  document.getElementById("forecast").classList.remove("hidden");
}

async function predict() {
  errorMsg.textContent = "";
  btn.disabled = true;
  btn.textContent = "Predicting…";

  const payload = {
    hour: document.getElementById("hour").value,
    day: document.getElementById("day").value,
    weather: document.getElementById("weather").value,
    holiday: document.getElementById("holiday").value,
    zone: document.getElementById("zone").value,
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Prediction request failed");
    const data = await res.json();

    document.getElementById("resultPlaceholder").classList.add("hidden");
    document.getElementById("resultData").classList.remove("hidden");

    setSignal(data.signal);
    const level = document.getElementById("levelLabel");
    level.textContent = data.level;
    level.className = `level-label level-${data.signal}`;

    animateCount(document.getElementById("trafficValue"), data.traffic);
    document.getElementById("congestionValue").textContent = data.congestion;
    document.getElementById("meterFill").style.width = `${data.congestion}%`;
    document.getElementById("delayValue").textContent = data.delay;
    document.getElementById("adviceValue").textContent = data.advice;

    drawForecast(data.forecast, parseInt(payload.hour, 10), data.best_hour, data.worst_hour);
    renderExplanation(data.explanation);
  } catch (err) {
    errorMsg.textContent = "Could not get a prediction. Is the server running?";
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict traffic";
  }
}

btn.addEventListener("click", predict);
document.getElementById("hour").addEventListener("keydown", (e) => {
  if (e.key === "Enter") predict();
});

/* ==========================================================
   MAP — alternative routes + on-map commute race
   ========================================================== */

const routeBtn = document.getElementById("routeBtn");
let map = null;
let routeLines = [];     // L.polyline per alternative
let routes = [];         // [{coords:[[lat,lng]...], distKm}]
let activeRoute = 0;
let endMarkers = null;
let racers = [];         // animated vehicle markers
let raceTimer = null;
let lastData = null;

function initMap() {
  map = L.map("map").setView([13.02, 80.2], 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "© OpenStreetMap contributors",
  }).addTo(map);
}

/* --- geometry helpers --- */

function haversineKm(a, b) {
  const R = 6371, rad = Math.PI / 180;
  const dLat = (b[0] - a[0]) * rad, dLng = (b[1] - a[1]) * rad;
  const s = Math.sin(dLat / 2) ** 2 +
    Math.cos(a[0] * rad) * Math.cos(b[0] * rad) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

function pathLengthKm(coords) {
  let d = 0;
  for (let i = 1; i < coords.length; i++) d += haversineKm(coords[i - 1], coords[i]);
  return d;
}

/* Synthetic curved route between two points (quadratic bezier bowed
   sideways) — used so we ALWAYS have 3 alternatives even when the
   free OSRM server returns fewer. */
function syntheticRoute(o, d, bow) {
  const midLat = (o.lat + d.lat) / 2, midLng = (o.lng + d.lng) / 2;
  // perpendicular offset
  const dx = d.lng - o.lng, dy = d.lat - o.lat;
  const len = Math.sqrt(dx * dx + dy * dy) || 1e-6;
  const ctrl = [midLat + (-dx / len) * bow, midLng + (dy / len) * bow];
  const pts = [];
  for (let t = 0; t <= 1.0001; t += 0.04) {
    const lat = (1 - t) ** 2 * o.lat + 2 * (1 - t) * t * ctrl[0] + t * t * d.lat;
    const lng = (1 - t) ** 2 * o.lng + 2 * (1 - t) * t * ctrl[1] + t * t * d.lng;
    pts.push([lat, lng]);
  }
  return pts;
}

/* Ask OSRM for one route (optionally passing THROUGH a via point).
   Returns {coords, distKm} following REAL roads, or null on failure. */
async function osrmRoute(points) {
  try {
    const path = points.map((p) => `${p.lng},${p.lat}`).join(";");
    const url = `https://router.project-osrm.org/route/v1/driving/${path}` +
      `?overview=full&geometries=geojson`;
    const res = await fetch(url);
    if (!res.ok) return null;
    const json = await res.json();
    if (!json.routes || !json.routes.length) return null;
    const r = json.routes[0];
    return {
      coords: r.geometry.coordinates.map((c) => [c[1], c[0]]),
      distKm: r.distance / 1000,
    };
  } catch {
    return null;
  }
}

/* A detour point placed sideways from the midpoint of the trip.
   Routing THROUGH this point forces OSRM onto different real roads. */
function viaPoint(o, d, offset) {
  const midLat = (o.lat + d.lat) / 2, midLng = (o.lng + d.lng) / 2;
  const dx = d.lng - o.lng, dy = d.lat - o.lat;
  const len = Math.sqrt(dx * dx + dy * dy) || 1e-6;
  // scale the sideways push with trip length (min ~1.2 km, max ~4 km)
  const push = Math.min(Math.max(len * 0.35, 0.012), 0.04) * Math.sign(offset);
  return { lat: midLat + (-dx / len) * push, lng: midLng + (dy / len) * push };
}

/* Two routes are "the same road" if their lengths are within 8% */
function isDuplicate(r, list) {
  return list.some((x) => Math.abs(x.distKm - r.distKm) / x.distKm < 0.08);
}

async function fetchRoutes(o, d) {
  const found = [];

  // 1. The direct route + OSRM's own alternatives (all real roads)
  try {
    const url = `https://router.project-osrm.org/route/v1/driving/` +
      `${o.lng},${o.lat};${d.lng},${d.lat}` +
      `?alternatives=3&overview=full&geometries=geojson`;
    const res = await fetch(url);
    if (res.ok) {
      const json = await res.json();
      (json.routes || []).slice(0, 3).forEach((r) => {
        const route = {
          coords: r.geometry.coordinates.map((c) => [c[1], c[0]]),
          distKm: r.distance / 1000,
          real: true,
        };
        if (!isDuplicate(route, found)) found.push(route);
      });
    }
  } catch { /* server unreachable — handled below */ }

  // 2. Not enough? Force REAL detour routes through side via-points.
  const offsets = [1, -1, 1.8, -1.8];
  for (const off of offsets) {
    if (found.length >= 3) break;
    const via = viaPoint(o, d, off);
    const r = await osrmRoute([o, via, d]);
    if (r && !isDuplicate(r, found)) {
      r.real = true;
      found.push(r);
    }
  }

  // 3. Absolute last resort (no internet at all): approximate curves,
  //    clearly labelled so nobody mistakes them for real roads.
  const bows = [0.0, 0.035, -0.03];
  let bi = 0;
  while (found.length < 3 && bi < bows.length) {
    const coords = bows[bi] === 0
      ? [[o.lat, o.lng], [d.lat, d.lng]]
      : syntheticRoute(o, d, bows[bi]);
    found.push({ coords, distKm: pathLengthKm(coords) * 1.35, real: false });
    bi++;
  }
  return found.slice(0, 3);
}

/* --- drawing + selecting alternatives --- */

function styleRoutes(signal) {
  routeLines.forEach((line, i) => {
    if (i === activeRoute) {
      line.setStyle({ color: SIGNAL_COLORS[signal], weight: 6, opacity: 0.95, dashArray: null });
      line.bringToFront();
    } else {
      line.setStyle({ color: ALT_COLORS[i], weight: 3, opacity: 0.45, dashArray: "6 10" });
    }
  });
}

function renderChips(signal) {
  const wrap = document.getElementById("routeChips");
  wrap.innerHTML = "";
  routes.forEach((r, i) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip" + (i === activeRoute ? " active" : "");
    const carEta = etaForDistance(r.distKm, "car", signal);
    const tag = i === 0 ? "Main road" : `Alternative ${i}`;
    const approx = r.real === false ? " (approximate path)" : "";
    chip.textContent = `${tag} · ${r.distKm.toFixed(1)} km · about ${carEta} min by car${approx}`;
    chip.addEventListener("click", () => {
      activeRoute = i;
      styleRoutes(signal);
      renderChips(signal);
      map.fitBounds(routeLines[i].getBounds(), { padding: [30, 30] });
      startMapRace(); // re-race on the newly chosen road
    });
    wrap.appendChild(chip);
  });
  wrap.classList.remove("hidden");
}

/* --- per-mode ETA physics (mirrors the backend) --- */

const MODE_SPEEDS = {
  car:   { green: 38, amber: 24, red: 14, wait: 0 },
  bike:  { green: 42, amber: 32, red: 22, wait: 0 },
  bus:   { green: 25, amber: 17, red: 10, wait: 8 },
  train: { green: 34, amber: 34, red: 34, wait: 12 },
  walk:  { green: 5,  amber: 5,  red: 5,  wait: 0 },
};

function etaForDistance(distKm, modeKey, signal) {
  const m = MODE_SPEEDS[modeKey];
  return Math.round((distKm / m[signal]) * 60 + m.wait);
}

/* --- THE RACE, on the actual route geometry --- */

function clearRacers() {
  racers.forEach((r) => map.removeLayer(r.marker));
  racers = [];
  if (raceTimer) cancelAnimationFrame(raceTimer);
}

function pointAlong(coords, cum, totalKm, frac) {
  const target = totalKm * frac;
  let i = 1;
  while (i < cum.length && cum[i] < target) i++;
  if (i >= cum.length) return coords[coords.length - 1];
  const segStart = cum[i - 1], segLen = cum[i] - segStart || 1e-9;
  const t = (target - segStart) / segLen;
  const a = coords[i - 1], b = coords[i];
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

function startMapRace() {
  if (!lastData || !routes.length) return;
  clearRacers();

  const route = routes[activeRoute];
  const coords = route.coords;
  // cumulative distance table for smooth interpolation
  const cum = [0];
  for (let i = 1; i < coords.length; i++) {
    cum.push(cum[i - 1] + haversineKm(coords[i - 1], coords[i]));
  }
  const totalKm = cum[cum.length - 1];

  // real ETAs on THIS route's distance
  const signal = lastData.signal;
  const modes = lastData.modes.map((m) => ({
    ...m,
    eta: etaForDistance(route.distKm, m.key, signal),
  }));
  const fastest = Math.min(...modes.map((m) => m.eta));
  const sorted = [...modes].sort((a, b) => a.eta - b.eta);

  // fastest crosses in 4s, others proportional (cap 12s)
  modes.forEach((m) => {
    const durMs = Math.min((m.eta / fastest) * 4000, 12000);
    const icon = L.divIcon({ className: "", html: `<span class="map-racer">${m.icon}</span>`, iconSize: [28, 28], iconAnchor: [14, 14] });
    const marker = L.marker(coords[0], { icon, zIndexOffset: 1000 }).addTo(map);
    marker.bindTooltip(`${m.name} — ${m.eta} min`, { direction: "top", offset: [0, -12] });
    racers.push({ marker, durMs, start: null, done: false, mode: m });
  });

  const t0 = performance.now();
  function frame(now) {
    let running = false;
    racers.forEach((r) => {
      const p = Math.min((now - t0) / r.durMs, 1);
      r.marker.setLatLng(pointAlong(coords, cum, totalKm, p));
      if (p < 1) running = true;
      else if (!r.done) {
        r.done = true;
        const el = r.marker.getElement()?.querySelector(".map-racer");
        if (el && r.mode.key === sorted[0].key) {
          el.classList.add("winner");
          r.marker.bindPopup(`🥇 <b>${r.mode.name}</b> wins — ${r.mode.eta} min!`).openPopup();
        }
      }
    });
    if (running) raceTimer = requestAnimationFrame(frame);
  }
  raceTimer = requestAnimationFrame(frame);

  const winner = sorted[0];
  const roadName = activeRoute === 0 ? "the main road" : `alternative road ${activeRoute}`;
  document.getElementById("raceCaption").textContent =
    `On ${roadName} (${route.distKm.toFixed(1)} km), the fastest way to travel is ` +
    `${winner.icon} ${winner.name} — about ${winner.eta} minutes. ` +
    `Each vehicle on the map moves at a speed matching its predicted travel time. ` +
    `Tip: click any road line or button below the map to compare routes.`;
  document.getElementById("raceTrack").classList.remove("hidden");
}

document.getElementById("replayBtn").addEventListener("click", startMapRace);

/* --- mode cards --- */

function renderModeCards(modes, fastestKey) {
  const wrap = document.getElementById("modeCards");
  wrap.innerHTML = "";
  modes.forEach((m, i) => {
    const card = document.createElement("div");
    card.className = "mode-card" + (m.key === fastestKey ? " fastest" : "");
    card.style.animationDelay = `${i * 0.12}s`;
    card.innerHTML =
      `<span class="mode-icon">${m.icon}</span>` +
      `<div class="mode-name">${m.name}</div>` +
      `<div class="mode-eta">${m.eta_minutes}<small> min</small></div>` +
      `<div class="mode-note">${m.note}</div>`;
    wrap.appendChild(card);
  });
  wrap.classList.remove("hidden");
}

/* --- the route prediction flow --- */

async function predictRoute() {
  const errEl = document.getElementById("routeError");
  errEl.textContent = "";

  const origin = document.getElementById("origin").value;
  const destination = document.getElementById("destination").value;
  if (origin === destination) {
    errEl.textContent = "Origin and destination must be different.";
    return;
  }

  routeBtn.disabled = true;
  routeBtn.textContent = "Predicting…";

  const payload = {
    origin, destination,
    hour: document.getElementById("hour").value,
    day: document.getElementById("day").value,
    weather: document.getElementById("weather").value,
    holiday: document.getElementById("holiday").value,
  };

  try {
    const res = await fetch("/api/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Route request failed");
    const data = await res.json();
    lastData = data;

    // result card
    document.getElementById("routeResult").classList.remove("hidden");
    document.getElementById("routeDistance").textContent = data.distance_km;
    document.getElementById("routeEta").textContent = data.eta_minutes;
    document.getElementById("routeOriginT").textContent = data.origin.traffic;
    document.getElementById("routeDestT").textContent = data.destination.traffic;
    const levelEl = document.getElementById("routeLevel");
    levelEl.textContent = data.level;
    levelEl.style.color = SIGNAL_COLORS[data.signal];
    document.getElementById("routeAdvice").textContent =
      `${data.advice} Best departure time today: ${hourLabel(data.best_hour)} (~${data.best_eta_minutes} min).`;

    renderModeCards(data.modes, data.fastest_mode);

    // map: clear previous, fetch up to 3 alternatives, draw all
    if (!map) initMap();
    clearRacers();
    routeLines.forEach((l) => map.removeLayer(l));
    if (endMarkers) map.removeLayer(endMarkers);
    routeLines = [];

    routes = await fetchRoutes(data.origin, data.destination);
    activeRoute = 0;

    routes.forEach((r, i) => {
      const line = L.polyline(r.coords).addTo(map);
      line.on("click", () => {
        activeRoute = i;
        styleRoutes(data.signal);
        renderChips(data.signal);
        startMapRace();
      });
      routeLines.push(line);
    });
    styleRoutes(data.signal);
    renderChips(data.signal);

    const oMark = L.marker([data.origin.lat, data.origin.lng])
      .bindPopup(`<b>${data.origin.name}</b><br>${data.origin.traffic} veh/hr`);
    const dMark = L.marker([data.destination.lat, data.destination.lng])
      .bindPopup(`<b>${data.destination.name}</b><br>${data.destination.traffic} veh/hr`);
    endMarkers = L.layerGroup([oMark, dMark]).addTo(map);

    map.fitBounds(routeLines[0].getBounds(), { padding: [40, 40] });

    // lights, camera… race!
    setTimeout(startMapRace, 500);
  } catch (err) {
    errEl.textContent = "Could not predict the route. Is the server running?";
    console.error(err);
  } finally {
    routeBtn.disabled = false;
    routeBtn.textContent = "Predict route";
  }
}

routeBtn.addEventListener("click", predictRoute);
document.addEventListener("DOMContentLoaded", initMap);

/* ==========================================================
   WEATHER EFFECTS
   ========================================================== */

function buildRain() {
  const layer = document.getElementById("rainLayer");
  if (layer.childElementCount) return;
  for (let i = 0; i < 70; i++) {
    const d = document.createElement("span");
    d.className = "drop";
    d.style.left = `${Math.random() * 100}%`;
    d.style.animationDuration = `${0.6 + Math.random() * 0.7}s`;
    d.style.animationDelay = `${Math.random() * 1.5}s`;
    layer.appendChild(d);
  }
}

function applyWeatherEffect() {
  const w = document.getElementById("weather").value;
  document.getElementById("rainLayer").classList.toggle("active", w === "1");
  document.getElementById("fogLayer").classList.toggle("active", w === "2");
  if (w === "1") buildRain();
}
document.getElementById("weather").addEventListener("change", applyWeatherEffect);

/* ==========================================================
   3D INTERACTIONS — tilt, scroll reveal, cursor glow
   ========================================================== */

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const isTouch = window.matchMedia("(pointer: coarse)").matches;

// scroll reveal for panels
const observer = new IntersectionObserver(
  (entries) => entries.forEach((e) => e.isIntersecting && e.target.classList.add("visible")),
  { threshold: 0.12 }
);
document.querySelectorAll(".panel").forEach((p) => observer.observe(p));

// 3D tilt on hover for cards and the result panel
function addTilt(el, maxDeg = 7) {
  el.addEventListener("mousemove", (e) => {
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    el.style.transform =
      `perspective(800px) rotateY(${px * maxDeg}deg) rotateX(${-py * maxDeg}deg) translateY(-4px)`;
  });
  el.addEventListener("mouseleave", () => { el.style.transform = ""; });
}
if (!reduceMotion && !isTouch) {
  document.querySelectorAll(".card, .graph-card, .result-side").forEach((el) => addTilt(el));
  // mode cards are created dynamically — tilt via delegation
  document.getElementById("modeCards").addEventListener("mouseover", (e) => {
    const card = e.target.closest(".mode-card");
    if (card && !card.dataset.tilt) { card.dataset.tilt = "1"; addTilt(card, 10); }
  });
}

// cursor glow
const glow = document.getElementById("cursorGlow");
if (!reduceMotion && !isTouch && glow) {
  document.addEventListener("mousemove", (e) => {
    glow.style.left = `${e.clientX}px`;
    glow.style.top = `${e.clientY}px`;
  });
} else if (glow) {
  glow.style.display = "none";
}



/* ---------- SHAP explanation bars ---------- */

function renderExplanation(exp) {
  if (!exp) return;
  document.getElementById("shapBase").textContent = exp.base_value;
  const wrap = document.getElementById("shapBars");
  wrap.innerHTML = "";
  const maxAbs = Math.max(...exp.contributions.map((c) => Math.abs(c.impact)), 1);
  exp.contributions.forEach((c) => {
    const row = document.createElement("div");
    row.className = "shap-row";
    const cls = c.impact >= 0 ? "pos" : "neg";
    const sign = c.impact >= 0 ? "+" : "";
    row.innerHTML =
      `<span class="shap-name">${c.feature}</span>` +
      `<span class="shap-track"><span class="shap-fill ${cls}"></span></span>` +
      `<span class="shap-val ${cls}">${sign}${c.impact}</span>`;
    wrap.appendChild(row);
    const fill = row.querySelector(".shap-fill");
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        fill.style.width = `${(Math.abs(c.impact) / maxAbs) * 48}%`;
      })
    );
  });
}

/* ---------- AI traffic report (typewriter) ---------- */

const reportBtn = document.getElementById("reportBtn");

function typewrite(el, text, speed = 12) {
  el.textContent = "";
  el.classList.remove("done");
  let i = 0;
  (function tick() {
    el.textContent = text.slice(0, ++i);
    if (i < text.length) setTimeout(tick, speed);
    else el.classList.add("done");
  })();
}

reportBtn.addEventListener("click", async () => {
  if (!lastData) return;
  reportBtn.disabled = true;
  reportBtn.textContent = "Generating…";
  try {
    const ctx = {
      origin: lastData.origin.name,
      destination: lastData.destination.name,
      hour: parseInt(document.getElementById("hour").value, 10),
      day: parseInt(document.getElementById("day").value, 10),
      weather: parseInt(document.getElementById("weather").value, 10),
      level: lastData.level,
      route_traffic: lastData.route_traffic,
      congestion: lastData.congestion,
      distance_km: lastData.distance_km,
      eta_minutes: lastData.eta_minutes,
      fastest_mode: lastData.modes.find((m) => m.key === lastData.fastest_mode).name,
      best_hour: lastData.best_hour,
    };
    const res = await fetch("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ctx),
    });
    const data = await res.json();
    document.getElementById("reportCard").classList.remove("hidden");
    document.getElementById("reportLabel").textContent =
      data.source === "llm" ? "AI TRAFFIC ADVISORY — GENERATED BY CLAUDE" : "TRAFFIC ADVISORY — RULE-BASED (set ANTHROPIC_API_KEY for AI)";
    typewrite(document.getElementById("reportText"), data.report);
  } catch (err) {
    console.error(err);
  } finally {
    reportBtn.disabled = false;
    reportBtn.textContent = "🤖 Generate AI Traffic Report";
  }
});

console.log("Traffic Prediction System Loaded Successfully — Neon Night Drive");
