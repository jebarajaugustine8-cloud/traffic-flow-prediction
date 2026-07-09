/* ==========================================================
   Traffic Flow Prediction — frontend logic
   - Sends form values to /api/predict as JSON (no page reload)
   - Lights the correct traffic-signal lamp
   - Animates the vehicle counter and congestion meter
   - Draws the 24-hour forecast as bars, highlighting:
       amber = selected hour, green = best hour, red = worst hour
   ========================================================== */

const btn = document.getElementById("predictBtn");
const errorMsg = document.getElementById("errorMsg");

const lamps = {
  red: document.getElementById("lampRed"),
  amber: document.getElementById("lampAmber"),
  green: document.getElementById("lampGreen"),
};

function setSignal(color) {
  Object.values(lamps).forEach((l) => l.classList.remove("on"));
  if (lamps[color]) lamps[color].classList.add("on");
}

function animateCount(el, target, duration = 700) {
  const start = performance.now();
  function frame(now) {
    const p = Math.min((now - start) / duration, 1);
    el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3))); // ease-out
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
    // set height on next frame so the CSS transition animates
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
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Prediction request failed");
    const data = await res.json();

    // Show result block
    document.getElementById("resultPlaceholder").classList.add("hidden");
    document.getElementById("resultData").classList.remove("hidden");

    // Signal + level
    setSignal(data.signal);
    const level = document.getElementById("levelLabel");
    level.textContent = data.level;
    level.className = `level-label level-${data.signal}`;

    // Animated values
    animateCount(document.getElementById("trafficValue"), data.traffic);
    document.getElementById("congestionValue").textContent = data.congestion;
    document.getElementById("meterFill").style.width = `${data.congestion}%`;
    document.getElementById("delayValue").textContent = data.delay;
    document.getElementById("adviceValue").textContent = data.advice;

    // Forecast
    drawForecast(
      data.forecast,
      parseInt(payload.hour, 10),
      data.best_hour,
      data.worst_hour
    );
  } catch (err) {
    errorMsg.textContent = "Could not get a prediction. Is the server running?";
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict traffic";
  }
}

btn.addEventListener("click", predict);

// Allow pressing Enter inside the hour field
document.getElementById("hour").addEventListener("keydown", (e) => {
  if (e.key === "Enter") predict();
});

console.log("Traffic Prediction System Loaded Successfully");
