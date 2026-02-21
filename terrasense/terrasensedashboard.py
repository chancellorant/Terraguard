#!/usr/bin/env python3
import time, threading, serial
from flask import Flask, jsonify, Response

import board, busio
import adafruit_bme680
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# -------------------------
# CONFIG
# -------------------------
ADS_ADDR = 0x48
PMS_PORT = "/dev/ttyAMA0"
PMS_BAUD = 9600

# -------------------------
# GLOBAL STATE (thread-updated)
# -------------------------
state = {
    "bme": {"ok": False, "addr": None, "t_c": None, "rh": None, "p_hpa": None, "gas_kohm": None, "ts": None, "err": None},
    "pms": {"ok": False, "pm1": None, "pm25": None, "pm10": None, "ts": None, "err": None},
    "ads": {"ok": False, "a0": None, "a1": None, "a2": None, "a3": None, "ts": None, "err": None},
}

app = Flask(__name__)

# -------------------------
# SENSOR THREAD
# -------------------------
def sensor_loop():
    i2c = busio.I2C(board.SCL, board.SDA)

    # BME680 autodetect
    bme = None
    bme_addr = None
    for addr in (0x77, 0x76):
        try:
            bme = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=addr)
            bme_addr = addr
            break
        except Exception as e:
            pass

    # ADS1115
    ads = ADS.ADS1115(i2c, address=ADS_ADDR)
    ch0 = AnalogIn(ads, 0)
    ch1 = AnalogIn(ads, 1)
    ch2 = AnalogIn(ads, 2)
    ch3 = AnalogIn(ads, 3)

    # PMS5003
    ser = serial.Serial(PMS_PORT, PMS_BAUD, timeout=2)
    buf = b""

    pm1 = pm25 = pm10 = None

    def pms_poll():
        nonlocal buf, pm1, pm25, pm10
        buf += ser.read(128)
        while True:
            i = buf.find(b"\x42\x4D")
            if i < 0:
                buf = buf[-200:]
                return
            if len(buf) < i + 32:
                return
            f = buf[i:i+32]
            buf = buf[i+32:]
            pm1  = f[10]*256 + f[11]
            pm25 = f[12]*256 + f[13]
            pm10 = f[14]*256 + f[15]

    while True:
        now = time.time()

        # PMS
        try:
            pms_poll()
            state["pms"].update({"ok": True, "pm1": pm1, "pm25": pm25, "pm10": pm10, "ts": now, "err": None})
        except Exception as e:
            state["pms"].update({"ok": False, "err": str(e), "ts": now})

        # BME
        try:
            if bme is None:
                raise RuntimeError("BME680 not found")
            state["bme"].update({
                "ok": True, "addr": bme_addr,
                "t_c": float(bme.temperature),
                "rh": float(bme.humidity),
                "p_hpa": float(bme.pressure),
                "gas_kohm": float(bme.gas)/1000.0,
                "ts": now, "err": None
            })
        except Exception as e:
            state["bme"].update({"ok": False, "err": str(e), "ts": now, "addr": bme_addr})

        # ADS
        try:
            state["ads"].update({
                "ok": True,
                "a0": float(ch0.voltage),
                "a1": float(ch1.voltage),
                "a2": float(ch2.voltage),
                "a3": float(ch3.voltage),
                "ts": now, "err": None
            })
        except Exception as e:
            state["ads"].update({"ok": False, "err": str(e), "ts": now})

        time.sleep(0.25)

threading.Thread(target=sensor_loop, daemon=True).start()

# -------------------------
# API ROUTES
# -------------------------
@app.get("/api/bme")
def api_bme(): return jsonify(state["bme"])

@app.get("/api/pms")
def api_pms(): return jsonify(state["pms"])

@app.get("/api/ads")
def api_ads(): return jsonify(state["ads"])

# -------------------------
# DASHBOARD (POPUPS)
# -------------------------
@app.get("/")
def index():
    html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>TerraSense Popups</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; padding: 18px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; border-radius: 14px; padding: 14px; width: 320px; }
    .title { font-weight: 700; font-size: 18px; margin-bottom: 8px; }
    .mini { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas; font-size: 13px; white-space: pre; }
    button { padding: 10px 12px; border-radius: 10px; border: 1px solid #ccc; background: #fff; cursor: pointer; }
    button:hover { background: #f6f6f6; }

    /* Modal */
    .modal { display:none; position: fixed; inset: 0; background: rgba(0,0,0,.45); }
    .modal-inner { background: #fff; width: min(680px, 92vw); margin: 8vh auto; border-radius: 16px; padding: 16px; }
    .modal-head { display:flex; justify-content: space-between; align-items:center; gap: 8px; }
    .x { font-size: 22px; border: none; background: transparent; cursor:pointer; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
    .kv { border: 1px solid #eee; border-radius: 12px; padding: 10px; }
    .k { color:#666; font-size: 12px; }
    .v { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas; font-size: 18px; margin-top: 6px; }
    .status { margin-top: 8px; color:#666; font-size: 12px; }
  </style>
</head>
<body>
  <h2>TerraSense â€” Sensor Popups</h2>

  <div class="row">
    <div class="card">
      <div class="title">BME680 (Temp / RH / Pressure / Gas)</div>
      <div id="bmeMini" class="mini">loading...</div>
      <div style="margin-top:10px;"><button onclick="openModal('bmeModal')">Open BME680 Popup</button></div>
    </div>

    <div class="card">
      <div class="title">PMS5003 (PM1 / PM2.5 / PM10)</div>
      <div id="pmsMini" class="mini">loading...</div>
      <div style="margin-top:10px;"><button onclick="openModal('pmsModal')">Open PMS5003 Popup</button></div>
    </div>

    <div class="card">
      <div class="title">ADS1115 (A0â€“A3 voltages)</div>
      <div id="adsMini" class="mini">loading...</div>
      <div style="margin-top:10px;"><button onclick="openModal('adsModal')">Open ADS1115 Popup</button></div>
    </div>
  </div>

  <!-- BME MODAL -->
  <div id="bmeModal" class="modal" onclick="bgClose(event,'bmeModal')">
    <div class="modal-inner">
      <div class="modal-head">
        <div class="title">BME680 Details</div>
        <button class="x" onclick="closeModal('bmeModal')">Ã—</button>
      </div>
      <div class="grid">
        <div class="kv"><div class="k">Temperature (Â°C)</div><div class="v" id="bmeT">â€”</div></div>
        <div class="kv"><div class="k">Humidity (%)</div><div class="v" id="bmeRH">â€”</div></div>
        <div class="kv"><div class="k">Pressure (hPa)</div><div class="v" id="bmeP">â€”</div></div>
        <div class="kv"><div class="k">Gas (kÎ©)</div><div class="v" id="bmeG">â€”</div></div>
      </div>
      <div class="status" id="bmeStatus"></div>
    </div>
  </div>

  <!-- PMS MODAL -->
  <div id="pmsModal" class="modal" onclick="bgClose(event,'pmsModal')">
    <div class="modal-inner">
      <div class="modal-head">
        <div class="title">PMS5003 Details</div>
        <button class="x" onclick="closeModal('pmsModal')">Ã—</button>
      </div>
      <div class="grid">
        <div class="kv"><div class="k">PM1.0 (Âµg/mÂ³)</div><div class="v" id="pm1">â€”</div></div>
        <div class="kv"><div class="k">PM2.5 (Âµg/mÂ³)</div><div class="v" id="pm25">â€”</div></div>
        <div class="kv"><div class="k">PM10 (Âµg/mÂ³)</div><div class="v" id="pm10">â€”</div></div>
        <div class="kv"><div class="k">UART</div><div class="v">/dev/ttyAMA0</div></div>
      </div>
      <div class="status" id="pmsStatus"></div>
    </div>
  </div>

  <!-- ADS MODAL -->
  <div id="adsModal" class="modal" onclick="bgClose(event,'adsModal')">
    <div class="modal-inner">
      <div class="modal-head">
        <div class="title">ADS1115 Details</div>
        <button class="x" onclick="closeModal('adsModal')">Ã—</button>
      </div>
      <div class="grid">
        <div class="kv"><div class="k">A0 (V)</div><div class="v" id="a0">â€”</div></div>
        <div class="kv"><div class="k">A1 (V)</div><div class="v" id="a1">â€”</div></div>
        <div class="kv"><div class="k">A2 (V)</div><div class="v" id="a2">â€”</div></div>
        <div class="kv"><div class="k">A3 (V)</div><div class="v" id="a3">â€”</div></div>
      </div>
      <div class="status" id="adsStatus"></div>
    </div>
  </div>

<script>
function openModal(id){ document.getElementById(id).style.display='block'; }
function closeModal(id){ document.getElementById(id).style.display='none'; }
function bgClose(ev,id){ if(ev.target.id===id) closeModal(id); }

function fmt(x, d=2){
  if(x===null || x===undefined) return "NA";
  if(typeof x === "number") return x.toFixed(d);
  return String(x);
}
async function poll(){
  const [bme,pms,ads] = await Promise.all([
    fetch('/api/bme').then(r=>r.json()),
    fetch('/api/pms').then(r=>r.json()),
    fetch('/api/ads').then(r=>r.json())
  ]);

  // minis
  document.getElementById('bmeMini').textContent =
    `ok=${bme.ok} addr=${bme.addr}\nT=${fmt(bme.t_c)}C  RH=${fmt(bme.rh,1)}%\nP=${fmt(bme.p_hpa,1)}hPa  GAS=${fmt(bme.gas_kohm,1)}kÎ©`;
  document.getElementById('pmsMini').textContent =
    `ok=${pms.ok}\nPM1=${pms.pm1 ?? 'NA'}  PM2.5=${pms.pm25 ?? 'NA'}\nPM10=${pms.pm10 ?? 'NA'}`;
  document.getElementById('adsMini').textContent =
    `ok=${ads.ok}\nA0=${fmt(ads.a0,3)}V  A1=${fmt(ads.a1,3)}V\nA2=${fmt(ads.a2,3)}V  A3=${fmt(ads.a3,3)}V`;

  // modal fields
  document.getElementById('bmeT').textContent  = fmt(bme.t_c);
  document.getElementById('bmeRH').textContent = fmt(bme.rh,1);
  document.getElementById('bmeP').textContent  = fmt(bme.p_hpa,1);
  document.getElementById('bmeG').textContent  = fmt(bme.gas_kohm,1);
  document.getElementById('bmeStatus').textContent = bme.ok ? "" : ("Error: " + bme.err);

  document.getElementById('pm1').textContent  = (pms.pm1 ?? "NA");
  document.getElementById('pm25').textContent = (pms.pm25 ?? "NA");
  document.getElementById('pm10').textContent = (pms.pm10 ?? "NA");
  document.getElementById('pmsStatus').textContent = pms.ok ? "" : ("Error: " + pms.err);

  document.getElementById('a0').textContent = fmt(ads.a0,3);
  document.getElementById('a1').textContent = fmt(ads.a1,3);
  document.getElementById('a2').textContent = fmt(ads.a2,3);
  document.getElementById('a3').textContent = fmt(ads.a3,3);
  document.getElementById('adsStatus').textContent = ads.ok ? "" : ("Error: " + ads.err);
}
setInterval(poll, 600);
poll();
</script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")

if __name__ == "__main__":
    # Accessible from your laptop on the same network
    app.run(host="0.0.0.0", port=8080, debug=False)
