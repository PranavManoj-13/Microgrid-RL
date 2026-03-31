import json
import random
import threading
import time
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from rl_core import (
    MicrogridEnv,
    QLearningAgent, SARSAAgent, DQNAgent,
    evaluate, smooth,
)

app = Flask(__name__, static_folder="static")
CORS(app)

_lock    = threading.Lock()
_state   = {
    "running":  False,
    "progress": 0,       
    "episode":  0,
    "total":    0,
    "log":      [],      
    "results":  None,
    "curves":   {"q": [], "s": [], "d": []},
    "error":    None,
}

def _reset_state():
    with _lock:
        _state.update(running=False, progress=0, episode=0, total=0,
                      log=[], results=None,
                      curves={"q": [], "s": [], "d": []}, error=None)



def _train(cfg):
    try:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        env_cfg = dict(
            battery_max_kwh = cfg["battery_max"],
            solar_peak_kw   = cfg["solar_peak"],
            demand_base_kw  = cfg["demand_base"],
            import_price    = cfg["import_price"],
            export_price    = cfg["export_price"],
            noise           = cfg["noise"],
        )
        n_ep    = cfg["episodes"]
        enabled = cfg["enabled"]       

        agent_kwargs = dict(
            alpha         = cfg["alpha"],
            gamma         = cfg["gamma"],
            epsilon       = cfg["epsilon"],
            epsilon_min   = 0.01,
            epsilon_decay = cfg["epsilon_decay"],
        )
        q_agent = QLearningAgent(**agent_kwargs) if enabled.get("q") else None
        s_agent = SARSAAgent(**agent_kwargs)     if enabled.get("s") else None
        d_agent = DQNAgent(
            hidden        = cfg.get("hidden", 64),
            lr            = cfg.get("lr", 1e-3),
            gamma         = cfg["gamma"],
            epsilon       = cfg["epsilon"],
            epsilon_min   = 0.01,
            epsilon_decay = cfg["epsilon_decay"],
            batch_size    = cfg.get("batch_size", 32),
            memory_size   = cfg.get("memory_size", 5000),
            target_update = cfg.get("target_update", 10),
        ) if enabled.get("d") else None

        make_env = lambda s=cfg["seed"]: MicrogridEnv(**env_cfg, seed=s)
        q_env = make_env(cfg["seed"])      if q_agent else None
        s_env = make_env(cfg["seed"] + 1)  if s_agent else None
        d_env = make_env(cfg["seed"] + 2)  if d_agent else None

        q_rewards, s_rewards, d_rewards = [], [], []

        for ep in range(n_ep):
            row = {"ep": ep + 1}

            if q_agent:
                r = q_agent.run_episode(q_env)
                q_rewards.append(r)
                row["q"] = round(r, 4)

            if s_agent:
                r = s_agent.run_episode(s_env)
                s_rewards.append(r)
                row["s"] = round(r, 4)

            if d_agent:
                r = d_agent.run_episode(d_env)
                d_rewards.append(r)
                row["d"] = round(r, 4)

            with _lock:
                _state["episode"]  = ep + 1
                _state["progress"] = round((ep + 1) / n_ep * 100, 1)
                _state["log"].append(row)
                if len(_state["log"]) > 200:
                    _state["log"].pop(0)

                if (ep + 1) % 5 == 0 or ep == n_ep - 1:
                    _state["curves"] = {
                        "q": smooth(q_rewards) if q_rewards else [],
                        "s": smooth(s_rewards) if s_rewards else [],
                        "d": smooth(d_rewards) if d_rewards else [],
                    }

        results = {}
        if q_agent:
            results["q"] = evaluate(q_agent, env_cfg)
        if s_agent:
            results["s"] = evaluate(s_agent, env_cfg)
        if d_agent:
            results["d"] = evaluate(d_agent, env_cfg)

        with _lock:
            _state["results"]  = results
            _state["running"]  = False
            _state["progress"] = 100

    except Exception as exc:
        import traceback
        with _lock:
            _state["error"]   = str(exc) + "\n" + traceback.format_exc()
            _state["running"] = False



@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/train", methods=["POST"])
def train():
    if _state["running"]:
        return jsonify({"error": "Already training"}), 409
    cfg = request.get_json(force=True)
    _reset_state()
    with _lock:
        _state["running"] = True
        _state["total"]   = cfg.get("episodes", 300)
    t = threading.Thread(target=_train, args=(cfg,), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop():
    with _lock:
        _state["running"] = False
    return jsonify({"ok": True})


@app.route("/api/status")
def status():
    with _lock:
        data = {
            "running":  _state["running"],
            "progress": _state["progress"],
            "episode":  _state["episode"],
            "total":    _state["total"],
            "error":    _state["error"],
        }
    return jsonify(data)


@app.route("/api/curves")
def curves():
    with _lock:
        return jsonify(_state["curves"])


@app.route("/api/log")
def log():
    since = int(request.args.get("since", 0))
    with _lock:
        entries = [r for r in _state["log"] if r["ep"] > since]
    return jsonify(entries)


@app.route("/api/results")
def results():
    with _lock:
        res = _state["results"]
    if res is None:
        return jsonify({"ready": False})
    return jsonify({"ready": True, "data": res})


@app.route("/api/profile")
def profile():
    """Return deterministic 24-h solar/demand profile for current env params."""
    p = request.args
    env = MicrogridEnv(
        battery_max_kwh = float(p.get("battery_max", 10)),
        solar_peak_kw   = float(p.get("solar_peak", 5)),
        demand_base_kw  = float(p.get("demand_base", 2)),
        noise           = 0.0,         
        seed            = 0,
    )
    solar, demand = [], []
    for h in range(24):
        env.noise = 0
        solar.append(round(float(env._solar(h)), 3))
        demand.append(round(float(env._demand(h)), 3))
    return jsonify({"solar": solar, "demand": demand})

@app.route("/api/stream")
def stream():
    def event_gen():
        last_ep = 0
        while True:
            with _lock:
                running  = _state["running"]
                progress = _state["progress"]
                episode  = _state["episode"]
                error    = _state["error"]
                new_log  = [r for r in _state["log"] if r["ep"] > last_ep]
                curves   = _state["curves"] if episode % 5 == 0 else None
                results  = _state["results"] if not running and _state["results"] else None

            payload = {"progress": progress, "episode": episode,
                       "running": running, "log": new_log}
            if curves:
                payload["curves"] = curves
            if results:
                payload["results"] = results
            if error:
                payload["error"] = error

            if new_log:
                last_ep = new_log[-1]["ep"]

            yield f"data: {json.dumps(payload)}\n\n"

            if not running and (results or error):
                break
            time.sleep(0.25)

    return Response(event_gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import os
    os.makedirs("static", exist_ok=True)
    print("\n  Microgrid RL Dashboard")
    print("  ─────────────────────────────────")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000, threaded=True)
