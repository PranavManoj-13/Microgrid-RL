# Microgrid RL Dashboard

A full-stack reinforcement learning project comparing **Q-Learning**, **SARSA**, and **DQN**
on a smart microgrid energy management problem.

## File Structure

```
microgrid_project/
├── app.py              ← Flask API server (run this)
├── rl_core.py          ← RL environment + all three agents (pure NumPy)
├── requirements.txt    ← Python dependencies
└── static/
    └── index.html      ← Frontend dashboard (served by Flask)
```

## How It Works

```
index.html  ──POST /api/train──►  app.py  ──imports──►  rl_core.py
    │                                │                      │
    │  ◄──GET /api/curves (poll)──   │   trains agents      │
    │  ◄──GET /api/log    (poll)──   │   in background      │
    │  ◄──GET /api/results ───────   │   thread             │
    │                                │                      │
    └── renders Chart.js charts ─────┘                      │
        from real Python results ◄──────────────────────────┘
```

- `rl_core.py` defines the `MicrogridEnv`, `QLearningAgent`, `SARSAAgent`, and `DQNAgent`.
- `app.py` runs training in a background thread and exposes REST endpoints.
- `static/index.html` polls the API every 600ms to stream live curves and logs.

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the Flask server
python app.py

# 3. Open your browser
#    http://localhost:5000
```

## Deployment on Render

1. Push this repository to GitHub.

2. Connect your Render account to the GitHub repo.

3. Create a new Web Service on Render.

4. Set the following:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Environment Variables**: Add `FLASK_ENV=production`

5. Deploy!

Note: Render's free tier has resource limits. Long training runs may timeout. Consider upgrading for intensive RL training.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/train` | Start training with JSON config |
| POST | `/api/stop` | Stop current training run |
| GET | `/api/status` | Progress, episode count, errors |
| GET | `/api/curves` | Smoothed reward curves (live) |
| GET | `/api/log?since=N` | Per-episode log rows since episode N |
| GET | `/api/results` | Final evaluation results (when done) |
| GET | `/api/profile` | 24h solar/demand profile for given params |

## Environment

| Parameter | Default | Description |
|-----------|---------|-------------|
| Battery capacity | 10 kWh | Max storable energy |
| Solar peak | 5 kW | Peak solar generation (noon) |
| Base demand | 2 kW | Baseline household demand |
| Import price | £0.28/kWh | Grid buy price |
| Export price | £0.10/kWh | Grid sell price |
| Noise | 0.30 | Stochastic variation in solar/demand |

## Algorithms

- **Q-Learning** — off-policy TD; bootstraps from `max Q(s',·)`
- **SARSA** — on-policy TD; bootstraps from `Q(s', a')` under current π
- **DQN** — neural Q-network (pure NumPy, 2-layer MLP, Adam, replay buffer, target net)