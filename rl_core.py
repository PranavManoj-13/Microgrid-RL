import numpy as np
import random
from collections import defaultdict, deque


class MicrogridEnv:

    HOURS = 24
    BATTERY_LEVELS = 11 
    N_ACTIONS = 3

    def __init__(self, battery_max_kwh=5.0, solar_peak_kw=2.0,
                 demand_base_kw=0.8, import_price=6.50, export_price=3.00,
                 noise=0.20, seed=42):
        self.battery_max   = battery_max_kwh
        self.solar_peak    = solar_peak_kw
        self.demand_base   = demand_base_kw
        self.import_price  = import_price   
        self.export_price  = export_price   
        self.noise         = noise
        self.rng           = np.random.default_rng(seed)
        self.reset()

    def _solar(self, h):
        base = max(0.0, self.solar_peak * np.exp(-0.5 * ((h - 13) / 2.5) ** 2))
        return max(0.0, base + self.rng.normal(0, self.noise))

    def _demand(self, h):
        m = 1.0 * np.exp(-0.5 * ((h - 7)  / 1.2) ** 2)
        e = 2.5 * np.exp(-0.5 * ((h - 19.5) / 1.8) ** 2)
        return max(0.3, self.demand_base + m + e + self.rng.normal(0, self.noise * 0.6))

    def _surplus_bin(self, h):
        net = (self.solar_peak * np.exp(-0.5 * ((h - 13) / 2.5) ** 2)
               - (self.demand_base
                  + 1.0 * np.exp(-0.5 * ((h - 7)   / 1.2) ** 2)
                  + 2.5 * np.exp(-0.5 * ((h - 19.5) / 1.8) ** 2)))
        return 0 if net < -0.5 else (2 if net > 0.5 else 1)

    def _batt_level(self):
        return int(round(self.battery_kwh / self.battery_max * 10))

    def _state(self):
        return (self.hour, self._batt_level(), self._surplus_bin(self.hour))

    def reset(self):
        self.hour        = 0
        self.battery_kwh = self.battery_max * 0.5
        self.total_cost  = 0.0
        return self._state()

    def step(self, action):
        solar  = self._solar(self.hour)
        demand = self._demand(self.hour)
        net    = solar - demand
        reward = cost = 0.0
        batt   = self.battery_kwh
        cap    = 1.5   

        if action == 0:         
            charge = min(cap, self.battery_max - batt)
            if net >= charge:
                batt  += charge
                sell   = net - charge
                reward += sell * self.export_price
                cost   -= sell * self.export_price
            else:
                batt  += max(0, net)
                buy    = charge - max(0, net)
                cost  += buy * self.import_price
                reward -= buy * self.import_price

        elif action == 1:       
            dis   = min(cap, batt)
            batt -= dis
            avail = solar + dis
            if avail >= demand:
                sell   = avail - demand
                reward += sell * self.export_price
                cost   -= sell * self.export_price
            else:
                buy    = demand - avail
                cost  += buy * self.import_price
                reward -= buy * self.import_price

        else:                   
            if net > 0:
                reward += net * self.export_price
                cost   -= net * self.export_price
            else:
                buy    = abs(net)
                cost  += buy * self.import_price
                reward -= buy * self.import_price

        self.battery_kwh  = max(0.0, min(self.battery_max, batt))
        self.total_cost  += cost
        self.hour        += 1
        done = self.hour >= self.HOURS
        info = dict(cost=cost, solar=solar, demand=demand, battery=self.battery_kwh)
        return (self._state() if not done else None), reward, done, info

    @property
    def n_states(self):
        return self.HOURS * self.BATTERY_LEVELS * 3


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(3))

    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randint(0, 2)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, sn, done):
        target = r if (done or sn is None) else r + self.gamma * np.max(self.Q[sn])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def run_episode(self, env):
        s = env.reset()
        total = 0.0
        done = False
        while not done:
            a = self.act(s)
            sn, r, done, _ = env.step(a)
            self.update(s, a, r, sn, done)
            if not done:
                s = sn
            total += r
        self.decay_epsilon()
        return total


class SARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(3))

    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randint(0, 2)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, sn, an, done):
        target = r if (done or sn is None) else r + self.gamma * self.Q[sn][an]
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def run_episode(self, env):
        s = env.reset()
        a = self.act(s)
        total = 0.0
        done = False
        while not done:
            sn, r, done, _ = env.step(a)
            an = self.act(sn) if not done else 0
            self.update(s, a, r, sn, an, done)
            if not done:
                s, a = sn, an
            total += r
        self.decay_epsilon()
        return total


class MLP:

    def __init__(self, in_dim=3, hidden=64, out_dim=3, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim),  (hidden, in_dim)).astype(np.float32)
        self.b1 = np.zeros((hidden, 1), dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), (out_dim, hidden)).astype(np.float32)
        self.b2 = np.zeros((out_dim, 1), dtype=np.float32)
        params = [self.W1, self.b1, self.W2, self.b2]
        self.m  = [np.zeros_like(p) for p in params]
        self.v  = [np.zeros_like(p) for p in params]
        self.t  = 0
        self.b1c, self.b2c, self.eps = 0.9, 0.999, 1e-8

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        self._x  = x
        self._z1 = self.W1 @ x + self.b1
        self._a1 = np.maximum(0, self._z1)
        self._z2 = self.W2 @ self._a1 + self.b2
        return self._z2.flatten()

    def backward(self, grad):
        g = np.asarray(grad, dtype=np.float32).reshape(-1, 1)
        dW2 = g @ self._a1.T
        db2 = g
        dz1 = (self.W2.T @ g) * (self._z1 > 0)
        dW1 = dz1 @ self._x.T
        db1 = dz1
        self.t += 1
        for i, (p, gr) in enumerate(zip(
                [self.W1, self.b1, self.W2, self.b2],
                [dW1,    db1,    dW2,    db2])):
            self.m[i] = self.b1c * self.m[i] + (1 - self.b1c) * gr
            self.v[i] = self.b2c * self.v[i] + (1 - self.b2c) * gr ** 2
            mh = self.m[i] / (1 - self.b1c ** self.t)
            vh = self.v[i] / (1 - self.b2c ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)

    def copy_weights_from(self, other):
        self.W1 = other.W1.copy(); self.b1 = other.b1.copy()
        self.W2 = other.W2.copy(); self.b2 = other.b2.copy()


class DQNAgent:
    def __init__(self, hidden=64, lr=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=32, memory_size=5000, target_update=10):
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps         = 0
        self.online = MLP(3, hidden, 3, lr)
        self.target = MLP(3, hidden, 3, lr)
        self.target.copy_weights_from(self.online)
        self.memory = deque(maxlen=memory_size)

    @staticmethod
    def _enc(s):
        if s is None:
            return np.zeros(3, dtype=np.float32)
        return np.array([s[0] / 23.0, s[1] / 10.0, s[2] / 2.0], dtype=np.float32)

    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randint(0, 2)
        return int(np.argmax(self.online.forward(self._enc(state))))

    def remember(self, s, a, r, sn, done):
        self.memory.append((s, a, r, sn, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for s, a, r, sn, done in batch:
            q = self.online.forward(self._enc(s))
            tgt = q.copy()
            tgt[a] = r if (done or sn is None) else (
                r + self.gamma * np.max(self.target.forward(self._enc(sn))))
            self.online.forward(self._enc(s))
            self.online.backward(2 * (q - tgt) / self.batch_size)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.copy_weights_from(self.online)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def run_episode(self, env):
        s = env.reset()
        total = 0.0
        done = False
        while not done:
            a = self.act(s)
            sn, r, done, _ = env.step(a)
            self.remember(s, a, r, sn, done)
            self.replay()
            if not done:
                s = sn
            total += r
        return total


def evaluate(agent, env_cfg, n=50, seed=99):
    env = MicrogridEnv(**env_cfg, seed=seed)
    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    costs, batt_prof, cost_prof, action_counts = [], np.zeros(24), np.zeros(24), np.zeros(3)
    for _ in range(n):
        s = env.reset()
        done = False
        running_cost = 0.0
        while not done:
            a = agent.act(s, explore=False)
            action_counts[a] += 1
            s, _, done, info = env.step(a)
            running_cost += info['cost']
            batt_prof[env.hour - 1]  += env.battery_kwh
            cost_prof[env.hour - 1]  += running_cost
        costs.append(env.total_cost)

    agent.epsilon = saved_eps
    costs = np.array(costs)
    return {
        'mean':    float(costs.mean()),
        'std':     float(costs.std()),
        'batt_profile': (batt_prof / n).tolist(),
        'cost_profile': (cost_prof / n).tolist(),
        'action_pct':   (action_counts / action_counts.sum() * 100).tolist(),
    }


def smooth(arr, window=20):
    out = []
    for i, v in enumerate(arr):
        sl = arr[max(0, i - window + 1): i + 1]
        out.append(sum(sl) / len(sl))
    return out