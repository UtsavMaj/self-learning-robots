"""Self-learning robots: bandits, crawler (walking) and navigation with a NumPy Deep Q-Network."""
import json, collections, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures') + os.sep
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.grid': True, 'grid.alpha': .3})
RES = {}

# ---------------------------------------------------------------- 1. Bandits
def bandit_experiment(method, steps=1000, runs=1000, k=10, eps=0.1, K=2.0, seed=0):
    rng = np.random.default_rng(seed)
    avg = np.zeros(steps); opt = np.zeros(steps)
    for _ in range(runs):
        mu = rng.normal(0, 1, k); best = mu.argmax()
        n = np.zeros(k); s1 = np.zeros(k); s2 = np.zeros(k)
        for t in range(steps):
            Q = np.divide(s1, n, out=np.zeros(k), where=n > 0)
            if method == 'naive':
                a = t % k if t < 20 * k else int(Q.argmax())
            elif method.startswith('egreedy'):
                a = int(rng.integers(k)) if rng.random() < eps else int(Q.argmax())
            else:
                if (n == 0).any(): a = int(np.where(n == 0)[0][0])
                else:
                    var = np.maximum(s2 / n - Q ** 2, 1e-4)
                    a = int((Q + K * np.sqrt(var) / np.sqrt(n)).argmax())
            r = rng.normal(mu[a], 1)
            n[a] += 1; s1[a] += r; s2[a] += r * r
            avg[t] += r; opt[t] += (a == best)
    return avg / runs, opt / runs * 100

curves = {}
for name, kw in [('naive', dict(method='naive')), ('egreedy 0.01', dict(method='egreedy', eps=0.01)),
                 ('egreedy 0.1', dict(method='egreedy', eps=0.1)), ('ucb', dict(method='ucb', K=2.0))]:
    curves[name] = bandit_experiment(**kw, runs=600)
labels = {'naive': 'Naïve (20 trials/arm)', 'egreedy 0.01': 'ε-Greedy (ε=0.01)',
          'egreedy 0.1': 'ε-Greedy (ε=0.1)', 'ucb': 'Upper bound (K=2)'}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for k_, (a, o) in curves.items():
    ax[0].plot(a, label=labels[k_], lw=1.4); ax[1].plot(o, lw=1.4)
ax[0].set(xlabel='Step', ylabel='Average reward', title='(a) Average reward'); ax[0].legend(fontsize=8)
ax[1].set(xlabel='Step', ylabel='% optimal action', title='(b) Optimal action chosen')
plt.tight_layout(); plt.savefig(OUT + 'fig_bandit.png', dpi=200); plt.close()
RES['bandit'] = {labels[k_]: {'avg_reward_last_500': float(a[500:].mean()), 'avg_reward_all': float(a.mean()),
                              'optimal_pct_last_500': float(o[500:].mean())} for k_, (a, o) in curves.items()}

# ---------------------------------------------------------------- numpy DQN
class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]; self.v = [np.zeros_like(p) for p in self.W + self.b]; self.t = 0
    def forward(self, x, cache=False):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1: x = np.maximum(x, 0)
            acts.append(x)
        return (x, acts) if cache else x
    def step(self, x, a_idx, target, lr=1e-3):
        q, acts = self.forward(x, True)
        n = len(x); err = np.zeros_like(q)
        d = q[np.arange(n), a_idx] - target
        d = np.clip(d, -1, 1)                      # Huber gradient
        err[np.arange(n), a_idx] = d / n
        gW, gb = [], []
        g = err
        for i in reversed(range(len(self.W))):
            gW.insert(0, acts[i].T @ g); gb.insert(0, g.sum(0))
            if i > 0: g = (g @ self.W[i].T) * (acts[i] > 0)
        params = self.W + self.b; grads = gW + gb
        self.t += 1
        for j, (p, gr) in enumerate(zip(params, grads)):
            self.m[j] = .9 * self.m[j] + .1 * gr; self.v[j] = .999 * self.v[j] + .001 * gr * gr
            p -= lr * (self.m[j] / (1 - .9 ** self.t)) / (np.sqrt(self.v[j] / (1 - .999 ** self.t)) + 1e-8)
    def copy_from(self, o):
        self.W = [w.copy() for w in o.W]; self.b = [b.copy() for b in o.b]

class DQN:
    def __init__(self, n_in, n_act, hidden=(64, 64), seed=0, gamma=.97, lr=1e-3, buf=50000, bs=64, target_every=250):
        self.rng = np.random.default_rng(seed)
        self.q = MLP([n_in, *hidden, n_act], self.rng); self.tq = MLP([n_in, *hidden, n_act], self.rng); self.tq.copy_from(self.q)
        self.buf = collections.deque(maxlen=buf); self.gamma, self.lr, self.bs, self.te, self.n_act = gamma, lr, bs, target_every, n_act
        self.updates = 0
    def act(self, s, eps):
        if self.rng.random() < eps: return int(self.rng.integers(self.n_act))
        return int(self.q.forward(s[None])[0].argmax())
    def remember(self, *t): self.buf.append(t)
    def learn(self):
        if len(self.buf) < self.bs: return
        idx = self.rng.integers(len(self.buf), size=self.bs); b = [self.buf[i] for i in idx]
        s, a, r, s2, d = map(np.array, zip(*b))
        a2 = self.q.forward(s2).argmax(1)                       # double DQN
        tgt = r + self.gamma * (1 - d) * self.tq.forward(s2)[np.arange(self.bs), a2]
        self.q.step(s, a, tgt, self.lr); self.updates += 1
        if self.updates % self.te == 0: self.tq.copy_from(self.q)

# ---------------------------------------------------------------- 2. Crawler (walking)
class Crawler:
    """Two-joint crawling robot: shoulder angle a, wrist angle h. Foot on the ground drags the body."""
    L1 = L2 = 1.0; H = 1.55; STEP = .1
    A = (0.1, 1.2); Hh = (-0.8, 0.8)
    def reset(self):
        self.a, self.h, self.x, self.t = .6, 0., 0., 0
        return self.obs()
    def foot(self, a, h):
        return (self.L1 * np.cos(a) + self.L2 * np.cos(a + h), -(self.L1 * np.sin(a) + self.L2 * np.sin(a + h)))
    def planted(self, a, h): return self.foot(a, h)[1] <= -self.H
    def obs(self):
        return np.array([(self.a - self.A[0]) / (self.A[1] - self.A[0]) * 2 - 1,
                         (self.h - self.Hh[0]) / (self.Hh[1] - self.Hh[0]) * 2 - 1])
    def step(self, act):
        a0, h0 = self.a, self.h
        da, dh = [(self.STEP, 0), (-self.STEP, 0), (0, self.STEP), (0, -self.STEP)][act]
        self.a = float(np.clip(self.a + da, *self.A)); self.h = float(np.clip(self.h + dh, *self.Hh))
        dx = 0.
        if self.planted(a0, h0) and self.planted(self.a, self.h):
            dx = -(self.foot(self.a, self.h)[0] - self.foot(a0, h0)[0])   # foot pushes body forward
        self.x += dx; self.t += 1
        return self.obs(), dx * 10, self.t >= 200

def run_crawler(agent=None, policy='greedy', seed=0, record=False):
    env = Crawler(); s = env.reset(); rng = np.random.default_rng(seed); xs = [0.]; ang = [(env.a, env.h)]
    done = False
    while not done:
        act = int(rng.integers(4)) if agent is None else agent.act(s, 0.0)
        s, r, done = env.step(act); xs.append(env.x); ang.append((env.a, env.h))
    return env.x, xs, ang

def train_crawler(episodes=500, seed=1):
    env = Crawler(); ag = DQN(2, 4, hidden=(32, 32), seed=seed, gamma=.99, lr=2e-3, target_every=200)
    log = []; snap = {}; srng = np.random.default_rng(seed)
    for ep in range(episodes):
        eps = max(.05, 1.0 - ep / (.6 * episodes)); s = env.reset(); done = False; tot = 0
        env.a = float(np.clip(round(srng.integers(1, 13) * .1, 1), .1, 1.2)); env.h = round(srng.integers(-8, 9) * .1, 1); s = env.obs()  # random start pose
        while not done:
            a = ag.act(s, eps); s2, r, done = env.step(a)
            ag.remember(s, a, r, s2, float(done)); ag.learn(); s = s2; tot += r
        log.append(env.x)
        if ep in (0, episodes // 4, episodes - 1): snap[ep] = run_crawler(ag)
    return ag, np.array(log), snap

crawler_runs = []
for sd in range(5):
    ag, log, snap = train_crawler(seed=sd + 1); crawler_runs.append((ag, log, snap))
logs = np.array([c[1] for c in crawler_runs])
rand_d = np.array([run_crawler(None, seed=i)[0] for i in range(200)])
greedy_d = np.array([run_crawler(c[0])[0] for c in crawler_runs])
best = int(greedy_d.argmax()); bag, blog, bsnap = crawler_runs[best]
RES['crawler'] = {'random_mean': float(rand_d.mean()), 'random_std': float(rand_d.std()),
                  'trained_mean': float(greedy_d.mean()), 'trained_std': float(greedy_d.std()),
                  'trained_best': float(greedy_d.max()), 'trained_each': [float(x) for x in greedy_d],
                  'first50_mean': float(logs[:, :50].mean()), 'last50_mean': float(logs[:, -50:].mean()),
                  'episodes': logs.shape[1], 'seeds': logs.shape[0]}

def smooth(x, w=15): return np.convolve(x, np.ones(w) / w, mode='valid')
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
m = logs.mean(0); sd_ = logs.std(0)
ax[0].plot(smooth(m), lw=1.6, color='C0', label='DQN (mean of 5 seeds)')
ax[0].fill_between(range(len(smooth(m))), smooth(m - sd_), smooth(m + sd_), alpha=.2)
ax[0].axhline(rand_d.mean(), color='C3', ls='--', label='Random policy')
ax[0].set(xlabel='Training episode', ylabel='Distance crawled / episode (units)', title='(a) Crawler learning curve'); ax[0].legend(fontsize=8)
for ep, (d, xs, ang) in bsnap.items():
    ax[1].plot(xs, label=f'After episode {ep + 1} ({d:.1f})', lw=1.5)
ax[1].plot(run_crawler(None, seed=3)[1], color='k', ls=':', label='Random policy')
ax[1].set(xlabel='Time step', ylabel='Body position x', title='(b) Position during one test episode'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT + 'fig_crawler.png', dpi=200); plt.close()

# gait picture: joint-space trajectory before / after training
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
_, _, ang0 = run_crawler(None, seed=3); _, _, angN = run_crawler(bag)
for axx, ang_, ttl in [(ax[0], ang0, '(a) Random policy'), (ax[1], angN, '(b) Learned policy')]:
    A_, H_ = np.meshgrid(np.linspace(*Crawler.A, 100), np.linspace(*Crawler.Hh, 100))
    foot_y = -(np.sin(A_) + np.sin(A_ + H_)); axx.contourf(A_, H_, foot_y <= -Crawler.H, levels=[.5, 1.5], colors=['#f4c7c3'], alpha=.7)
    axx.plot([p[0] for p in ang_], [p[1] for p in ang_], '-o', ms=2, lw=.8, color='C0')
    axx.set(xlabel='Shoulder angle a (rad)', ylabel='Wrist angle h (rad)', title=ttl)
ax[0].text(.97, -.75, 'shaded = foot on ground', fontsize=7)
plt.tight_layout(); plt.savefig(OUT + 'fig_gait.png', dpi=200); plt.close()

# ---------------------------------------------------------------- 3. Navigation
N = 10; MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up(y+), down, right, left
def bfs(grid, s, g):
    q = collections.deque([(s, 0)]); seen = {s}
    while q:
        (x, y), d = q.popleft()
        if (x, y) == g: return d
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and not grid[ny, nx] and (nx, ny) not in seen:
                seen.add((nx, ny)); q.append(((nx, ny), d + 1))
    return None

def make_map(rng, density=.2):
    while True:
        grid = (rng.random((N, N)) < density)
        free = np.argwhere(~grid)
        s = tuple(free[rng.integers(len(free))][::-1]); g = tuple(free[rng.integers(len(free))][::-1])
        if s == g: continue
        d = bfs(grid, s, g)
        if d is not None and d >= 6: return grid, s, g, d

class NavEnv:
    def __init__(self, grid, s, g): self.grid, self.pos, self.goal, self.t = grid, s, g, 0
    def obs(self):
        x, y = self.pos; f = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                f.append(1. if not (0 <= nx < N and 0 <= ny < N) or self.grid[ny, nx] else 0.)
        gx, gy = self.goal[0] - x, self.goal[1] - y
        return np.array(f + [gx / N, gy / N, np.sign(gx), np.sign(gy)])
    def step(self, a):
        dx, dy = MOVES[a]; nx, ny = self.pos[0] + dx, self.pos[1] + dy; self.t += 1
        if not (0 <= nx < N and 0 <= ny < N) or self.grid[ny, nx]: r = -1.0       # collision
        else:
            self.pos = (nx, ny); r = -0.05
        done = False
        if self.pos == self.goal: r, done = 10.0, True
        return self.obs(), r, done or self.t >= 60

def evaluate_nav(agent, maps, rng_seed=0):
    rng = np.random.default_rng(rng_seed); succ = 0; ratio = []; coll = 0; steps_ = 0
    for grid, s, g, d in maps:
        env = NavEnv(grid, s, g); o = env.obs(); done = False; ok = False; c = 0
        while not done:
            a = int(rng.integers(4)) if agent is None else agent.act(o, 0.0)
            before = env.pos; o, r, done = env.step(a)
            if r == -1.0: c += 1
            if env.pos == g: ok = True
        succ += ok; coll += c; steps_ += env.t
        if ok: ratio.append(d / env.t)
    n = len(maps)
    return {'success': succ / n * 100, 'path_efficiency': float(np.mean(ratio)) if ratio else 0., 'collisions': coll / n, 'steps': steps_ / n}

def train_nav(episodes=4000, seed=0):
    rng = np.random.default_rng(seed + 100); ag = DQN(28, 4, hidden=(128, 128), seed=seed, gamma=.98, lr=1e-3, target_every=300)
    log = []; evalcurve = []
    test_maps = [make_map(np.random.default_rng(10_000 + i)) for i in range(200)]
    for ep in range(episodes):
        grid, s, g, d = make_map(rng); env = NavEnv(grid, s, g); o = env.obs(); done = False; tot = 0
        eps = max(.05, 1 - ep / (.6 * episodes))
        while not done:
            a = ag.act(o, eps); o2, r, done = env.step(a)
            ag.remember(o, a, r, o2, float(done)); ag.learn(); o = o2; tot += r
        log.append(tot)
        if ep % 100 == 0 or ep == episodes - 1:
            evalcurve.append((ep, evaluate_nav(ag, test_maps)['success']))
    return ag, np.array(log), evalcurve, test_maps

nav_agent, nav_log, nav_eval, test_maps = train_nav()
res_random = evaluate_nav(None, test_maps); res_trained = evaluate_nav(nav_agent, test_maps)
RES['nav'] = {'random': res_random, 'trained': res_trained, 'train_episodes': 4000, 'test_maps': len(test_maps),
              'first200_mean_return': float(nav_log[:200].mean()), 'last200_mean_return': float(nav_log[-200:].mean())}

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(smooth(nav_log, 50), color='C0'); ax[0].set(xlabel='Training episode', ylabel='Episode return (50-ep avg)', title='(a) Navigation learning curve')
e = np.array(nav_eval); ax[1].plot(e[:, 0], e[:, 1], '-o', ms=3, color='C2'); ax[1].axhline(res_random['success'], color='C3', ls='--', label='Random policy')
ax[1].set(xlabel='Training episode', ylabel='Success on 200 unseen maps (%)', title='(b) Test success rate', ylim=(0, 105)); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + 'fig_nav_curve.png', dpi=200); plt.close()

def trace(agent, m, rng_seed=5):
    grid, s, g, d = m; env = NavEnv(grid, s, g); o = env.obs(); path = [s]; done = False; rng = np.random.default_rng(rng_seed)
    while not done:
        a = int(rng.integers(4)) if agent is None else agent.act(o, 0.0); o, r, done = env.step(a); path.append(env.pos)
    return path, env.pos == g
fig, ax = plt.subplots(2, 3, figsize=(9, 6.2))
shown = 0
for m in test_maps:
    if shown == 3: break
    pt, ok = trace(nav_agent, m)
    if not ok or len(pt) < 8: continue
    pr, okr = trace(None, m)
    for row, (path, lab) in enumerate([(pr, 'Random policy'), (pt, 'Learned policy')]):
        a_ = ax[row, shown]; a_.imshow(m[0], cmap='Greys', origin='lower', vmin=0, vmax=1.6)
        a_.plot([p[0] for p in path], [p[1] for p in path], '-', color='C0', lw=1.5)
        a_.scatter(*m[1], c='green', s=80, marker='o', zorder=3); a_.scatter(*m[2], c='red', s=110, marker='*', zorder=3)
        a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
        a_.set_title(f'{lab}\n{len(path) - 1} steps' + ('' if (row == 1 or okr) else ' (goal not reached)'), fontsize=9)
    shown += 1
fig.text(.5, .005, 'Green circle = start, red star = goal, black = obstacle', ha='center', fontsize=8)
plt.tight_layout(); plt.savefig(OUT + 'fig_nav_paths.png', dpi=200); plt.close()

# summary bar chart
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].bar(['Random', 'Learned'], [RES['crawler']['random_mean'], RES['crawler']['trained_mean']],
          yerr=[RES['crawler']['random_std'], RES['crawler']['trained_std']], color=['C3', 'C0'], capsize=4)
ax[0].set(ylabel='Distance in 200 steps (units)', title='(a) Crawler'); 
ax[1].bar(['Random', 'Learned'], [res_random['success'], res_trained['success']], color=['C3', 'C0'])
ax[1].set(ylabel='Success on unseen maps (%)', title='(b) Navigation', ylim=(0, 105))
for i, v in enumerate([res_random['success'], res_trained['success']]): ax[1].text(i, v + 2, f'{v:.1f}', ha='center')
plt.tight_layout(); plt.savefig(OUT + 'fig_summary.png', dpi=200); plt.close()

json.dump(RES, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json'), 'w'), indent=1)
print(json.dumps(RES, indent=1))
