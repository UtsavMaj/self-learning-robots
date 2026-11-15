# Self-Learning Robots: Deep Reinforcement Learning for Walking and Navigation

Case study for **21CSE424T – Deep Learning for Data Analytics** (SRM Institute of Science and Technology).

**Authors:** Utsav Majumdar (RA2311056010143), Shirin Gupta (RA2311056010113)

A pure NumPy implementation (no deep-learning framework needed) of:

1. **Exploration vs exploitation** – Naïve, ε-Greedy and Upper-Bounding algorithms on a 10-armed bandit.
2. **Crawling robot (walking)** – a two-joint robot that learns to crawl forward using a Double DQN.
3. **Navigation** – a robot that learns to reach a goal on unseen random obstacle maps (10 × 10 grid) using a Double DQN.

## Run

```bash
pip install -r requirements.txt
python rl_robots.py
```

Figures are written to `figures/` and the measured numbers to `results.json`. A full run takes roughly 10–15 minutes on one CPU core.

## Results (from `results.json`)

| Task | Random policy | Learned policy |
|---|---|---|
| Crawler distance in 200 steps | 0.11 | 6.28 (best possible 7.27) |
| Navigation success on 200 unseen maps | 7.0% | 43.5% |
| Navigation collisions per episode | 17.3 | 0.3 |

## Files

- `rl_robots.py` – all experiments, the NumPy DQN and the plotting code
- `results.json` – measured results used in the report
- `figures/` – learning curves, gait and path plots

Note: the crawler is a simplified kinematic simulation, not a full physics-based walker. Exact numbers can vary slightly with library versions.
