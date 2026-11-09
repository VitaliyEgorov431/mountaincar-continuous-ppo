"""Reproduce the seed-4 failure and evaluate final policies on new starts."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .agent import Agent
from .evaluation import action_repeat, choose_action
from .controls import step_action
from .recording import record

OUT = Path('results/improvements/diagnostics')


def metrics(rows):
    lengths = [r['length'] for r in rows]
    return {'mean_reward': float(np.mean([r['reward'] for r in rows])),
            'success_rate': float(np.mean([r['success'] for r in rows])),
            'mean_length': float(np.mean(lengths)), 'mean_action_energy': float(np.mean([r['action_energy'] for r in rows])), 'median_length': float(np.median(lengths)), 'p95_length': float(np.percentile(lengths, 95)),
            'timeout_count': sum(not r['success'] for r in rows), 'environment_steps': sum(lengths)}


def intervals(rows):
    """Return runs of failed sampled positions; boundaries are grid points, not continuous thresholds."""
    runs, run = [], []
    for row in rows:
        if not row['success']:
            run.append(row['initial_position'])
        elif run:
            runs.append([run[0], run[-1], len(run)])
            run = []
    if run:
        runs.append([run[0], run[-1], len(run)])
    return runs


def episodes(policy, seeds, positions=None):
    rows = []
    for start in range(0, len(seeds), 20):
        batch = seeds[start:start+20]
        envs = [gym.make('MountainCarContinuous-v0') for _ in batch]
        try:
            obs = [env.reset(seed=int(seed))[0] for env, seed in zip(envs, batch)]
            if positions is not None:
                for i, env in enumerate(envs):
                    env.unwrapped.state = np.array([positions[start+i], 0.0], dtype=np.float64)
                    obs[i] = np.array(env.unwrapped.state, dtype=np.float32)
            rngs = [np.random.default_rng(seed) for seed in batch]
            batch_rows = [dict(seed=int(seed), initial_position=float(o[0]), reward=0., success=False, length=0, action_energy=0.) for seed, o in zip(batch, obs)]
            active = list(range(len(batch)))
            while active:
                if isinstance(policy, str):
                    actions = [choose_action(policy, obs[i], rngs[i]) for i in active]
                else:
                    with torch.no_grad():
                        actions, _, _ = policy.act(torch.as_tensor(np.stack([obs[i] for i in active]), dtype=torch.float32), deterministic=True)
                    actions = np.clip(actions.numpy(), -1, 1)
                remaining = []
                for i, action in zip(active, actions):
                    row = batch_rows[i]
                    obs[i], _, terminated, truncated, info = step_action(envs[i], action, action_repeat(policy), max_steps=999-row['length'])
                    row['reward'] += info['raw_reward']
                    row['action_energy'] += info['action_energy']
                    row['length'] += info['duration']
                    row['success'] = bool(terminated)
                    if not (terminated or truncated) and row['length'] < 999:
                        remaining.append(i)
                active = remaining
            rows.extend(batch_rows)
        finally:
            for env in envs:
                env.close()
    return rows


def trace(policy, seed):
    env = gym.make('MountainCarContinuous-v0')
    rng = np.random.default_rng(seed)
    rows = []
    try:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done and len(rows) < 999:
            action = choose_action(policy, obs, rng)
            for _ in range(min(action_repeat(policy), 999-len(rows))):
                before = obs.copy()
                obs, reward, terminated, truncated, _ = env.step(action)
                rows.append(dict(step=len(rows)+1, position=float(before[0]), speed=float(before[1]), action=float(action[0]), next_position=float(obs[0]), next_speed=float(obs[1]), reward=float(reward), terminated=bool(terminated), truncated=bool(truncated)))
                done = terminated or truncated
                if done:
                    break
    finally:
        env.close()
    return rows


def trace_stats(rows):
    actions = np.array([float(row['action']) for row in rows])
    speeds = [float(row['speed']) for row in rows] + [float(rows[-1]['next_speed'])]
    signs = [np.sign(speed) for speed in speeds if speed != 0]
    positions = [float(row['position']) for row in rows] + [float(rows[-1]['next_position'])]
    return {'action_abs_max': float(np.max(np.abs(actions))),
            'action_rms': float(np.sqrt(np.mean(actions * actions))),
            'speed_sign_switches': sum(int(a != b) for a, b in zip(signs, signs[1:])),
            'position_min': min(positions), 'position_max': max(positions)}


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path('results/revised/final').glob('seed_*/best.pt'))
    if len(paths) != 5:
        raise ValueError(f'Expected five final checkpoints, found {len(paths)}')
    models = {p.parent.name: Agent.load(p) for p in paths}
    models.update({'random_hold': 'random_hold', 'heuristic': 'heuristic'})
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    summary = {'provenance': {'python': platform.python_version(), 'platform': platform.platform(), 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('requirements.txt'), Path('mountaincar/agent.py'), Path('mountaincar/controls.py'), Path('mountaincar/evaluation.py'), Path('mountaincar/recording.py'), Path('mountaincar/diagnostics.py')]}, 'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'torch', 'gymnasium', 'matplotlib', 'imageio', 'imageio-ffmpeg', 'pygame-ce']}, 'checkpoint_sha256': hashes, 'seed_range': [30000, 30999], 'grid': {'min': -0.6, 'max': -0.4, 'points': 201, 'speed': 0.0, 'rng_seed_per_point': '30000 + index'}, 'max_steps': 999, 'policy': 'deterministic, clipped to [-1, 1]', 'random_hold_repeat': 20, 'torch_threads': 1}, 'results': {}}
    grid = np.linspace(-0.6, -0.4, 201)
    for name, policy in models.items():
        print('Evaluating', name, flush=True)
        random_rows = episodes(policy, list(range(30000, 31000)))
        grid_rows = episodes(policy, list(range(30000, 30201)), grid)
        write_csv(OUT / f'{name}_new_seeds.csv', random_rows)
        write_csv(OUT / f'{name}_grid.csv', grid_rows)
        summary['results'][name] = {'new_seeds': metrics(random_rows), 'grid': metrics(grid_rows), 'grid_failure_intervals': intervals(grid_rows)}
    for seed in [20047, 20046]:
        rows = trace(models['seed_4'], seed)
        write_csv(OUT / f'seed_4_trace_{seed}.csv', rows)
        record(models['seed_4'], OUT / f'seed_4_{seed}.mp4', seed)
        summary.setdefault('traces', {})[str(seed)] = {'length': len(rows), 'success': rows[-1]['terminated'], 'initial_position': rows[0]['position'], 'reward': sum(r['reward'] for r in rows), **trace_stats(rows)}
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for seed in [20047, 20046]:
        with (OUT / f'seed_4_trace_{seed}.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        for ax, key in zip(axes, ['position', 'speed', 'action']):
            ax.plot([int(r['step']) for r in rows], [float(r[key]) for r in rows], label=str(seed))
            ax.set_ylabel(key)
    axes[0].legend(); axes[-1].set_xlabel('Raw environment step'); fig.tight_layout()
    fig.savefig(OUT / 'seed_4_trace_comparison.png', dpi=150); plt.close(fig)
    summary['evaluation_environment_steps'] = sum(v[k]['environment_steps'] for v in summary['results'].values() for k in ['new_seeds', 'grid'])
    summary['trace_environment_steps'] = sum(x['length'] for x in summary['traces'].values())
    summary['video_environment_steps'] = summary['trace_environment_steps']
    summary['elapsed_seconds'] = time.monotonic()-started
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()



