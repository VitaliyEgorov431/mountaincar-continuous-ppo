"""Validation-gated action-repeat study, with atomic completion records."""
import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .agent import Agent
from . import evaluation
from .train import DEFAULT_CONFIG, VALIDATION_SEEDS, train

REPEATS = (1, 5, 10, 20)
PILOT_SEEDS = (0, 11, 12)
FINAL_SEEDS = (1, 2, 3, 4, 5)
HELDOUT_SEEDS = tuple(range(40000, 41000))
BUDGET = 200_000
ROOT = Path('results/improvements/study')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(temp, path)


def provenance():
    package = Path(__file__).parent
    return {'python': platform.python_version(), 'gymnasium': gym.__version__,
            'numpy': np.__version__, 'torch': str(torch.__version__),
            'requirements_sha256': digest(package.parent / 'requirements.txt'),
            'sources': {name: digest(package / name) for name in
                        ('study.py', 'train.py', 'evaluation.py', 'agent.py', 'controls.py')}}


def config(repeat):
    return {**DEFAULT_CONFIG, 'total_timesteps': BUDGET, 'action_repeat': repeat, 'eval_every_updates': max(1, round(40 / repeat))}


def valid_record(directory, expected):
    marker = directory / 'completed.json'
    if not marker.exists():
        return False
    try:
        saved = json.loads(marker.read_text(encoding='utf-8'))
        return saved['spec'] == expected and all(digest(directory / name) == value
                    for name, value in saved['files'].items()) and set(saved['files']) == {
                        'run_summary.json', 'best.pt', 'final.pt', 'validation.csv',
                        'stochastic_validation.csv', 'training.csv', 'episodes.csv', 'budget.json'}
    except (ValueError, KeyError, OSError, AttributeError, TypeError):
        return False


def run_one(repeat, seed, phase, resume):
    directory = ROOT / phase / f'repeat_{repeat}' / f'seed_{seed}'
    spec = {'config': config(repeat), 'seed': seed, 'validation_seeds': VALIDATION_SEEDS,
            'provenance': provenance()}
    if resume and valid_record(directory, spec):
        return json.loads((directory / 'run_summary.json').read_text(encoding='utf-8'))
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f'Incomplete or mismatched run: {directory}; preserve it and use a fresh output location')
    summary = train(config(repeat), seed, directory)
    if summary['steps'] != BUDGET:
        raise ValueError('Training did not consume the original-step budget')
    validation_steps = {}
    for name in ('validation.csv', 'stochastic_validation.csv'):
        with (directory / name).open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        validation_steps[name] = round(sum(float(row['mean_length']) * len(VALIDATION_SEEDS) for row in rows))
    with (directory / 'training.csv').open(newline='', encoding='utf-8') as f:
        updates = sum(1 for _ in csv.DictReader(f))
    atomic_json(directory / 'budget.json', {'training_original_steps': summary['steps'],
                'training_decisions': summary['decisions'], 'updates': updates,
                'validation_original_steps': validation_steps})
    names = ('run_summary.json', 'best.pt', 'final.pt', 'validation.csv',
             'stochastic_validation.csv', 'training.csv', 'episodes.csv', 'budget.json')
    atomic_json(directory / 'completed.json', {'spec': spec, 'files': {n: digest(directory / n) for n in names}})
    return summary


def checked(repeat, seed, phase):
    directory = ROOT / phase / f'repeat_{repeat}' / f'seed_{seed}'
    spec = {'config': config(repeat), 'seed': seed, 'validation_seeds': VALIDATION_SEEDS,
            'provenance': provenance()}
    if not valid_record(directory, spec):
        raise ValueError(f'Missing or mismatched completed run: {directory}')
    return json.loads((directory / 'run_summary.json').read_text(encoding='utf-8'))


def passes(directory):
    with (directory / 'validation.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return any(all(float(row['success_rate']) >= .9 for row in rows[i:i + 3])
               for i in range(len(rows) - 2))


def choose(candidates):
    eligible = [item for item in candidates if item['passes']]
    if not eligible:
        raise ValueError('No repeat passed validation on all three pilot seeds')
    return max(eligible, key=lambda x: (x['mean_best_reward'], x['mean_best_success'], -x['repeat']))


def select():
    candidates = []
    for repeat in REPEATS:
        summaries = [checked(repeat, seed, 'pilot') for seed in PILOT_SEEDS]
        candidates.append({'repeat': repeat,
            'passes': all(passes(ROOT / 'pilot' / f'repeat_{repeat}' / f'seed_{s}') for s in PILOT_SEEDS),
            'mean_best_reward': float(np.mean([s['best_mean_reward'] for s in summaries])),
            'mean_best_success': float(np.mean([s['best_success_rate'] for s in summaries]))})
    selected = choose(candidates)
    result = {'candidates': candidates, 'selected_repeat': selected['repeat'],
              'rule': 'All three runs must pass three consecutive validation success rates >= 0.9; maximize mean best deterministic validation reward, then success, then prefer lower repeat.',
              'provenance': provenance()}
    atomic_json(ROOT / 'selection.json', result)
    return result


def read_selection():
    selection = json.loads((ROOT / 'selection.json').read_text(encoding='utf-8'))
    if selection['provenance'] != provenance():
        raise ValueError('Selection provenance changed')
    return selection


def freeze():
    selection = read_selection()
    repeats = sorted({selection['selected_repeat'], 20})
    chosen = {}
    for repeat in repeats:
        summaries = [checked(repeat, seed, 'final') for seed in FINAL_SEEDS]
        best = max(summaries, key=lambda x: (x['best_mean_reward'], x['best_success_rate'], -x['best_step']))
        path = ROOT / 'final' / f'repeat_{repeat}' / f"seed_{best['seed']}" / 'best.pt'
        models = []
        for summary in summaries:
            model_path = ROOT / 'final' / f'repeat_{repeat}' / f"seed_{summary['seed']}" / 'best.pt'
            models.append({'seed': summary['seed'], 'checkpoint': str(model_path),
                           'sha256': digest(model_path), 'validation_reward': summary['best_mean_reward'],
                           'validation_success': summary['best_success_rate']})
        chosen[str(repeat)] = {'selected_seed': best['seed'], 'models': models}
    result = {'selection_sha256': digest(ROOT / 'selection.json'), 'models': chosen,
              'provenance': provenance(), 'heldout_seeds': list(HELDOUT_SEEDS)}
    atomic_json(ROOT / 'frozen.json', result)
    return result


def evaluate():
    frozen = json.loads((ROOT / 'frozen.json').read_text(encoding='utf-8'))
    if frozen['provenance'] != provenance() or frozen['selection_sha256'] != digest(ROOT / 'selection.json'):
        raise ValueError('Frozen provenance changed')
    results = {}
    original = evaluation.action_repeat
    try:
        for repeat, group in frozen['models'].items():
            for model in group['models']:
                if digest(model['checkpoint']) != model['sha256']:
                    raise ValueError('Frozen checkpoint changed')
                results[f"ppo_repeat_{repeat}_seed_{model['seed']}"] = evaluation.evaluate_policy(Agent.load(model['checkpoint']), HELDOUT_SEEDS)
        results['heuristic'] = evaluation.evaluate_policy('heuristic', HELDOUT_SEEDS)
        for repeat in REPEATS:
            evaluation.action_repeat = lambda policy, n=repeat: n if policy == 'random_hold' else original(policy)
            results[f'random_hold_repeat_{repeat}'] = evaluation.evaluate_policy('random_hold', HELDOUT_SEEDS)
    finally:
        evaluation.action_repeat = original
    aggregates = {}
    for repeat, group in frozen['models'].items():
        summaries = [results[f"ppo_repeat_{repeat}_seed_{m['seed']}"]['summary'] for m in group['models']]
        aggregates[repeat] = {key: {'mean': float(np.mean([s[key] for s in summaries])),
                                     'sd_across_training_seeds': float(np.std([s[key] for s in summaries], ddof=1))}
                              for key in summaries[0]}
        aggregates[repeat]['selected_seed'] = group['selected_seed']
    atomic_json(ROOT / 'heldout.json', {'frozen_sha256': digest(ROOT / 'frozen.json'), 'aggregates': aggregates, 'results': results})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pilot', 'select', 'final', 'freeze', 'evaluate'))
    parser.add_argument('--repeat', type=int, choices=REPEATS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase in ('pilot', 'final'):
        repeats = REPEATS if args.phase == 'pilot' else sorted({read_selection()['selected_repeat'], 20})
        seeds = PILOT_SEEDS if args.phase == 'pilot' else FINAL_SEEDS
        if args.repeat is not None:
            if args.repeat not in repeats: parser.error('repeat not in phase')
            repeats = [args.repeat]
        if args.seed is not None:
            if args.seed not in seeds: parser.error('seed not in phase')
            seeds = [args.seed]
        for repeat in repeats:
            for seed in seeds:
                run_one(repeat, seed, args.phase, args.resume)
    else:
        {'select': select, 'freeze': freeze, 'evaluate': evaluate}[args.phase]()


if __name__ == '__main__':
    main()


