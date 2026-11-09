import json

import pytest

from mountaincar.study import atomic_json, choose, digest, valid_record


def test_selection_uses_validation_gate_and_tiebreak():
    choices = [dict(repeat=1, passes=False, mean_best_reward=100, mean_best_success=1),
               dict(repeat=20, passes=True, mean_best_reward=3, mean_best_success=.9),
               dict(repeat=5, passes=True, mean_best_reward=3, mean_best_success=.9)]
    assert choose(choices)['repeat'] == 5
    with pytest.raises(ValueError):
        choose(choices[:1])


def test_resume_requires_matching_spec_and_untampered_files(tmp_path):
    names = ('run_summary.json', 'best.pt', 'final.pt', 'validation.csv',
             'stochastic_validation.csv', 'training.csv', 'episodes.csv', 'budget.json')
    for name in names:
        (tmp_path / name).write_text(name, encoding='utf-8')
    spec = {'seed': 1}
    atomic_json(tmp_path / 'completed.json', {'spec': spec, 'files': {n: digest(tmp_path / n) for n in names}})
    assert valid_record(tmp_path, spec)
    assert not valid_record(tmp_path, {'seed': 2})
    (tmp_path / 'best.pt').write_text('changed', encoding='utf-8')
    assert not valid_record(tmp_path, spec)
