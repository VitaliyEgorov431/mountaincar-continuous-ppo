from mountaincar.diagnostics import intervals, metrics, trace, trace_stats


def test_metrics_and_grid_interval_boundaries():
    rows = [{'initial_position': x, 'success': s, 'reward': r, 'length': n, 'action_energy': 0.0} for x, s, r, n in [(-.6, False, 0, 999), (-.5, True, 90, 100), (-.4, False, 0, 999)]]
    assert intervals(rows) == [[-.6, -.6, 1], [-.4, -.4, 1]]
    assert metrics(rows)['timeout_count'] == 2
    assert metrics(rows)['environment_steps'] == 2098


def test_trace_is_raw_steps_and_stops_at_limit():
    rows = trace('heuristic', 20046)
    assert [r['step'] for r in rows] == list(range(1, len(rows) + 1))
    assert len(rows) <= 999
    assert all(rows[i]['next_position'] == rows[i+1]['position'] for i in range(len(rows)-1))
    assert rows[-1]['terminated'] or rows[-1]['truncated'] or len(rows) == 999



def test_trace_stats_ignores_zero_and_includes_terminal_state():
    rows = [dict(position=0, speed=0, action=0.5, next_position=1, next_speed=-1),
            dict(position=1, speed=-1, action=-0.5, next_position=2, next_speed=0),
            dict(position=2, speed=0, action=0.5, next_position=3, next_speed=1)]
    result = trace_stats(rows)
    assert result['speed_sign_switches'] == 1
    assert result['position_min'] == 0 and result['position_max'] == 3
    assert result['action_rms'] == 0.5
