"""Explicit, validation-gated experiment phases."""
import csv
import json
from pathlib import Path
import numpy as np
from .agent import Agent
from .evaluation import TEST_SEEDS, evaluate_policy
from .recording import record
from .train import DEFAULT_CONFIG, train


def _write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _run(config, seed, directory, resume):
    effective = {**DEFAULT_CONFIG, **config}
    marker = directory / "run_summary.json"
    if resume and marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if (saved.get("config") == effective and saved.get("seed") == seed and saved.get("steps") == effective["total_timesteps"] and (directory / "best.pt").exists() and (directory / "final.pt").exists()):
            return saved
    return train(effective, seed, directory)


def _passed(directory, config, seed):
    marker = directory / "run_summary.json"
    if not marker.exists() or not (directory / "best.pt").exists():
        return False
    summary = json.loads(marker.read_text(encoding="utf-8"))
    if summary.get("seed") != seed or summary.get("config") != {**DEFAULT_CONFIG, **config} or summary.get("steps") != config["total_timesteps"]:
        return False
    with (directory / "validation.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return any(all(float(row["success_rate"]) >= 0.9 for row in rows[i:i+3]) for i in range(len(rows)-2))


def run_experiment(config, output_dir, resume=False, phase="pilot"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {**DEFAULT_CONFIG, **config}
    pilot = output / "pilot" / "seed_0"
    if phase == "pilot":
        return _run(config, 0, pilot, resume)
    if config["total_timesteps"] != 200_000:
        raise ValueError("Confirmation and final phases require the frozen 200000-step budget")
    if not _passed(pilot, config, 0):
        raise ValueError("Pilot must reach 90% deterministic validation success in three consecutive checks")
    if phase == "confirm":
        return {str(seed): _run(config, seed, output / "confirm" / f"seed_{seed}", resume) for seed in (11, 12)}
    if not all(_passed(output / "confirm" / f"seed_{seed}", config, seed) for seed in (11, 12)):
        raise ValueError("Both confirmation seeds must pass three consecutive validation checks")
    if phase == "final":
        return {str(seed): _run(config, seed, output / "final" / f"seed_{seed}", resume) for seed in range(1, 6)}
    if phase != "report":
        raise ValueError(f"Unknown phase: {phase}")
    summaries = []
    for seed in range(1, 6):
        directory = output / "final" / f"seed_{seed}"
        marker = directory / "run_summary.json"
        if not marker.exists():
            raise ValueError(f"Missing completed final run: seed {seed}")
        item = json.loads(marker.read_text(encoding="utf-8"))
        if item.get("config") != config or item.get("steps") != config["total_timesteps"] or item.get("seed") != seed or not (directory / "best.pt").exists():
            raise ValueError(f"Final run does not match frozen configuration: seed {seed}")
        summaries.append(item)
    selected = max(summaries, key=lambda x: (x["best_mean_reward"], x["best_success_rate"], -x["best_step"]))
    results = {f"seed_{s['seed']}": evaluate_policy(Agent.load(output / "final" / f"seed_{s['seed']}" / "best.pt"), TEST_SEEDS) for s in summaries}
    for name in ("random", "random_hold", "heuristic"):
        results[name] = evaluate_policy(name, TEST_SEEDS)
    _write(output / "test_results.json", results)
    _write(output / "experiment_summary.json", {"finals": summaries, "selected_seed": selected["seed"], "config": config})
    with (output / "test_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["policy", "seed", "reward", "success", "length", "action_energy", "max_position", "max_speed"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            writer.writerows({"policy": name, **episode} for episode in result["episodes"])
    from .plots import plot_results
    plot_results(output)
    video_error = None
    try:
        record(Agent.load(output / "final" / f"seed_{selected['seed']}" / "best.pt"), output / "selected.mp4", seed=20000)
    except Exception as exc:
        video_error = str(exc)
    metrics = ("mean_reward", "success_rate", "mean_length", "mean_action_energy")
    ppo = [results[f"seed_{i}"]["summary"] for i in range(1, 6)]
    lines = ["# Эксперимент MountainCar", "", f"Выбранный seed обучения: {selected['seed']} (только по валидации).", "", "| Метод | Награда | Доля успехов | Длина | Энергия |", "| --- | ---: | ---: | ---: | ---: |"]
    lines.append("| PPO | " + " | ".join(f"{np.mean([v[k] for v in ppo]):.3f} ± {np.std([v[k] for v in ppo], ddof=1):.3f}" for k in metrics) + " |")
    for name in ("random", "random_hold", "heuristic"):
        lines.append(f"| {name} | " + " | ".join(f"{results[name]['summary'][k]:.3f}" for k in metrics) + " |")
    success = float(np.mean([v["success_rate"] for v in ppo]))
    target = "достигнута" if success >= 0.9 else "не достигнута"
    lines += ["", f"Цель 90% успеха по среднему пяти запусков {target}: {success:.1%}.", "", "Все методы оценены на одинаковых 100 тестовых seed. PPO удерживает действие согласно конфигурации, random и heuristic принимают решение на каждом шаге; random_hold удерживает случайное действие 20 шагов. Отклонение PPO рассчитано между пятью seed обучения, а не между эпизодами.", "", "Данные: test_episodes.csv и test_results.json. Графики: training_curves.png и comparison.png."]
    lines.append(f"Ошибка записи видео: {video_error}" if video_error else "Видео: selected.mp4.")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"selected_seed": selected["seed"], "video_error": video_error}
