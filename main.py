import hydra
from omegaconf import DictConfig


"""Entry point selecting between paper tasks.

We import task modules lazily inside main() to avoid importing heavy
dependencies (e.g., vLLM) when not needed.
"""

@hydra.main(config_path="configs", config_name="base")
def main(cfg: DictConfig):
    if cfg.debug:
        cfg.run_name = cfg.run_name + "_debug"
    task = cfg.task.name
    optimizer = getattr(cfg, "optimizer", "lse")

    # Keep GEPA logs separate under logs/gepa/...
    if optimizer == "gepa":
        cfg.log_dir = f"{cfg.log_dir}/gepa/{cfg.task.name}"
    else:
        cfg.log_dir = f"{cfg.log_dir}/{cfg.task.name}"

    # GEPA baseline (bird/mmlu only)
    if optimizer == "gepa":
        if task == "bird":
            from lse.gepa.gepa_bird_simulator import GEPABirdSimulator
            simulator = GEPABirdSimulator(cfg)
            results = simulator.run_gepa()
            print(f'Accuracy: {results["average_accuracy"]}')
            return
        if task == "mmlu":
            from lse.gepa.gepa_mmlu_simulator import GEPAMMLUSimulator
            simulator = GEPAMMLUSimulator(cfg)
            results = simulator.run_gepa()
            print(f'Accuracy: {results["average_accuracy"]}')
            return
        raise ValueError(f"GEPA optimizer not supported for task: {task}")

    if task == 'bird':
        from lse.simulators.bird_simulator import BirdSimulator
        simulator = BirdSimulator(cfg)
        results = simulator.run_self_evolve()
        print(f'Accuracy: {results["average_accuracy"]}')
    elif task == 'mmlu':
        from lse.simulators.mmlu_simulator import MMLUSimulator
        simulator = MMLUSimulator(cfg)  
        results = simulator.run_self_evolve()
        print(f'Accuracy: {results["average_accuracy"]}')
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
