import numpy as np
import optuna

import yaml

data_x = np.linspace(0, 1, 1000)
data_y: np.ndarray = 2 * data_x + 5 + np.random.randn(1000) * 0.1


def objective(trial: optuna.Trial) -> float:
    # Get some parameters
    a = trial.suggest_uniform("a", -10, 10)
    b = trial.suggest_uniform("b", -10, 10)

    # Calculate the loss
    loss: np.ndarray = np.mean((a * data_x + b - data_y) ** 2)

    return float(loss)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)

    with open("study.yaml", "w") as f:
        yaml.dump(study.trials_dataframe().to_dict(), f)
