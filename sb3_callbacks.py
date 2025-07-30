# %% Imports
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import csv
# %% Callback and environment definition
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1, do_saves=True):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.do_saves = do_saves

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if the callback should be called
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if self.do_saves and mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


class CsvLoggerCallback(BaseCallback):
    """
    Custom callback for logging training metrics to a CSV file.
    Logs: timestep, episode_reward, episode_length, loss (if available), etc.
    """
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "training_metrics.csv")
        # Create an empty CSV file (truncate if exists)
        with open(self.csv_path, "w", newline="") as csvfile:
            pass
        self.header_written = False

    def _on_step(self) -> bool:
        # Only log at the end of an episode
        if len(self.locals.get("infos", [])) > 0 and "episode" in self.locals["infos"][0]:
            info = self.locals["infos"][0]["episode"]
            row = {
                "timesteps": self.num_timesteps,
                "reward": info["r"],
                "length": info["l"],
            }
            # Try to log loss if available (for PPO, loss is not directly exposed)
            # Try to log additional metrics if available
            # Policy loss
            if hasattr(self.model, "logger") and "train/loss" in self.model.logger.name_to_value:
                row["loss"] = self.model.logger.name_to_value["train/loss"]
            else:
                row["loss"] = None
            if hasattr(self.model, "logger") and "train/policy_loss" in self.model.logger.name_to_value:
                row["policy_loss"] = self.model.logger.name_to_value["train/policy_loss"]
            else:
                row["policy_loss"] = None

            # Value loss
            if hasattr(self.model, "logger") and "train/value_loss" in self.model.logger.name_to_value:
                row["value_loss"] = self.model.logger.name_to_value["train/value_loss"]
            else:
                row["value_loss"] = None

            # Entropy
            if hasattr(self.model, "logger") and "train/entropy" in self.model.logger.name_to_value:
                row["entropy"] = self.model.logger.name_to_value["train/entropy"]
            else:
                row["entropy"] = None

            # Learning rate
            if hasattr(self.model, "logger") and "train/learning_rate" in self.model.logger.name_to_value:
                row["learning_rate"] = self.model.logger.name_to_value["train/learning_rate"]
            else:
                row["learning_rate"] = None

            # Explained variance
            if hasattr(self.model, "logger") and "train/explained_variance" in self.model.logger.name_to_value:
                row["explained_variance"] = self.model.logger.name_to_value["train/explained_variance"]
            else:
                row["explained_variance"] = None

            # Clip fraction (PPO specific)
            if hasattr(self.model, "logger") and "train/clip_fraction" in self.model.logger.name_to_value:
                row["clip_fraction"] = self.model.logger.name_to_value["train/clip_fraction"]
            else:
                row["clip_fraction"] = None

            # Number of updates
            if hasattr(self.model, "logger") and "train/n_updates" in self.model.logger.name_to_value:
                row["n_updates"] = self.model.logger.name_to_value["train/n_updates"]
            else:
                row["n_updates"] = None

            # Success rate (if available in info)
            if "success" in info:
                row["success_rate"] = info["success"]
            else:
                row["success_rate"] = None

            # Action distribution statistics (mean, std)
            if hasattr(self.model, "logger") and "train/action_mean" in self.model.logger.name_to_value:
                row["action_mean"] = self.model.logger.name_to_value["train/action_mean"]
            else:
                row["action_mean"] = None
            if hasattr(self.model, "logger") and "train/action_std" in self.model.logger.name_to_value:
                row["action_std"] = self.model.logger.name_to_value["train/action_std"]
            else:
                row["action_std"] = None

            # Observation statistics (mean, std, min, max)
            if hasattr(self.model, "logger") and "train/obs_mean" in self.model.logger.name_to_value:
                row["obs_mean"] = self.model.logger.name_to_value["train/obs_mean"]
            else:
                row["obs_mean"] = None
            if hasattr(self.model, "logger") and "train/obs_std" in self.model.logger.name_to_value:
                row["obs_std"] = self.model.logger.name_to_value["train/obs_std"]
            else:
                row["obs_std"] = None
            if hasattr(self.model, "logger") and "train/obs_min" in self.model.logger.name_to_value:
                row["obs_min"] = self.model.logger.name_to_value["train/obs_min"]
            else:
                row["obs_min"] = None
            if hasattr(self.model, "logger") and "train/obs_max" in self.model.logger.name_to_value:
                row["obs_max"] = self.model.logger.name_to_value["train/obs_max"]
            else:
                row["obs_max"] = None

            # Custom environment info (e.g., distance to goal, collisions, etc.)
            if "distance_to_goal" in info:
                row["distance_to_goal"] = info["distance_to_goal"]
            else:
                row["distance_to_goal"] = None
            if "collisions" in info:
                row["collisions"] = info["collisions"]
            else:
                row["collisions"] = None

            # Write to CSV
            with open(self.csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                if not self.header_written:
                    writer.writeheader()
                    self.header_written = True
                writer.writerow(row)
        return True


