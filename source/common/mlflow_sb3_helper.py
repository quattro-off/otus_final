import os, shutil
from typing import Any, Dict, Tuple, Union

import mlflow
import numpy as np
import gymnasium

from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from moviepy.editor import ImageSequenceClip

class MLflowCheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        eval_env: gymnasium.Env,
        save_freq: int,
        save_path: str,
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        #self._deterministic = deterministic

    def _checkpoint_path(self, checkpoint_type: str = "") -> str:
        return f'{checkpoint_type}{self.save_path}/{self.n_calls}'

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:

            save_path = self._checkpoint_path()
            model_path = save_path + '/model.zip'
            self.model.save(model_path)
            mlflow.log_artifact(model_path, save_path + '/sb3')

            if self.verbose >= 2:
                print(f"Saving model checkpoint to {save_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too

                replay_buffer_path = save_path + "/replay_buffer.pkl"
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                mlflow.log_artifact(replay_buffer_path, save_path + '/replay_buffer')
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = save_path + "/vecnormalize.pkl"
                self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                mlflow.log_artifact(vec_normalize_path, save_path + '/vecnormalize')
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


def VideoRecord(model, eval_env: gymnasium.Env, path: str, file_name: str):
    
    #Video
    screens = []
    def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        screen = eval_env.render()
        screens.append(screen)

    evaluate_policy(
        model,
        eval_env,
        callback=grab_screens,
        n_eval_episodes=1,
        deterministic=True,
    )
    
    if len(screens) > 1:
        video_path = path + '/' + file_name
        clip = ImageSequenceClip(screens, fps=30)
        clip.write_videofile(video_path)
        return True

    return False


class MLflowServerHelper():

    def __init__(self, mlflow_tracking_uri: str):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

    def new_experiment(self, experiment_name: str):
        experiment = mlflow.set_experiment(experiment_name)
        return experiment.experiment_id
    
    def load_artifact(self, artifact_uri: str, local_path: str):
        mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=local_path)

    def learn_and_fix(self, model, env, run_name: str, episode_count: int, parameters: Dict, experiment_id: int, experiment_name: str = "", checkpoint_interval: int = 0, log_interval: int = 10):

        experiment = {}
        if experiment_name == "":
            experiment = mlflow.get_experiment(experiment_id)
        else:
            experiment = mlflow.set_experiment(experiment_name)

        save_callback = None
        if checkpoint_interval > 0 :
            save_callback = MLflowCheckpointCallback(
                eval_env=env,
                save_freq=checkpoint_interval,
                save_path=experiment.name,
                verbose=0,
            )

        with mlflow.start_run(run_name=run_name) as run:
            
            mlflow.log_dict(parameters, experiment.name + '/parameters.json')
            mlflow.log_params(parameters)
            
            loggers = Logger(
                        folder=None,
                        output_formats=[MLflowOutputFormat()],
                    )
            
            model.set_logger(loggers)
            model.learn(total_timesteps=episode_count,
                        log_interval=log_interval,
                        callback = save_callback,
                        progress_bar=True
                        )
            
            model.save(experiment.name + '/model.zip')
            
            #mlflow.pytorch.log_model(model.actor, experiment.name + '/actor')

            mlflow.log_artifact(experiment.name + '/model.zip', experiment.name + '/sb3')

            #Video
            if VideoRecord(model, env, experiment.name, '/agent.mp4') == True:
                mlflow.log_artifact(experiment.name  + '/agent.mp4', experiment.name  + '/video')


            shutil.rmtree(os.path.join(experiment.name))

        return experiment.artifact_location, experiment.name, run.info.run_id