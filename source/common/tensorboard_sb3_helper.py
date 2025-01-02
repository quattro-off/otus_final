import os, shutil
from typing import Any, Dict, Tuple, Union

import numpy as np
import gymnasium
from PIL import Image

from datetime import datetime
from pytz import timezone
TZ = timezone('Europe/Moscow')

from stable_baselines3.common.logger import KVWriter, Logger, configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from moviepy.editor import ImageSequenceClip

import sys
import os


sys.path.append(os.path.abspath('../env'))

from env_simple_move import HumanMoveSimpleAction

class TensorboardCallback(BaseCallback):

    def __init__(
        self,
        eval_env: gymnasium.Env,
        save_freq: int,
        save_path: str,
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        save_log:int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        self.save_log = save_log
        #self._deterministic = deterministic

    def _checkpoint_path(self, checkpoint_type: str = "") -> str:
        return f'{checkpoint_type}{self.save_path}/{self.n_calls}'

    def _on_step(self) -> bool:

        #if self.n_calls % self.save_freq == 0:
        #    env_act = HumanMoveSimpleAction(self.model.get_env())
        #    sum_rewards = env_act.get_sum_rewards()
        #    for k, reward in sum_rewards.items():
        #        self.model.logger.record(f"rollout_rew/{k}", reward)

        if self.n_calls % self.save_freq == 0:

            save_path = self._checkpoint_path()
            model_path = save_path + '/model.zip'
            self.model.save(model_path)

            if self.verbose >= 2:
                print(f"Saving model checkpoint to {save_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too

                replay_buffer_path = save_path + "/replay_buffer.pkl"
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = save_path + "/vecnormalize.pkl"
                self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True


def VideoRecord(model, eval_env: gymnasium.Env, path: str, file_name: str, n_episodes: int = 1, fps:int = 30):
    
    #Video
    screens = []
    def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
        screen = eval_env.render()
        screens.append(screen)

    evaluate_policy(
        model,
        eval_env,
        callback=grab_screens,
        n_eval_episodes=n_episodes,
        deterministic=True,
    )
    
    if len(screens) > 1:
        video_path = path + '/' + file_name + '.mp4'
        clip = ImageSequenceClip(screens[:-1], fps=fps)
        clip.write_videofile(video_path)
        im_path = path + '/' + file_name + '.jpg'
        im = Image.fromarray(screens[-2])
        im.save(im_path)
        return True

    return False

class TBexperiment():
    def __init__(self, name:str="", id:int=0):
        self.name = name
        self.experiment_id = id


class TensorboardHelper():

    free_temp = True

    def __init__(self, tracking_uri: str, free_temp: bool = True):
        self.mlflow_tracking_uri = tracking_uri
        self.free_temp = free_temp
        #mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.experiment = TBexperiment("name", int( datetime.now(TZ).strftime("%H%M") ) )

    def new_experiment(self, experiment_name: str):
    #    experiment = mlflow.set_experiment(experiment_name)
    #    return experiment.experiment_id
        self.experiment.name = experiment_name
        return self.experiment.experiment_id
    
    def get_experiment(self, experiment_id: int):
    #    experiment = mlflow.get_experiment(experiment_id)
    #    return experiment.name
        self.experiment.experiment_id = experiment_id
        return self.experiment.name
    
    def load_artifact(self, artifact_uri: str, local_path: str):
    #    mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=local_path)
        return

    def learn_and_fix(self, model, env, 
                      run_name: str, 
                      episode_count: int, 
                      parameters: Dict, 
                      experiment_id: int, 
                      experiment_name: str = "", 
                      checkpoint_interval: int = 0, 
                      log_interval: int = 10,
                      video_episods:int = 3,
                      video_fps:int = 30
                      ):


        if experiment_name != "":
            self.experiment.name = experiment_name
        
        self.experiment.experiment_id = experiment_id

        folder = self.experiment.name + '/' + run_name

        save_callback = None
        if checkpoint_interval > 0 :
            save_callback = TensorboardCallback(
                eval_env=env,
                save_freq=checkpoint_interval,
                save_path=folder,
                save_log=log_interval,
                verbose=0,
            )

            
        loggers = configure(folder, ["tensorboard"])
            
        model.set_logger(loggers)
        model.learn(total_timesteps=episode_count,
                        log_interval=log_interval,
                        callback = save_callback,
                        progress_bar=True
                        )
            
        model.save(folder + '/model.zip')
            
        #Video
        VideoRecord(model, env, folder, '/agent', video_episods, video_fps)

        return "", folder, 1