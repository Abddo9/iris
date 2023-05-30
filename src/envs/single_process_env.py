from typing import Any, Tuple
import torch
import numpy as np

from .done_tracker import DoneTrackerEnv


class SingleProcessEnv(DoneTrackerEnv):
    def __init__(self, env_fn):
        super().__init__(num_envs=1)
        self.env = env_fn()
        self.num_actions = self.env.action_space[0].n

    def should_reset(self) -> bool:
        return self.num_envs_done == 1

    def reset(self) -> np.ndarray:
        self.reset_done_tracker()
        obs = self.env.reset()
        return obs[None, ...]

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)  
        if torch.is_tensor(done):
            done = done.cpu().detach().numpy()     # 1 True/False
        else:
            done = np.array([done])
        #obs = np.array([ob.cpu().detach().numpy().tolist() for ob in obs], dtype=float)
        self.update_done_tracker(done)
        return obs[None, ...], reward, done, None

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
