"""
Credits to https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

from typing import Tuple
import time

import gym
import numpy as np
from PIL import Image
from vmas import make_env
import torch
from env_wrappers import SubprocVecEnv, DummyVecEnv
from mpe.MPE_env import MPEEnv
import argparse

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def get_mpe_args(num_agents):
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=num_agents, help="number of players")
    parser.add_argument('--episode_length', type=int, default=25, help="Max length for any episode")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--env_name", type=str, default='MPE', help="specify the name of environment")
    parser.add_argument("--n_rollout_threads", type=int, default=1, help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1, help="Number of parallel envs for evaluating rollouts")
    args, unknown = parser.parse_known_args()
    return args


def make_atari(id, size=64, max_episode_steps=None, noop_max=30, frame_skip=4, 
               done_on_life_loss=False, clip_reward=False, device = 'cpu', num_agents=3):
    if 'vmas' in id :
        env = make_env( scenario_name= id.split('.')[1],
                        num_envs=1,
                        device=device,
                        continuous_actions=False,
                        wrapper=None,
                        max_steps= 200,
                        # Environment specific variables
                        n_agents=1,
                        penalise_by_time = True,
                        use_relative_obs = True,
                        use_distance_reward = True,
                        penalise_by_collision = True,
                        agent_info_first = True,
                        n_obstacles = 4,
                        food_reward = 15,
                        collision_penalty = -10,
                        time_penalty = -0.5,
                        
                        t_positive_progress_coeff = 1, t_negative_progress_coeff = 1,
                        c_positive_progress_coeff = 0.1, c_negative_progress_coeff = 0.1,
                        use_distance_reciprocal = False,
                        collision_reward_range = 0.3,
                        use_clear_target = True)
        env = VmasWrapper(env)
    elif 'mpe' in id:
        env = make_train_env(get_mpe_args(num_agents)) 
        env = MPEWrapper(env)
    else:
        env = gym.make(id)
        assert 'NoFrameskip' in env.spec.id or 'Frameskip' not in env.spec
    env = ResizeObsWrapper(env, (size, size))
    if clip_reward:
        env = RewardClippingWrapper(env)
    if max_episode_steps is not None and 'vmas' not in id:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if noop_max is not None and 'vmas' not in id  and 'mpe' not in id:
        env = NoopResetEnv(env, noop_max=noop_max)
    if 'vmas' not in id and 'mpe' not in id:
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if done_on_life_loss:
            env = EpisodicLifeEnv(env)
    return env


class ResizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: Tuple[int, int]) -> None:
        gym.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
        #self.unwrapped.original_obs = None

    def resize(self, obs: np.ndarray):
        images = []
        for ob in obs:
            img = Image.fromarray(ob)
            img = img.resize(self.size, Image.BILINEAR)
            images.append(np.array(img))
        return np.array(images)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        #self.unwrapped.original_obs = observation
        return self.resize(observation)

class VmasWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, rews, dones, info = self.env.step(action)
        obs2 = self.get_obs()
        return obs2, rews, dones, info
    
    def get_obs(self):
        return self.env.render(mode="rgb_array", agent_index_focus=None,visualize_when_rgb=True)
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs2 = self.get_obs()
        return obs2

class MPEWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        tmp_obs, rews, dones, info = self.env.step(action)
        obs = self.get_obs(tmp_obs.shape[1])
        return obs, rews, dones, info
    
    def get_obs(self, n_agents):
        img_obs = []
        for i in range(n_agents):
            obs = self.env.render(mode="rgb_array", agent_index_focus=i,
                        visualize_when_rgb=True)[0]
            img_obs.append(obs)
        return np.array(img_obs)
    
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self.get_obs(obs.shape[1])
        return obs

class RewardClippingWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        assert skip > 0
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        self.max_frame = self._obs_buffer.max(axis=0)

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
