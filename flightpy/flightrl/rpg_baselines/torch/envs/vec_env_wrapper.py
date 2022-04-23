import os
import pickle
from abc import ABC
from typing import Any, List, Type

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv,
                                                           VecEnvIndices)


def _unnormalize_obs(obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
    """
    Helper to unnormalize observation.
    :param obs:
    :param obs_rms: associated statistics
    :return: unnormalized observation
    """
    return (obs * np.sqrt(obs_rms.var + 1e-8)) + obs_rms.mean


def _normalize_obs(obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
    """
    Helper to normalize observation.
    :param obs:
    :param obs_rms: associated statistics
    :return: normalized observation
    """
    return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)


class FlightEnvVec(VecEnv, ABC):
    def __init__(self, impl):
        self.render_id = 0
        self.wrapper = impl
        self.var = None
        self.mean = None
        self.envs = None
        self._reward = None
        self.rgb_channel = 3  # rgb channel
        self.depth_channel = 1
        self.act_dim = self.wrapper.getActDim()
        self.obs_dim = self.wrapper.getObsDim()  # C++ obs shape

        self.rew_dim = self.wrapper.getRewDim()
        self.img_width = self.wrapper.getImgWidth()
        self.img_height = self.wrapper.getImgHeight()
        self._observation_space = spaces.Box(
                np.ones([self.rgb_channel + self.depth_channel, self.img_width, self.img_height]) * 0,
                np.ones([self.rgb_channel + self.depth_channel, self.img_width, self.img_height]) * 255,
                dtype=np.int,
        )
        self._action_space = spaces.Box(
                low=np.ones(self.act_dim) * -1.0,
                high=np.ones(self.act_dim) * 1.0,
                dtype=np.float64,
        )
        self._observation = np.zeros([self.num_envs, self.obs_dim], dtype=np.float64)
        self._rgb_img_obs = np.zeros(
                [self.num_envs, self.img_width * self.img_height * self.rgb_channel], dtype=np.uint8
        )
        self._gray_img_obs = np.zeros(
                [self.num_envs, self.img_width * self.img_height], dtype=np.uint8
        )
        self._depth_img_obs = np.zeros(
                [self.num_envs, self.img_width * self.img_height], dtype=np.float32
        )
        self.compactimage = self.getCompactImage()

        self._reward_components = np.zeros(
                [self.num_envs, self.rew_dim], dtype=np.float64
        )
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self.reward_names = self.wrapper.getRewardNames()
        self._extraInfo = np.zeros(
                [self.num_envs, len(self._extraInfoNames)], dtype=np.float64
        )

        self.rewards = [[] for _ in range(self.num_envs)]
        self.sum_reward_components = np.zeros(
                [self.num_envs, self.rew_dim - 1], dtype=np.float64
        )

        self._quadstate = np.zeros([self.num_envs, 25], dtype=np.float64)
        self._quadact = np.zeros([self.num_envs, 4], dtype=np.float64)
        self._flightmodes = np.zeros([self.num_envs, 1], dtype=np.float64)

        #  state normalization
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.obs_dim])
        self.obs_rms_new = RunningMeanStd(shape=[self.num_envs, self.obs_dim])

        self.max_episode_steps = 1000
        # VecEnv.__init__(self, self.num_envs,
        #                 self._observation_space, self._action_space)
        self.is_unity_connected = False

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def update_rms(self):
        self.obs_rms = self.obs_rms_new

    def teststep(self, action):
        self.wrapper.testStep(
                action,
                self._observation,
                self._reward_components,
                self._done,
                self._extraInfo,
        )
        obs = self.normalize_obs(self._observation)
        return (
                obs,
                self._reward_components[:, -1].copy(),
                self._done.copy(),
                self._extraInfo.copy(),
        )

    def step(self, action):
        if action.ndim <= 1:
            action = action.reshape((-1, self.act_dim))
        self.wrapper.step(
                action,
                self._observation,
                self._reward_components,
                self._done,
                self._extraInfo,
        )

        # update the mean and variance of the Running Mean STD
        self.obs_rms_new.update(self._observation)
        obs = self.normalize_obs(self._observation)

        if len(self._extraInfoNames) != 0:
            info = [
                    {
                            "extra_info": {
                                    self._extraInfoNames[j]: self._extraInfo[i, j]
                                    for j in range(0, len(self._extraInfoNames))
                            }
                    }
                    for i in range(self.num_envs)
            ]
        else:
            info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward_components[i, -1])
            for j in range(self.rew_dim - 1):
                self.sum_reward_components[i, j] += self._reward_components[i, j]
            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                for j in range(self.rew_dim - 1):
                    epinfo[self.reward_names[j]] = self.sum_reward_components[i, j]
                    self.sum_reward_components[i, j] = 0.0
                info[i]["episode"] = epinfo
                self.rewards[i].clear()

        if self.is_unity_connected:
            self.render_id = self.render(self.render_id)
        return (
                self.getCompactImage(),
                self._reward_components[:, -1].copy(),
                self._done.copy(),
                info.copy(),
        )

    def sample_actions(self):
        actions = []
        for i in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float64)

    def reset(self, random=True):
        self._reward_components = np.zeros(
                [self.num_envs, self.rew_dim], dtype=np.float64
        )
        self.wrapper.reset(self._observation, random)
        obs = self._observation
        #
        self.obs_rms_new.update(self._observation)
        obs = self.normalize_obs(self._observation)
        if self.num_envs == 1:
            return obs[0]
        if self.is_unity_connected:
            self.render_id = self.render(self.render_id)

        return np.reshape(self.compactimage, (self.num_envs, self.rgb_channel + self.depth_channel, self.img_width, self.img_height))

    def getObs(self):
        self.wrapper.getObs(self._observation)
        return self.normalize_obs(self._observation)

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def get_obs_norm(self):
        return self.obs_rms.mean, self.obs_rms.var

    def getProgress(self):
        return self._reward_components[:, 0]

    def getImage(self, rgb=False):
        if rgb:
            self.wrapper.getImage(self._rgb_img_obs, True)
            return self._rgb_img_obs.copy()
        else:
            self.wrapper.getImage(self._gray_img_obs, False)
            return self._gray_img_obs.copy()

    def getCompactImage(self):
        self.compactimage = np.concatenate((
            np.reshape(self.getImage(True), (self.rgb_channel, self.num_envs, self.img_width, self.img_height)),
            np.reshape(self.getDepthImage(), (self.depth_channel, self.num_envs, self.img_width, self.img_height))))
        return np.reshape(self.compactimage.copy(), (self.num_envs, self.rgb_channel + self.depth_channel, self.img_width, self.img_height))

    def getDepthImage(self):
        self.wrapper.getDepthImage(self._depth_img_obs)
        return self._depth_img_obs.copy()

    def stepUnity(self, action, send_id):
        receive_id = self.wrapper.stepUnity(
                action,
                self._observation,
                self._reward,
                self._done,
                self._extraInfo,
                send_id,
        )

        return receive_id

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        # Avoid modifying by reference the original object
        # obs_ = deepcopy(obs)
        obs_ = _normalize_obs(obs, self.obs_rms).astype(np.float64)
        return obs_

    def getQuadState(self):
        self.wrapper.getQuadState(self._quadstate)
        return self._quadstate

    def getQuadAct(self):
        self.wrapper.getQuadAct(self._quadact)
        return self._quadact

    def getExtraInfo(self):
        return self._extraInfo

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            for j in range(self.rew_dim - 1):
                epinfo[self.reward_names[j]] = self.sum_reward_components[i, j]
                self.sum_reward_components[i, j] = 0.0
            info[i]["episode"] = epinfo
            self.rewards[i].clear()
        return info

    def render(self, frame_id=0):
        return self.wrapper.updateUnity(frame_id)

    def close(self):
        self.wrapper.close()

    def connectUnity(self):
        self.is_unity_connected = True
        self.wrapper.connectUnity()

    def disconnectUnity(self):
        self.is_unity_connected = False
        self.wrapper.disconnectUnity()

    def curriculumUpdate(self):
        self.wrapper.curriculumUpdate()

    def env_method(
            self,
            method_name: str,
            *method_args,
            indices: VecEnvIndices = None,
            **method_kwargs
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
                getattr(env_i, method_name)(*method_args, **method_kwargs)
                for env_i in target_envs
        ]

    def env_is_wrapped(
            self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        # the implementation in the original file gives runtime error
        # here I return true as I don't have access to the single env
        # but it should be considered when a callback using this method is used
        return [True]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    @property
    def num_envs(self):  # For PPO is the batch size
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    def start_recording_video(self, file_name):
        raise RuntimeError("This method is not implemented")

    def stop_recording_video(self):
        raise RuntimeError("This method is not implemented")

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def step_async(self, actions: np.ndarray):
        raise RuntimeError("This method is not implemented")

    def step_wait(self):
        raise RuntimeError("This method is not implemented")

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError("This method is not implemented")

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError("This method is not implemented")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):  # TODO it is really necessary?
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError("This method is not implemented")

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> "Any":
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize

    def save(self, save_path: str) -> None:
        """
        Save current VecNormalize object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)

    def save_rms(self, save_dir, n_iter) -> None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        data_path = save_dir + "/iter_{0:05d}".format(n_iter)
        np.savez(
                data_path,
                mean=np.asarray(self.obs_rms.mean),
                var=np.asarray(self.obs_rms.var),
        )

    def load_rms(self, data_dir) -> None:
        self.mean, self.var = None, None
        np_file = np.load(data_dir)
        #
        self.mean = np_file["mean"]
        self.var = np_file["var"]
        #
        self.obs_rms.mean = np.mean(self.mean, axis=0)
        self.obs_rms.var = np.mean(self.var, axis=0)
