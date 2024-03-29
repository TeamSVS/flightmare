import os
import pickle
import signal
import subprocess
import sys
import threading
import time
from abc import ABC
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import json
import psutil
import torch
import torchvision.transforms as transforms
import random
from threading import Timer, Thread, Event
import logging
import gym
import numpy as np
from flightgym import VisionEnv_v1
from gym import spaces
from ruamel.yaml import RoundTripDumper, YAML, dump
from numpy.core.fromnumeric import shape
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv,
                                                           VecEnvIndices,
                                                           VecEnvObs,
                                                           VecEnvStepReturn)
from stable_baselines3.common.vec_env.util import (copy_obs_dict, dict_to_obs,
                                                   obs_space_info)

######################################
##########--COSTANT VALUES--##########
######################################

FLIGHTMAER_EXE = "RPG_Flightmare.x86_64"
DEPTH_CHANNELS = 1
RGB_CHANNELS = 3
HEARTBEAT_INTERVAL = 4
FLIGHTMAER_NEXT_FOLDER = "/flightrender/"


######################################
############--FUNCTIONS--#############
######################################


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


def _normalize_img(obs: np.ndarray) -> np.ndarray:
    return obs / 255

    ############################################
    ############--HB-DEAMON-CLASS--#############
    ############################################


class PingThread(Thread):
    def __init__(self, event, vecEnv):
        Thread.__init__(self)
        self.stopped = event
        self.env = vecEnv

    def run(self):
        while True:
            time.sleep(2)
            while not self.stopped.wait(HEARTBEAT_INTERVAL):
                self.env.wrapper.sendUnityPing()

    #######################################
    ############--MAIN-CLASS--#############
    #######################################


class FlightEnvVec(VecEnv, ABC):
    def __init__(self, env_cfg, name, mode, n_frames=3, in_port=0, out_port=0):
        self._flightmare_process = None
        self.render_id = 0
        self.stacked_drone_state = []
        self.stacked_depth_imgs = []
        self.stacked_rgb_imgs = []
        self.name = name
        self.n_frames = n_frames
        self.env_cfg = env_cfg
        self.stopFlag = Event()
        self.thread = PingThread(self.stopFlag, self)
        if in_port == 0 or out_port == 0:
            self.in_port = env_cfg["unity"]["input_port"]
            self.out_port = env_cfg["unity"]["output_port"]
        else:
            self.in_port = in_port
            self.out_port = out_port
            env_cfg["unity"]["input_port"] = in_port
            env_cfg["unity"]["output_port"] = out_port

        self.wrapper = VisionEnv_v1(dump(self.env_cfg, Dumper=RoundTripDumper), False)
        self.is_unity_connected = False
        self.var = None
        self.mean = None
        self.envs = None
        self._reward = None
        self.mode = mode  # rgb, depth, both
        self.seed_val = 0
        self._heartbeat = True if env_cfg["simulation"]["heartbeat"] == "yes" else False

        self.obs_ranges_dic = {0: [-10, 65],
                               1: [-10, 10],
                               2: [0, 10],
                               7: [-3, 65],
                               8: [-25, 30],
                               9: [-20, 20],
                               10: [-9, 9],
                               11: [-9, 9],
                               12: [-9, 9]}

        # if os.path.exists("NEW_VAL_NORMALIZATION.txt"):
        #    with open('NEW_VAL_NORMALIZATION.txt') as json_file:
        #        self.obs_ranges_dic = json.load(json_file)

        self.act_dim = self.wrapper.getActDim()
        self.obs_dim = self.wrapper.getObsDim()  # C++ obs shape
        self.rew_dim = self.wrapper.getRewDim()
        self.img_width = self.wrapper.getImgWidth()
        self.img_height = self.wrapper.getImgHeight()

        ###########################################
        ##############--HB-DEAMON---###############
        ###########################################
        if self._heartbeat:
            self.thread.daemon = True
            self.thread.start()

        ###########################################
        ###############--OBS-SPACE--###############
        ###########################################

        drone_spaces = {'state': spaces.Box(
            low=-1., high=1.,
            shape=(13,), dtype=np.float32
        )}

        if 'depth' == self.mode or 'both' == self.mode:
            drone_spaces['depth'] = spaces.Box(
                low=0., high=1.,
                shape=(DEPTH_CHANNELS, self.n_frames, self.img_height, self.img_width), dtype=np.float32
            )
        if 'rgb' == self.mode or 'both' == self.mode:
            drone_spaces['rgb'] = spaces.Box(
                low=0, high=255,
                shape=(RGB_CHANNELS, self.n_frames, self.img_height, self.img_width), dtype=np.uint8
            )

        self._observation_space = spaces.Dict(spaces=drone_spaces)

        ###########################################
        ###############--ACT-SPACE--###############
        ###########################################
        self._action_space = spaces.Box(
            low=np.ones(self.act_dim) * -1.0,
            high=np.ones(self.act_dim) * 1.0,
            dtype=np.float64,
        )

        self._observation = np.zeros([self.num_envs, self.obs_dim], dtype=np.float64)

        self._rgb_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height * RGB_CHANNELS], dtype=np.uint8
        )
        self._gray_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height], dtype=np.uint8
        )
        self._depth_img_obs = np.zeros(
            [self.num_envs, self.img_width * self.img_height], dtype=np.float32
        )
        #
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
        self.spawn_flightmare(self.in_port, self.out_port)

    def seed(self, seed=0):
        if seed != 0:
            self.seed_val = seed

        self.wrapper.setSeed(self.seed_val)

    def spawn_flightmare(self, input_port=10277, output_port=10278):
        if input_port > 0 and output_port > 0:
            ports = " -input-port {0} -output-port {1}".format(input_port, output_port)
            self.in_port = input_port
            self.out_port = output_port
        else:
            ports = " -input-port {0} -output-port {1}".format(self.in_port, self.out_port)
        try:
            self._flightmare_process = subprocess.Popen(
                [os.environ["FLIGHTMARE_PATH"] + FLIGHTMAER_NEXT_FOLDER + FLIGHTMAER_EXE + ports],
                stdout=subprocess.PIPE,
                shell=True, preexec_fn=os.setsid)
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    def kill_flightmare(self):
        os.killpg(os.getpgid(self._flightmare_process.pid), signal.SIGTERM)
        self._flightmare_process = None

    def change_obstacles(self, seed=0, difficult="medium", level=0, random=False):
        # TODO Random not yet implemented

        self.close()
        self.kill_flightmare()
        self.spawn_flightmare()

        self.env_cfg["environment"]["level"] = difficult
        self.env_cfg["environment"]["env_folder"] = "environment_" + str(level)

        self.wrapper = VisionEnv_v1(dump(self.env_cfg, Dumper=RoundTripDumper), False)
        if seed != 0:
            self.seed_val = seed

        self.stopFlag.clear()
        self.seed(self.seed_val)
        # Require render cfg to be True
        self.connectUnity()
        return self.reset(True)

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

    def _stack_frames(self, frame_list, new_frame):
        if len(frame_list) == 0:
            frame_list = [new_frame for _ in range(self.n_frames)]
        else:
            frame_list = frame_list[:self.n_frames - 1]
            frame_list.insert(0, new_frame)
        return frame_list

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

        logging.info("." + self.name)
        if self.is_unity_connected:
            self.render_id = self.render(
                self.render_id)  # TODO INCREASE RENDER ID IT IS REALLY NECESSARY TO DO RENDER ID +1
            logging.info(self.getImage(True))
        new_obs = self.getObs()

        return (
            new_obs,
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
        logging.info("Reset")
        self.stacked_drone_state = []
        self.stacked_depth_imgs = []
        self.stacked_rgb_imgs = []
        self._reward_components = np.zeros(
            [self.num_envs, self.rew_dim], dtype=np.float64
        )
        self.wrapper.reset(self._observation, random)
        obs = self._observation
        #
        self.obs_rms_new.update(self._observation)
        if self.is_unity_connected:
            self.render_id = self.render(self.render_id)
        new_obs = self.getObs()
        return new_obs

    def getObs(self):
        ## Old Obs ##
        # self.wrapper.getObs(self._observation)
        # self.normalize_obs(self._observation)

        ## New Obs ##
        new_obs = {}
        # position (z, x, y) = [0:3], attitude=[3:7], linear_velocity=[7:10], angular_velocity=[10:13]
        drone_state = self.getQuadState()[:, 1:14].copy()
        # normalize between -1 and 1
        for key in self.obs_ranges_dic:
            # extract values
            value = drone_state[:, int(key)]
            lower_bound = self.obs_ranges_dic[key][0]
            upper_bound = self.obs_ranges_dic[key][1]
            # compute normalization
            new_val = 2 * (value - lower_bound) / (upper_bound - lower_bound) - 1
            old_range = self.obs_ranges_dic[key]
            changed_range = False
            if new_val.max() > 1:
                # update upper bound
                self.obs_ranges_dic[key][1] = value.max()
                changed_range = True
                # update normalization based on new range
                lower_bound = self.obs_ranges_dic[key][0]
                upper_bound = self.obs_ranges_dic[key][1]
                new_val = 2 * (value - lower_bound) / (upper_bound - lower_bound) - 1

            if new_val.min() < -1:
                # update lower bound
                self.obs_ranges_dic[key][0] = value.min()
                changed_range = True
                # update normalization based on new range
                lower_bound = self.obs_ranges_dic[key][0]
                upper_bound = self.obs_ranges_dic[key][1]
                new_val = 2 * (value - lower_bound) / (upper_bound - lower_bound) - 1
            drone_state[:, int(key)] = new_val

            if changed_range:
                logging.info("state out of normalization range. Range updated from {0} to {1}".format(
                    old_range, self.obs_ranges_dic[key]))
                with open("NEW_VAL_NORMALIZATION.txt", "w") as myfile:
                    myfile.write(json.dumps(self.obs_ranges_dic))

            new_obs['state'] = np.array(drone_state)
        if 'depth' == self.mode or 'both' == self.mode:
            depth_imgs = self.getDepthImage().reshape((self.num_envs, 1, self.img_height, self.img_width))
            self.stacked_depth_imgs = self._stack_frames(self.stacked_depth_imgs, depth_imgs)
            new_obs['depth'] = np.array(self.stacked_depth_imgs).swapaxes(0, 1).swapaxes(1, 2)
        if 'rgb' == self.mode or 'both' == self.mode:
            rgb_imgs = _normalize_img(
                np.reshape(self.getImage(True), (self.num_envs, RGB_CHANNELS, self.img_width, self.img_height)))
            self.stacked_rgb_imgs = self._stack_frames(self.stacked_rgb_imgs, rgb_imgs)
            new_obs['rgb'] = np.array(self.stacked_rgb_imgs).swapaxes(0, 1).swapaxes(1, 2)

        return new_obs.copy()

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
        ret = self.wrapper.updateUnity(frame_id)
        return ret

    def close(self):
        self.stopFlag.set()
        self.reset()
        self.disconnectUnity()
        self.wrapper.close()

    def connectUnity(self):
        self.is_unity_connected = True
        self.wrapper.connectUnity()

    def sendUnityPing(self):
        self.wrapper.sendUnityPing()

    def setFakeQuadrotorScale(self, scale=1.0):
        self.wrapper.setFakeQuadrotorScale(scale)

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

    def curriculum_callback(self):

        self.wrapper.curriculumUpdate()

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

    #########################################################################################################
    #######################################--NOT IMPLEMENTED METHODS--#######################################
    #########################################################################################################
    def env_is_wrapped(
            self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        # the implementation in the original file gives runtime error
        # here I return true as I don't have access to the single env
        # but it should be considered when a callback using this method is used
        return [True]

    def step_async(self, actions: np.ndarray):
        raise RuntimeError("This method is not implemented")

    def step_wait(self):
        raise RuntimeError("This method is not implemented")

    def start_recording_video(self, file_name):
        raise RuntimeError("This method is not implemented")

    def stop_recording_video(self):
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
