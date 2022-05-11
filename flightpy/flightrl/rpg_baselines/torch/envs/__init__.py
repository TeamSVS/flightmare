from gym.envs.registration import register
import logging
 
import os
from ruamel.yaml import YAML

cfg = YAML().load(
    open(
        os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
    )
)


try:
  register(
    id='compass_env-v0',
    entry_point='flightmare.flightpy.flightrl.rpg_baselines.torch.envs.vec_env_wrapper:FlightEnvVec',
    max_episode_steps=300,
    kwargs=dict(env_cfg=cfg,name="train",mode="depth"),
  )
except Exception as e:

  logging.error(traceback.format_exc())
