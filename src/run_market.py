import os

import gymnasium as gym
import matlab.engine
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from config import Config
from env import MMMktEnv
from gencon import GenCon
from markets import CarbonMarket, ElectricityMarket


def evaluate_rule(
    env,
):
    assert len(env.carb_mkts) == len(env.elec_mkts) == 1

    obs_eval, _ = env.reset()
    cnt = 0
    while 1:
        # if len(episodic_returns) < Config.n_trading_days:
        # action_eval = agent.get_action_mean(torch.Tensor(obs_eval).to(device))

        next_obs_eval, _, _, _, info_eval = env.step(np.ones(25).reshape(1, -1))

        print(
            f"eval_day={cnt}, elec_r={info_eval['elec_r'][0]}, carb_r={info_eval['carb_r'][0]}"
        )

        # final_info means 1 day is finished
        if "final_info" in info_eval:
            epi_r_total = info_eval["final_info"][0]["episode"]["r"]
            epi_r_carb = info_eval["final_info"][0]["episode"]["carb_r"]
            epi_r_elec = info_eval["final_info"][0]["episode"]["elec_r"]
            print(
                f"LAST DAY, elec rewards={epi_r_elec}, carb_r={epi_r_carb}, total_r={epi_r_total}"
            )
            break
        obs_eval = next_obs_eval
        cnt += 1
    return epi_r_total, epi_r_elec, epi_r_carb


def make_env(env_id, gamma, config=Config, engine=None):
    env = gym.make(env_id, config=config, engine=engine)
    # env = gym.wrappers.FlattenObservation(
    # env
    # )  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


config = Config()
config.num_mkt = 1
envs = make_env("MMMkt-v0", 0.99, config=config, engine=None)
evaluate_rule(envs)
