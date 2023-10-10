import os

import matlab.engine
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from agent import RuleAgent
from config import Config
from env import CarbMktEnv, ElecMktEnv
from gencon import GenCon
from markets import CarbonMarket, ElectricityMarket

import gymnasium as gym

# test carb env loop
env = gym.make("CarbMkt-v0")

env.reset()
terminated = False
while not terminated:
    env.set_emissions(np.ones(Config.n_gens, dtype=np.float32))
    obs, reward, terminated, truncated, info = env.step(None)
    print(obs, reward, terminated, info)

# test elec
config_eval = Config()
config_eval.num_mkt = 1
env = gym.make("ElecMkt-v0", config=config_eval)

env.reset()
terminated = False
i = 0
while not terminated:
    # env.set_emissions(np.ones(Config.n_gens, dtype=np.float32))
    print(i)
    obs, reward, terminated, truncated, info = env.step(np.array([1], dtype=np.float32))
    print(obs, reward, terminated, info)
    i += 1
