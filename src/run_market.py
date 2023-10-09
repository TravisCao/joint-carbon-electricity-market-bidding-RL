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

# test env loop
env = gym.make("CarbMkt-v0")

env.reset()
terminated = False
while not terminated:
    env.set_emissions(np.ones(Config.n_gens, dtype=np.float32))
    obs, reward, terminated, truncated, info = env.step(None)
    print(obs, reward, terminated, info)
