from gencon import GenCon
from utils import calc_gen_reward
import numpy as np
import matlab.engine
from config import Config

# import markets
from markets import ElectricityMarket, CarbonMarket

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

register(
    id="ElecMkt-v0",
    entry_point="env:ElecMktEnv",
)


# reinforcment learning env for electricity market
# use gym env as a template
class ElecMktEnv(gym.Env):
    def __init__(self, config=None, engine=None) -> None:
        if config is None:
            config = Config
        if engine is None:
            engine = matlab.engine.start_matlab()

        self.engine = engine
        self.config = config
        self.num_mkt = config.num_mkt
        self.num_envs = config.num_mkt
        self.is_vector_env = True

        self.mkts = [ElectricityMarket(config, engine) for _ in range(self.num_mkt)]
        self.action_space = spaces.Box(low=1.0, high=2.0, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.config.elec_obs_dim,),
            dtype=np.float32,
        )
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def reset(self, seed=None, options=None):
        obs = np.zeros((self.num_mkt, self.config.elec_obs_dim))
        info = {}
        for i, mkt in enumerate(self.mkts):
            obs[i] = mkt.reset()
        return obs, info

    # implement step function
    def step(self, actions):
        assert len(actions) == len(self.mkts)
        # get multiple run step input from each mkt
        run_step_inputs = []
        for i, mkt in enumerate(self.mkts):
            run_step_inputs += [mkt.get_run_step_input(actions[i])]

        winds, solars, loads, offer_qtys, offer_prcs = list(zip(*run_step_inputs))
        max_new_loads = [self.config.MAX_NEW_LOAD for i in range(len(actions))]
        results = self.engine.multi_price_sim(
            matlab.double(loads),
            matlab.double(solars),
            matlab.double(winds),
            matlab.double(max_new_loads),
            matlab.double(offer_qtys),
            matlab.double(offer_prcs),
        )
        obs_list = np.zeros((self.num_mkt, self.config.elec_obs_dim), dtype=np.float32)
        rewards = np.zeros(self.num_mkt, dtype=np.float32)
        terminations = np.zeros(self.num_mkt, dtype=np.bool)
        infos = []
        for i, mkt in enumerate(self.mkts):
            obs, r, terminated, info = mkt.step_no_run(np.array(results[i]["clear"]))
            obs_list[i] = obs
            rewards[i] = r
            terminations[i] = terminated
            infos.append(info)

        infos = {k: [d[k] for d in infos] for k in infos[0]}

        truncated = np.zeros(self.num_mkt, dtype=np.bool)
        return obs_list, rewards, terminations, truncated, infos

    def get_state(self):
        obs_list = np.zeros((self.num_mkt, self.config.elec_obs_dim))
        for i, mkt in enumerate(self.mkts):
            obs_list[i] = mkt.get_state()
        return obs_list

    def close(self):
        self.engine.quit()


# reinforcement learning env for carbon market
class CarbMktEnv(gym.Env):
    def __init__(self, config=None) -> None:
        self.market = CarbonMarket(config)
        self.gen_id = config.agent_gen_id

        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(5,), dtype=np.float32
        )

    def reset(self):
        self.market.reset_system()
        return self.get_agent_state()

    def reset_rule(self):
        self.market.reset_system()
        return self.market.get_rule_obs()

    def step_rule(self, action):
        r, obs, terminated, info = self.market.run_step_rule(action)
        return r, obs, terminated, info

    # implement step function
    def step(
        self,
        action,
    ):
        r, obs, terminated, info = self.market.run_step(action, self.gen_id)
        return (r, obs, terminated, info)

    def get_agent_state(self):
        return self.market.get_agent_obs(self.gen_id)
