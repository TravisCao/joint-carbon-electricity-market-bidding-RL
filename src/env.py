import gymnasium as gym
import matlab.engine
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from config import Config
from gencon import GenCon

# import markets
from markets import CarbonMarket, ElectricityMarket

register(
    id="ElecMkt-v0",
    entry_point="env:ElecMktEnv",
)

register(
    id="CarbMkt-v0",
    entry_point="env:CarbMktEnv",
)


# reinforcment learning env for electricity market
# use gym env as a template
class ElecMktEnv(gym.Env):
    def __init__(self, config=Config, engine=None) -> None:
        if engine is None:
            engine = matlab.engine.start_matlab()

        self.engine = engine
        self.config = config
        self.num_mkt = config.num_mkt
        self.num_envs = config.num_mkt
        self.is_vector_env = True

        self.gencons = [
            GenCon(i, config.gen_units[i], config) for i in range(config.n_gens)
        ]
        self.gen_emissions = np.zeros(
            (config.num_mkt, config.n_timesteps, config.n_gens), dtype=np.float32
        )

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
        self.gen_emissions = np.zeros(
            (self.config.num_mkt, self.config.n_timesteps, self.config.n_gens)
        )
        obs = np.zeros((self.num_mkt, self.config.elec_obs_dim), dtype=np.float32)
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

            qtys = np.array(results[i]["clear"])[:, self.config.QTY_COL]
            # TODO: avoid for
            for j in range(self.config.n_gens):
                self.gen_emissions[i, mkt.timestep, j] = np.array(
                    self.gencons[j].calc_carbon_emission(qtys[j])
                )

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

    def get_gen_emissions_day(self):
        return np.sum(self.gen_emissions, axis=1, dtype=np.float32)


# reinforcement learning env for carbon market
class CarbMktEnv(gym.Env):
    def __init__(self, config=Config) -> None:
        self.gen_id = config.agent_gen_id
        self.num_envs = config.num_mkt
        self.num_mkt = config.num_mkt
        self.config = config
        self.is_vector_env = True

        self.mkts = [CarbonMarket(config) for _ in range(self.num_mkt)]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(config.carb_obs_dim,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def reset(self, seed=None, options=None):
        obs = np.zeros((self.num_mkt, self.config.carb_obs_dim), dtype=np.float32)
        for i, mkt in enumerate(self.mkts):
            obs[i] = mkt.reset()
        info = {}
        return self.get_agent_state(), info

    # implement step function
    def step(
        self,
        action=None,
    ):
        assert len(action) == len(self.mkts)
        obs_list = np.zeros((self.num_mkt, self.config.carb_obs_dim), dtype=np.float32)
        rewards = np.zeros(self.num_mkt, dtype=np.float32)
        terminations = np.zeros(self.num_mkt, dtype=np.bool)
        truncated = np.zeros(self.num_mkt, dtype=np.bool)
        infos = []
        for i, mkt in enumerate(self.mkts):
            obs, r, terminated, truncated, info = mkt.run_step(action[i])
            obs_list[i] = obs
            rewards[i] = r
            terminations[i] = terminated
            infos.append(info)

        infos = {k: [d[k] for d in infos] for k in infos[0]}
        return obs_list, rewards, terminations, truncated, infos

    def get_agent_state(self):
        obs_list = np.zeros((self.num_mkt, self.config.carb_obs_dim))
        for i, mkt in enumerate(self.mkts):
            obs_list[i] = mkt.get_agent_obs()
        return obs_list

    def set_gen_emissions(self, gen_emissions):
        for i, mkt in enumerate(self.mkts):
            mkt.set_gen_emission(gen_emissions[i])


if __name__ == "__main__":
    from ppo import make_env
    import numpy as np
    from config import Config

    carb_envs = make_env("CarbMkt-v0", 0.99, Config)
    carb_envs.set_emissions(np.ones((Config.num_mkt, Config.n_gens)))
