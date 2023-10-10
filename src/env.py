import gymnasium as gym
import matlab.engine
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from config import Config
from gencon import GenCon
# import markets
from markets import CarbonMarket, ElectricityMarket
from utils import calc_gen_reward

register(
    id="MMMkt-v0",
    entry_point="env:MMMktEnv",
)


# reinforcment learning env for electricity market
# use gym env as a template
class MMMktEnv(gym.Env):
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

        self.n_time_steps = config.n_timesteps

        self.elec_rewards = None

        self.carb_mkts = [CarbonMarket(config) for _ in range(self.num_mkt)]
        self.elec_mkts = [
            ElectricityMarket(config, engine) for _ in range(self.num_mkt)
        ]

        self.gencons = [
            GenCon(i, config.gen_units[i], config) for i in range(self.config.n_gens)
        ]

        action_low = np.array([1 for _ in range(config.elec_act_dim)] + [-np.inf])
        action_high = np.array([2 for _ in range(config.elec_act_dim)] + [np.inf])
        self.action_space = spaces.Box(
            low=action_low, high=action_high, shape=(config.act_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.config.obs_dim,),
            dtype=np.float32,
        )
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def reset(self, seed=None, options=None):
        self.elec_rewards = np.zeros((self.num_mkt, self.config.n_trading_days))
        obs = np.zeros((self.num_mkt, self.config.obs_dim))
        info = {}
        for i in range(self.num_mkt):
            obs[i, : self.config.elec_obs_dim] = self.elec_mkts[i].reset()
            obs[i, self.config.elec_obs_dim :] = self.carb_mkts[i].reset()
        return obs, info

    # implement step function
    def step(self, actions):
        # step one day directly
        if len(actions.shape) == 2:
            assert actions.shape == (self.num_mkt, self.config.act_dim)
        else:
            assert len(actions) == self.num_mkt * self.config.act_dim
        # get multiple run step input from each mkt
        run_step_inputs = []

        elec_actions = actions[:, : self.config.elec_act_dim]
        carb_actions = actions[:, -1].flatten()
        for i in range(self.num_mkt):
            # TODO: elect receive 24 actions and run once
            run_step_inputs += [self.elec_mkts[i].get_run_step_input(elec_actions[i])]

        winds, solars, loads, offer_qtys, offer_prcs = list(zip(*run_step_inputs))
        winds = np.array(winds).flatten().tolist()
        solars = np.array(solars).flatten().tolist()
        loads = np.array(loads).flatten().tolist()
        offer_qtys = np.array(offer_qtys).reshape((-1, self.config.n_gens, 3)).tolist()
        offer_prcs = np.array(offer_prcs).reshape((-1, self.config.n_gens, 3)).tolist()
        max_new_loads = (
            [self.config.MAX_NEW_LOAD] * self.config.n_timesteps * self.num_mkt
        )
        results = self.engine.multi_price_sim(
            matlab.double(loads),
            matlab.double(solars),
            matlab.double(winds),
            matlab.double(max_new_loads),
            matlab.double(offer_qtys),
            matlab.double(offer_prcs),
        )

        obs_list = np.zeros((self.num_mkt, self.config.obs_dim), dtype=np.float32)
        elec_rewards = np.zeros((self.num_mkt, self.n_time_steps), dtype=np.float32)
        rewards = np.zeros(self.num_mkt, dtype=np.float32)
        terminations = np.zeros(self.num_mkt, dtype=bool)
        truncateds = np.zeros(self.num_mkt, dtype=bool)

        gen_emissions = np.zeros(
            (self.num_mkt, self.n_time_steps, self.config.n_gens), dtype=np.float32
        )
        # all elec mkt obs are the same
        obs_list[:, : self.config.elec_obs_dim] = self.elec_mkts[0].get_state()

        # info for carbon and elec reward
        infos = []
        for i, result in enumerate(results):
            mkt_idx = i // self.n_time_steps
            time_idx = i % self.n_time_steps
            res = np.array(result["clear"])
            elec_rewards[mkt_idx, time_idx] = self.elec_mkts[0].calc_gen_reward(res)

            for j in range(self.config.n_gens):
                gen_emissions[mkt_idx, time_idx, j] = self.gencons[
                    j
                ].calc_carbon_emission(res[j, self.config.QTY_COL])
            # set emissions
        for i, carb_mkt in enumerate(self.carb_mkts):
            elec_r_day = elec_rewards[i].sum()
            self.elec_rewards[i, carb_mkt.day_t] = elec_r_day
            carb_mkt.set_gen_emission(gen_emissions[i].sum(axis=0)[i])
            (
                carb_obs,
                carb_r,
                carb_terminated,
                _,
                carb_info,
            ) = carb_mkt.run_step(carb_actions[i])

            terminations[i] = carb_terminated
            rewards[i] = carb_r + elec_r_day
            obs_list[i, self.config.elec_obs_dim :] = carb_obs

            if "final_info" in carb_info:
                elec_r_mkt = self.elec_rewards[i].sum()
                carb_info["final_info"]["episode"]["carb_r"] = carb_info["final_info"][
                    "episode"
                ]["r"]
                carb_info["final_info"]["episode"]["elec_r"] = elec_r_mkt
                carb_info["final_info"]["episode"]["r"] = (
                    carb_info["final_info"]["episode"]["r"] + elec_r_mkt
                )
            carb_info["carb_r"] = carb_r
            carb_info["elec_r"] = elec_r_day

            infos.append(carb_info)

        infos = {k: [d[k] for d in infos] for k in infos[0]}

        return obs_list, rewards, terminations, truncateds, infos


# reinforcement learning env for carbon market
class CarbMktEnv:
    def __init__(self, config) -> None:
        self.market = CarbonMarket(config)
        self.gen_id = config.agent_gen_id

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


if __name__ == "__main__":
    config = Config()
    config.num_mkt = 2
    config.n_trading_days = 2
    env = MMMktEnv(config=config)
    env.reset()
    for i in range(4):
        obs, r, terminations, _, infos = env.step(actions=np.ones((2, 25)))
        print(obs)
        print(r)
        print(terminations)
        print(infos)

