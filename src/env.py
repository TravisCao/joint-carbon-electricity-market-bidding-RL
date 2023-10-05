from gencon import GenCon
from utils import calc_gen_reward
import numpy as np
import matlab.engine

# import markets
from markets import ElectricityMarket, CarbonMarket


# reinforcment learning env for electricity market
# use gym env as a template
class ElecMktEnv:
    def __init__(self, config, engine) -> None:
        self.engine = engine

        self.num_mkt = config.num_mkt
        self.config = config

        self.mkts = [ElectricityMarket(config, engine) for _ in range(self.num_mkt)]

    def reset(self):
        obs = np.zeros((self.num_mkt, self.config.elec_obs_dim))
        for i, mkt in enumerate(self.mkts):
            obs[i] = mkt.reset()
        return obs

    # implement step function
    def step(self, actions):
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
        obs_list = np.zeros((self.num_mkt, self.config.elec_obs_dim))
        rewards = np.zeros(self.num_mkt)
        terminateds = np.zeros(self.num_mkt)
        infos = []
        for i, mkt in enumerate(self.mkts):
            obs, r, terminated, info = mkt.step_no_run(np.array(results[i]['clear']))
            obs_list[i] = obs
            rewards[i] = r
            terminateds[i] = terminated
            infos.append(info)
        return obs_list, rewards, terminateds, infos

    def get_state(self):
        obs_list = np.zeros((self.num_mkt, self.config.elec_obs_dim))
        for i, mkt in enumerate(self.mkts):
            obs_list[i] = mkt.get_state()
        return obs_list

    def close(self):
        self.engine.quit()


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
