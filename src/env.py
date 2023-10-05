from gencon import GenCon

# import markets
from markets import ElectricityMarket, CarbonMarket


# reinforcment learning env for electricity market
# use gym env as a template
class ElecMktEnv:
    def __init__(self, config, engine) -> None:
        self.market = ElectricityMarket(config, engine)

    def reset(self):
        return self.market.reset()

    # implement step function
    def step(self, action):
        res = self.market.run_step(action)
        r = self.market.calc_gen_reward(res)
        obs = self.market.get_state()
        terminated = self.market.terminated
        info = None
        return (obs, r, terminated, info)

    def get_state(self):
        return self.market.get_state()


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
