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
