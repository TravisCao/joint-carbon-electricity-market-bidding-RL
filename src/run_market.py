import matlab.engine
import numpy as np
import pandas as pd
import os
from tqdm import trange, tqdm

from config import Config
from gencon import GenCon
from agent import RuleAgent
from markets import ElectricityMarket, CarbonMarket
from env import ElecMktEnv, CarbMktEnv


# EM competitively, CM rule-based

# agent_gen_action = 1.0
# engine = matlab.engine.start_matlab()
# non_renewable_gens = [
#     GenCon(i, Config.gen_units[i], Config) for i in range(Config.n_gens)
# ]
# elec_market = ElectricityMarket(
#     Config,
#     engine,
# )

# days = 5

# agent = RuleAgent()

# carb_market = CarbonMarket(Config)
# for day in range(days):
#     emission = np.zeros((48, 6))
#     # for t in trange(Config.n_timesteps):
#     for t in range(Config.n_timesteps):
#         res = elec_market.run_step(agent_gen_action)
#         qtys = res[:, elec_market.QTY_COL]
# for i, gen in tqdm(enumerate((non_renewable_gens))):
#     e = gen.calc_carbon_emission(qtys[i])
#     emission[t, i] = e
# print(
# f"day:{day} timestep: {elec_market.timestep} gen: {gen.gen_id}, emission: {e}"
# )

# carb_market.set_gen_emission(np.sum(emission, axis=0))
# agent_action = agent.act(*carb_market.get_rule_obs())
# print(agent_gen_action)
# info = carb_market.trade(agent_action[0], agent_action[1])
# print(info)


# np.savetxt(
#     f"{os.path.dirname(os.path.abspath(__file__))}/../data/emission.csv",
#     emission_overall,
#     delimiter=",",
# )


# if __name__ == "__main__":
#     for i in trange(100):
#         emission_overall[i : i + 48, :] = run_electricity_market_simple(1.0)
#     pd.DataFrame(emission_overall).to_csv("emission.csv")

# engine = matlab.engine.start_matlab()
# elec_mkt_env = ElecMktEnv(Config, engine)
# carb_mkt_env = CarbMktEnv(Config)

# elec_obs = elec_mkt_env.reset()
# carb_obs = carb_mkt_env.reset_rule()

# agent = RuleAgent()

# while 1:
#     emission = np.random.rand(48, 6)
#     carb_mkt_env.market.set_gen_emission(np.sum(emission, axis=0))
#     action = agent.act(*carb_obs)
#     r, carb_obs, terminated, info = carb_mkt_env.step_rule(action)
#     print("day: ", carb_mkt_env.market.day_t)
#     print(r, carb_obs, terminated, info)
#     if terminated:
#         break

from config import Config
import matlab.engine
from markets import ElectricityMarket, CarbonMarket
from env import ElecMktEnv, CarbMktEnv
import numpy as np

engine = matlab.engine.start_matlab()
elec_market = ElectricityMarket(
    Config,
    engine,
)

config = Config
config.num_mkt = 21
elec_mkt_env = ElecMktEnv(config, engine)

elec_obs = elec_mkt_env.reset()
actions = np.arange(100, 201, 5) / 100.0
for i in range(48):
    print(i)
    obs_list, rewards, terminateds, infos = elec_mkt_env.step(actions)
    print(rewards)
    if "final_info" in infos[0]:
        for info in infos:
            print(info["final_info"]["r"])

# EM competitively, CM rule-based

agent_gen_action = 1.0
engine = matlab.engine.start_matlab()
non_renewable_gens = [
    GenCon(i, Config.gen_units[i], Config) for i in range(Config.n_gens)
]
elec_market = ElectricityMarket(
    Config,
    engine,
)

days = 5

agent = RuleAgent()

carb_market = CarbonMarket(Config)
for day in range(days):
    emission = np.zeros((24, 6))
    print("day", day)
    # for t in trange(Config.n_timesteps):
    for t in range(Config.n_timesteps):
        print("t", t)
        res = elec_market.run_step(agent_gen_action)
        qtys = res[:, elec_market.QTY_COL]
for i, gen in tqdm(enumerate((non_renewable_gens))):
    e = gen.calc_carbon_emission(qtys[i])
    emission[t, i] = e
print(f"day:{day} timestep: {elec_market.timestep} gen: {gen.gen_id}, emission: {e}")

# carb_market.set_gen_emission(np.sum(emission, axis=0))
# agent_action = agent.act(*carb_market.get_rule_obs())
# print(agent_gen_action)
# info = carb_market.trade(agent_action[0], agent_action[1])
# print(info)


# np.savetxt(
#     f"{os.path.dirname(os.path.abspath(__file__))}/../data/emission.csv",
#     emission_overall,
#     delimiter=",",
# )


# if __name__ == "__main__":
#     for i in trange(100):
#         emission_overall[i : i + 48, :] = run_electricity_market_simple(1.0)
#     pd.DataFrame(emission_overall).to_csv("emission.csv")

# engine = matlab.engine.start_matlab()
# elec_mkt_env = ElecMktEnv(Config, engine)
# carb_mkt_env = CarbMktEnv(Config)

# elec_obs = elec_mkt_env.reset()
# carb_obs = carb_mkt_env.reset_rule()

# agent = RuleAgent()

# while 1:
#     emission = np.random.rand(48, 6)
#     carb_mkt_env.market.set_gen_emission(np.sum(emission, axis=0))
#     action = agent.act(*carb_obs)
#     r, carb_obs, terminated, info = carb_mkt_env.step_rule(action)
#     print("day: ", carb_mkt_env.market.day_t)
#     print(r, carb_obs, terminated, info)
#     if terminated:
#         break

# from config import Config
# import matlab.engine
# from markets import ElectricityMarket, CarbonMarket
# import numpy as np

# engine = matlab.engine.start_matlab()
# elec_market = ElectricityMarket(
#     Config,
#     engine,
# )

# # mkts = [ElectricityMarket(Config, engine) for _ in range(10)]
# # tmp = []
# # actions = np.ones((10, 1))
# # for i, mkt in enumerate(mkts):
# #     tmp += [mkt.get_run_step_input(actions[i])]
# # len(tmp)


# mkts = [ElectricityMarket(Config, engine) for _ in range(10)]
# run_step_inputs = []
# actions = np.ones(10)
# for i, mkt in enumerate(mkts):
#     run_step_inputs += [mkt.get_run_step_input(actions[i])]

# winds, solars, loads, offer_qtys, offer_prcs = list(zip(*run_step_inputs))
# max_new_loads = [Config.MAX_NEW_LOAD for i in range(len(actions))]

# # save to mat

# # matlab.double(loads), matlab.double(solars), matlab.double(winds), matlab.double(max_new_loads), matlab.double(offers_qtys), matlab.double(offer_prcs)
# save to mat file
# from scipy.io import loadmat, savemat

# mdict = {
#     "loads": matlab.double(loads),
#     "solars": matlab.double(solars),
#     "winds": matlab.double(winds),
#     "max_new_loads": matlab.double(max_new_loads),
#     "offer_qtys": matlab.double(offer_qtys),
#     "offer_prcs": matlab.double(offer_prcs),
# }
# savemat("value.mat", mdict)


# results = engine.multi_price_sim(
#     matlab.double(loads),
#     matlab.double(solars),
#     matlab.double(winds),
#     matlab.double(max_new_loads),
#     matlab.double(offer_qtys),
#     matlab.double(offer_prcs),
# )


# mkts = [ElectricityMarket(Config, engine) for _ in range(10)]
# run_step_inputs = []
# actions = np.ones(10)
# for i, mkt in enumerate(mkts):
#     run_step_inputs += [mkt.get_run_step_input(actions[i])]

# winds, solars, loads, offer_qtys, offer_prcs = list(zip(*run_step_inputs))
# max_new_loads = [Config.MAX_NEW_LOAD for i in range(len(actions))]

# save to mat

# matlab.double(loads), matlab.double(solars), matlab.double(winds), matlab.double(max_new_loads), matlab.double(offers_qtys), matlab.double(offer_prcs)
# save to mat file
# from scipy.io import loadmat, savemat

# mdict = {
#     "loads": matlab.double(loads),
#     "solars": matlab.double(solars),
#     "winds": matlab.double(winds),
#     "max_new_loads": matlab.double(max_new_loads),
#     "offer_qtys": matlab.double(offer_qtys),
#     "offer_prcs": matlab.double(offer_prcs),
# }
# savemat("value.mat", mdict)


# results = engine.multi_price_sim(
#     matlab.double(loads),
#     matlab.double(solars),
#     matlab.double(winds),
#     matlab.double(max_new_loads),
#     matlab.double(offer_qtys),
#     matlab.double(offer_prcs),
# )
