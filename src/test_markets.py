import numpy as np
from markets import CarbonMarket, Config


def test_get_agent_obs():
    # Create a CarbonMarket object with a Config object
    config = Config()
    carbon_market = CarbonMarket(config)

    # Set some initial values for the CarbonMarket object
    carbon_market.n_trading_days = 10
    carbon_market.day_t = 2
    carbon_market.carbon_prices = [10, 11, 12, 13, 14]
    carbon_market.gen_emissions = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    carbon_market.config.carbon_allowance_initial = 100
    carbon_market.config.n_gens = 5
    carbon_market.buying_volumes = np.array([[1, 2, 3, 4, 5], [6, 6, 6, 6, 6]])
    carbon_market.selling_volumes = np.array([[2, 2, 2, 2, 2], [10, 8, 8, 8, 8]])

    # Call the get_agent_obs method for gen_id=0
    obs = carbon_market.get_agent_obs(0)

    # Check that the observation is a numpy array
    assert isinstance(obs, np.ndarray)

    # Check that the observation has 5 elements
    assert len(obs) == 5

    # Check that the first element of the observation is the total gen emission now
    assert obs[0] == np.sum(carbon_market.gen_emissions[: carbon_market.day_t, 0])

    # Check that the second element of the observation is the total system emission now
    assert obs[1] == np.sum(carbon_market.gen_emissions)

    # Check that the third element of the observation is the carbon price
    assert obs[2] == carbon_market.carbon_prices[-1]

    # Check that the fourth element of the observation is the carbon allowance for gen_id=0
    assert obs[3] == carbon_market.carbon_allowance[-1][0]

    # Check that the fifth element of the observation is the time remaining
    assert obs[4] == carbon_market.n_trading_days - carbon_market.day_t - 1


test_get_agent_obs()
