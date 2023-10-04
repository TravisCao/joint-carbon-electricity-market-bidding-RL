import numpy as np
from markets import CarbonMarket, Config


def test_price_clearing():
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

    print(carbon_market.carbon_prices[-1])

    # Call the price_clearing method
    p_e_now = carbon_market.price_clearing()

    print(p_e_now)

    # Check that the carbon price is a float
    assert isinstance(p_e_now, float)

    # Check that the carbon price is greater than or equal to 0
    assert p_e_now >= 0


test_price_clearing()
