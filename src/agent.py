import numpy as np


class RuleAgent:
    def __init__(self, gen_id):
        # Initialize any necessary variables or data structures
        pass

    def act(self, prev_emission, allowance, remaining_time):
        # Implement the agent's decision-making logic based on the observation

        prev_emission_total = np.sum(prev_emission)
        prev_emission_mean = np.mean(prev_emission)
        remaining_allowance = allowance - prev_emission_total
        emission_expected = prev_emission_mean * remaining_time

        # if action > 0, buy extra allowance, else sell allowance
        action = emission_expected - remaining_allowance

        # return (buy, sell) pair
        return (max(action, 0), max(-action, 0))
