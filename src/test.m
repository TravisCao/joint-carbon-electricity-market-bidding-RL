inputs = load("value.mat");
loads = inputs.loads;
sola_loads = inputs.solars;
wind_loads = inputs.winds;
max_new_loads = inputs.max_new_loads;
offers_qtys = inputs.offer_qtys;
offers_prcs = inputs.offer_prcs;
multi_price_sim(inputs.loads, inputs.solars, inputs.winds, inputs.max_new_loads, inputs.offer_qtys, inputs.offer_prcs)
