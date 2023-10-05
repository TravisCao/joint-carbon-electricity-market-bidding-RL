function [results] = multi_price_sim(loads, sola_loads, wind_loads, max_new_loads, offers_qtys, offers_prcs)
    num_inputs = length(loads);
    results = cell(num_inputs, 1);
    for i = 1:num_inputs
        results{i} = price_sim(loads(i), sola_loads(i), wind_loads(i), max_new_loads(i), squeeze(offers_qtys(i, :, :)), squeeze(offers_prcs(i,:, :)));
    end
end