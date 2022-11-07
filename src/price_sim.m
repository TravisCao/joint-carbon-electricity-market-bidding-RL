% function [result] = priceSim(load,sola_load,wind_load,max_new_load)
function [result] = priceSim(load, sola_load, wind_load, max_new_load, offers_qty, offers_prc)
    mkt.OPF = 'AC';
    mkt.auction_type = 1;
    mpc = loadcase('t_auction_case');
    mpc.bus(:, 3) = mpc.bus(:, 3) * load(1);


    mpopt = mpoption('verbose', 0, 'out.all', 0);
%    mpopt = mpoption();

    mpc.bus(26, 2) = 2;
    mpc.bus(28, 2) = 2;

    gen_26 = [26, 5, 0, 60, -15, 1, 100, 1, max_new_load, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    gen_28 = [28, 5, 0, 60, -15, 1, 100, 1, max_new_load, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
%    display(size(mpc.gen(1, :)));
%    display(size(gen_26));
    gencost_26 = [1, 0, 0, 4, 0, 0, sola_load * 1/3, 0, sola_load * 2/3, 0, sola_load, 0];
    gencost_28 = [1, 0, 0, 4, 0, 0, wind_load * 1/3, 0, wind_load * 2/3, 0, wind_load, 0];
%    display(size(mpc.gencost(1, :)));
%    display(size(gencost_26));
    mpc.gen = [mpc.gen(1:6, :); gen_26; gen_28; mpc.gen(7:end, :)];
    mpc.gencost = [mpc.gencost(1:6, :); gencost_26; gencost_28; mpc.gencost(7:end, :)];
    mpc.gencost(1:6, 10) = [1440; 1200; 1248; 1296; 1344; 1392];

    mpc.gencost(1:6, 12) = [2880; 2880; 3168; 3456; 3144; 2832];

    % cha main node 0,1,5
    % dis main node 16,18,19

    bids.P.qty = [ ...
            0 0 0;
            0 0 0;
            0 0 0];
    bids.P.prc = [ ...
            0 0 0;
            0 0 0;
            0 0 0];

    offers.P.qty = [ ...
                offers_qty;
%                12 24 24;
%                12 24 24;
%                12 24 24;
%                12 24 24;
%                12 24 24;
%                12 24 24;
                sola_load * 1/3, sola_load * 1/3, sola_load * 1/3;
                wind_load * 1/3, wind_load * 1/3, wind_load * 1/3];

    offers.P.prc = [ ...
                offers_prc;
%                20. 50. 60.;
%                20. 40. 70.;
%                20. 42. 80.;
%                20. 44. 90.;
%                20. 46. 75.;
%                20. 48. 60.;
                0. 0. 0.;
                0. 0. 0.];

%    display(offers.P.qty);
%    display(offers.P.prc);

    [mpc_out, co, cb, f, dispatch, success, et] = runmarket(mpc, offers, bids, mkt, mpopt);
    % dispatch matrix defines [QUANTITY, PRICE, FCOST, VCOST, SCOST, PENALTY]
    % 0010 %   columns 1-6
    %    1  QUANTITY    quantity produced by generator in MW
    %    2  PRICE       market price for power produced by generator in $/MWh
    %    3  FCOST       fixed cost in $/MWh
    %    4  VCOST       variable cost in $/MWh
    %    5  SCOST       startup cost in $
    %    6  PENALTY     penalty cost in $ (not used)
    result.clear = dispatch(1:6, [1 2 4]);
    result.success = success;
end
