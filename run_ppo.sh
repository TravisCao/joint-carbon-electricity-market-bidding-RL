#!/bin/bash

# Define the variables
PY_FILE="src/ppo.py"
NUM_ENVS=4
UPDATE_EPOCHS=10
SEED=203
N_TRADING_DAYS=50
ENT_COEF=0.5
NUM_MINIBATCHES=32

# Construct the exp-name to reflect the settings
EXP_NAME="MM-mkt-${N_TRADING_DAYS}days-envs${NUM_ENVS}-epochs${UPDATE_EPOCHS}-seed${SEED}-ent${ENT_COEF}-minibatches${NUM_MINIBATCHES}"

# Execute the python script with the variables
python $PY_FILE --exp-name "$EXP_NAME" \
				--total-timesteps 1000000 \
				--n-trading-days $N_TRADING_DAYS \
				--ent-coef $ENT_COEF \
				--num-envs $NUM_ENVS \
				--num-minibatches $NUM_MINIBATCHES \
				--update-epochs $UPDATE_EPOCHS \
				--seed $SEED \
				--save-model True
