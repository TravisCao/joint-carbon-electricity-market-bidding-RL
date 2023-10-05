python src/ddpg.py --exp-name "EM-test" \
                --seed 42 \
                --total-timesteps 10000 \
                --learning-rate 3e-4 \
                --buffer-size 1000 \
                --gamma 0.99 \
                --tau 0.005 \
                --batch-size 32 \
                --exploration-noise 0.1 \
                --learning-starts 100 \
                --policy-frequency 2 \
                --noise-clip 0.5


