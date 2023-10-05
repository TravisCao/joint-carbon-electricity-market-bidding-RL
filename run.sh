python src/ddpg.py --exp-name "EM-test-multi" \
                --seed 42 \
                --total-timesteps 100000 \
                --learning-rate 3e-4 \
                --buffer-size 10000 \
                --gamma 0.99 \
                --tau 0.005 \
                --batch-size 256 \
                --exploration-noise 0.1 \
                --learning-starts 1000 \
                --policy-frequency 2 \
                --noise-clip 0.5


