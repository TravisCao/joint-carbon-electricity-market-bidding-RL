python src/ddpg.py --exp-name "EM-test-multi-fix-new-1e-3-300-0.99-0.5-64" \
                --seed 42 \
                --total-timesteps 300 \
                --learning-rate 1e-3 \
                --buffer-size 300 \
                --gamma 0.99 \
                --tau 0.005 \
                --batch-size 64 \
                --exploration-noise 0.5 \
                --learning-starts 47 \
                --policy-frequency 2 \
                --noise-clip 0.5 \
                --save-model 


