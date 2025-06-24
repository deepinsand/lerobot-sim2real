seed=3
python lerobot_sim2real/scripts/train_ppo.py --env-id="PushCube-v1" \
  --ppo.seed=${seed} \
  --ppo.update_epochs=2 --ppo.num_minibatches=1 \
  --ppo.num_envs=1 --ppo.num_steps=16 \
  --ppo.total_timesteps=600_000 \
  --ppo.num_eval_envs=1 --ppo.num-eval-steps=64 --ppo.no-partial-reset \
  --ppo.exp-name="ppo-PushCube-v1-rgb-${seed}" --ppo.no-cuda
  