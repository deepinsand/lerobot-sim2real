seed=3
python lerobot_sim2real/scripts/train_epo.py --env-id="SO100GraspCube-v1" --env-kwargs-json-path=env_config.json \
  --epo.seed=${seed} \
  --epo.num_envs=1 --epo.num-steps-per-latent-eval=8  --epo.num_latents=1 --epo.num_envs_per_latent=1\
  --epo.total_timesteps=100_000_000 \
  --epo.num_eval_envs=1 --epo.num-eval-steps=64 --epo.no-partial-reset \
  --epo.exp-name="ppo-SO100GraspCube-v1-rgb-${seed}" --epo.no-cuda