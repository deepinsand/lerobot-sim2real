"""
EPO (Evolutionary Policy Optimization) implementation for state-based RL in ManiSkill.
"""
from collections import defaultdict
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# EPO specific imports
from evolutionary_policy_optimization import (
    LatentGenePool,
    Actor as BaseEPOActor, # Renamed to avoid potential conflicts if Actor is defined elsewhere
    Critic as BaseEPOCritic # Renamed
)
import torch.optim as optim # Import optim


@dataclass
class EPOArgs:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO_EPO_STATE"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "rgb_array"
    """the environment rendering mode"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_kwargs: dict = field(default_factory=dict)
    """extra environment kwargs to pass to the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps_per_latent_eval: int = 200 # Renamed from num_steps
    """the number of steps to run in each environment for evaluating a single latent vector"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of generations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of generations"""

    # EPO specific arguments
    num_latents: int = 128
    """Number of latent vectors (genes) in the population"""
    dim_latent: int = 32
    """Dimension of each latent vector"""
    actor_mlp_depth: int = 3
    """MLP depth for the EPO actor"""
    critic_mlp_depth: int = 3
    """MLP depth for the EPO critic"""
    actor_dim: int = 256
    """Hidden dimension for the EPO actor MLP"""
    critic_dim: int = 256
    """Hidden dimension for the EPO critic MLP"""
    frac_elitism: float = 0.1
    """Fraction of the population to preserve from mutation (elitism)."""
    frac_natural_selected: float = 0.25
    """Fraction of the population to be replaced by offspring of the fittest individuals."""
    mutation_strength: float = 1.0
    """Standard deviation of the Gaussian noise added during mutation."""
    apply_ga_every: int = 2
    """Apply the genetic algorithm every N generations."""

    # PPO specific arguments
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """The K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = 0.1
    """the target KL divergence threshold"""
    anneal_lr: bool = False
    """Toggle learning rate annealing"""
    finite_horizon_gae: bool = False
    """Whether to use finite horizon GAE calculation variant"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of generations (computed in runtime, renamed from num_iterations)"""


class EPOAgent(nn.Module):
    def __init__(self, envs, args: EPOArgs, device):
        super().__init__()
        self.device = device
        
        dim_state = np.array(envs.single_observation_space.shape).prod()
        self.num_actions = np.prod(envs.single_action_space.shape)
        
        # Use a learnable log_std, similar to the original PPO agent
        self.actor_logstd = nn.Parameter(torch.ones(1, self.num_actions, device=device) * -0.5)

        self.latent_pool = LatentGenePool(
            num_latents=args.num_latents,
            dim_latent=args.dim_latent,
            frac_elitism=args.frac_elitism,
            frac_natural_selected=args.frac_natural_selected,
            mutation_strength=args.mutation_strength,
            apply_genetic_algorithm_every=args.apply_ga_every,
        )
        
        self.actor = BaseEPOActor(
            dim_state=dim_state,
            dim=args.actor_dim,
            mlp_depth=args.actor_mlp_depth,
            num_actions=self.num_actions,
            dim_latent=args.dim_latent
        )
        
        self.critic = BaseEPOCritic(
            dim_state=dim_state,
            dim=args.critic_dim,
            mlp_depth=args.critic_mlp_depth,
            dim_latent=args.dim_latent
        )

    def _prepare_latents(self, features: torch.Tensor, latent_id: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Fetches latent vectors and expands them to match the batch size of features.
        """
        current_latents = self.latent_pool(latent_id=latent_id)
        
        # actor expects features: (B, D_state), latent_batch: (B, D_latent)
        if features.ndim > 1: # Batched features
            batch_size = features.shape[0]
            if current_latents.ndim == 1 and batch_size > 1: # Single latent for a batch
                return current_latents.unsqueeze(0).expand(batch_size, -1)
        return current_latents

    def get_action(self, obs, latent_id: int, deterministic: bool = True):
        # obs is already a flat tensor for state-based env
        latent_batch = self._prepare_latents(obs, latent_id)
        action_mean = self.actor(obs, latent_batch)
        if deterministic:
            act = action_mean
        else:
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            distribution = torch.distributions.Normal(action_mean, action_std)
            act = distribution.sample()
        return act.detach()

    def get_value(self, obs, latent_id: int):
        latent_batch = self._prepare_latents(obs, latent_id)
        return self.critic(obs, latent_batch)

    def genetic_step(self, fitness_scores: torch.Tensor):
        self.latent_pool.genetic_algorithm_step(fitness_scores)

    def get_action_and_value(self, obs, latent_id: Union[int, torch.Tensor], action: Optional[torch.Tensor] = None):
        latent_batch = self._prepare_latents(obs, latent_id)

        action_mean = self.actor(obs, latent_batch)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        value = self.critic(obs, latent_batch)

        distribution = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            sampled_action = distribution.sample()
        else:
            sampled_action = action
        
        log_prob = distribution.log_prob(sampled_action).sum(axis=-1)
        entropy = distribution.entropy().sum(axis=-1)
        
        return sampled_action, log_prob, entropy, value

class GenerationBasedTrigger:
    def __init__(self, freq):
        self.freq = freq
        self.current_generation = 0

    def __call__(self, episode_id): # episode_id is passed by RecordEpisode
        if self.freq is None or self.freq <= 0:
            return False
        return self.current_generation > 0 and self.current_generation % self.freq == 0

    def set_generation(self, generation):
        self.current_generation = generation

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

def train(args: EPOArgs):
    # Calculate number of generations based on total_timesteps
    # Each latent is evaluated for num_steps_per_latent_eval across num_envs
    env_steps_per_generation = args.num_latents * args.num_envs * args.num_steps_per_latent_eval
    if env_steps_per_generation == 0 :
        raise ValueError("env_steps_per_generation is zero. Check num_latents, num_envs, or num_steps_per_latent_eval.")
    args.num_iterations = args.total_timesteps // env_steps_per_generation # num_iterations is num_generations

    # Calculate batch_size and minibatch_size for PPO
    # This batch_size is for the PPO update, accumulating data from all latents
    args.batch_size = int(args.num_latents * args.num_envs * args.num_steps_per_latent_eval)
    if args.num_minibatches > 0:
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
    else: 
        args.minibatch_size = args.batch_size
        if args.batch_size > 0: # Avoid division by zero if batch_size is 0
            args.num_minibatches = 1 # Ensure num_minibatches is at least 1 if minibatch_size is batch_size

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(
        obs_mode="state", render_mode=args.render_mode, sim_backend=("physx_cuda" if torch.cuda.is_available() and args.cuda else "physx_cpu"),
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    env_kwargs.update(args.env_kwargs)

    # Create base single environments
    base_eval_env = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    base_train_env = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    if isinstance(base_train_env.action_space, gym.spaces.Dict):
        wrapped_single_train_env = FlattenActionSpaceWrapper(base_train_env)
        wrapped_single_eval_env = FlattenActionSpaceWrapper(base_eval_env)
    else:
        wrapped_single_train_env = base_train_env
        wrapped_single_eval_env = base_eval_env

    training_video_trigger = None
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            training_video_trigger = GenerationBasedTrigger(args.save_train_video_freq)
            wrapped_single_train_env = RecordEpisode(
                wrapped_single_train_env, 
                output_dir=f"runs/{run_name}/train_videos", 
                save_trajectory=False, 
                save_video_trigger=training_video_trigger, 
                max_steps_per_video=args.num_steps_per_latent_eval, 
                video_fps=wrapped_single_train_env.control_freq if hasattr(wrapped_single_train_env, 'control_freq') else 30 # Use actual control_freq
            )
        wrapped_single_eval_env = RecordEpisode(wrapped_single_eval_env, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=wrapped_single_eval_env.control_freq if hasattr(wrapped_single_eval_env, 'control_freq') else 30, info_on_video=True)

    envs = ManiSkillVectorEnv(wrapped_single_train_env, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(wrapped_single_eval_env, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True, # Note: wandb save_code might have issues with large projects
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    # These buffers will store data for one full generation (all latents) for PPO update
    ppo_buffer_num_samples = args.num_latents * args.num_steps_per_latent_eval * args.num_envs
    
    # Check if ppo_buffer_num_samples is zero before creating tensors
    if ppo_buffer_num_samples == 0:
        raise ValueError("ppo_buffer_num_samples is zero. Check EPO args: num_latents, num_steps_per_latent_eval, num_envs.")

    ppo_obs = torch.zeros((ppo_buffer_num_samples,) + envs.single_observation_space.shape, device=device)
    ppo_actions = torch.zeros((ppo_buffer_num_samples,) + envs.single_action_space.shape, device=device)
    ppo_logprobs = torch.zeros(ppo_buffer_num_samples, device=device)
    ppo_advantages = torch.zeros(ppo_buffer_num_samples, device=device)
    ppo_returns = torch.zeros(ppo_buffer_num_samples, device=device)
    ppo_values = torch.zeros(ppo_buffer_num_samples, device=device)
    ppo_latent_ids = torch.zeros(ppo_buffer_num_samples, dtype=torch.long, device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"Num Generations (args.num_iterations): {args.num_iterations}, Num Envs: {args.num_envs}, Num Eval Envs: {args.num_eval_envs}")
    print(f"PPO Batch Size (total samples per gen): {args.batch_size}, PPO Minibatch Size: {args.minibatch_size}, PPO Update Epochs: {args.update_epochs}")
    print(f"####")
    agent = EPOAgent(envs=envs,
                     args=args,
                     device=device).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    action_space_low, action_space_high = torch.from_numpy(envs.single_action_space.low).to(device), torch.from_numpy(envs.single_action_space.high).to(device)
    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)

    for generation in range(1, args.num_iterations + 1):
        generation_start_time = time.time()
        print(f"Generation: {generation}/{args.num_iterations}, Global Step: {global_step}")
        if training_video_trigger:
            training_video_trigger.set_generation(generation)
        agent.eval()
        # Variables for PPO data collection for the entire generation
        current_ppo_buffer_idx = 0
        generation_fitness_scores = torch.zeros(args.num_latents, device=device)

        if generation % args.eval_freq == 0 or generation == 1: # Evaluate at start and then periodically
            print("Evaluating")
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    # Using latent_id=0 for evaluation (can be changed to best_latent_id if tracked)
                    eval_action = agent.get_action(eval_obs, latent_id=0, deterministic=True) 
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_action)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        if args.save_model and (generation % args.eval_freq == 0 or generation == 1 or generation == args.num_iterations):
            model_path = f"runs/{run_name}/ckpt_gen_{generation}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (generation - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        rollout_time = time.perf_counter()

        # Latent Evaluation Loop
        for latent_idx in range(args.num_latents):
            # Buffers for the current latent's rollout data
            latent_obs_buffer = torch.zeros((args.num_steps_per_latent_eval, args.num_envs) + envs.single_observation_space.shape, device=device)
            latent_actions_buffer = torch.zeros((args.num_steps_per_latent_eval, args.num_envs) + envs.single_action_space.shape, device=device)
            latent_logprobs_buffer = torch.zeros((args.num_steps_per_latent_eval, args.num_envs), device=device)
            latent_rewards_buffer = torch.zeros((args.num_steps_per_latent_eval, args.num_envs), device=device)
            latent_dones_buffer = torch.zeros((args.num_steps_per_latent_eval, args.num_envs), device=device)
            latent_values_buffer = torch.zeros((args.num_steps_per_latent_eval, args.num_envs), device=device)
            
            current_latent_total_reward = 0.0

            for step in range(args.num_steps_per_latent_eval):
                global_step += args.num_envs # Global step increments for each env step
                latent_obs_buffer[step] = next_obs # next_obs is from previous step/latent
                latent_dones_buffer[step] = next_done # next_done is from previous step/latent

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs, latent_id=latent_idx)
                    latent_values_buffer[step] = value.flatten()
                latent_actions_buffer[step] = action
                latent_logprobs_buffer[step] = logprob

                obs_after_step, reward, terminations, truncations, infos = envs.step(clip_action(action))
                current_done_flags = torch.logical_or(terminations, truncations).to(torch.float32)
                latent_rewards_buffer[step] = reward.view(-1) * args.reward_scale
                current_latent_total_reward += latent_rewards_buffer[step].sum().item()

                if "final_info" in infos and logger:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    for k_info, v_info in final_info["episode"].items():
                        if done_mask.any():
                            logger.add_scalar(f"train_latent_{latent_idx}/{k_info}", v_info[done_mask].float().mean(), global_step)
                
                next_obs = obs_after_step
                next_done = current_done_flags

            generation_fitness_scores[latent_idx] = current_latent_total_reward / (args.num_steps_per_latent_eval * args.num_envs) # Avg reward as fitness

            # GAE Calculation for this latent's data
            with torch.no_grad():
                # Value of the state after the last step of this latent's rollout
                next_value_for_gae = agent.get_value(next_obs, latent_id=latent_idx).reshape(1, -1)
                advantages_for_latent = torch.zeros_like(latent_rewards_buffer).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps_per_latent_eval)):
                    if t == args.num_steps_per_latent_eval - 1:
                        nextnonterminal = 1.0 - next_done # next_done is after this latent's last step
                        nextvalues_gae = next_value_for_gae
                    else:
                        nextnonterminal = 1.0 - latent_dones_buffer[t + 1]
                        nextvalues_gae = latent_values_buffer[t + 1]
                    
                    delta = latent_rewards_buffer[t] + args.gamma * nextvalues_gae * nextnonterminal - latent_values_buffer[t]
                    advantages_for_latent[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns_for_latent = advantages_for_latent + latent_values_buffer

            # Store this latent's data into the generation-wide PPO buffers
            start_idx = current_ppo_buffer_idx
            end_idx = start_idx + args.num_steps_per_latent_eval * args.num_envs
            
            ppo_obs[start_idx:end_idx] = latent_obs_buffer.reshape((-1,) + envs.single_observation_space.shape)
            ppo_actions[start_idx:end_idx] = latent_actions_buffer.reshape((-1,) + envs.single_action_space.shape)
            ppo_logprobs[start_idx:end_idx] = latent_logprobs_buffer.reshape(-1)
            ppo_advantages[start_idx:end_idx] = advantages_for_latent.reshape(-1)
            ppo_returns[start_idx:end_idx] = returns_for_latent.reshape(-1)
            ppo_values[start_idx:end_idx] = latent_values_buffer.reshape(-1)
            ppo_latent_ids[start_idx:end_idx] = torch.full(((args.num_steps_per_latent_eval * args.num_envs),), latent_idx, device=device, dtype=torch.long)
            current_ppo_buffer_idx = end_idx

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            # Genetic Algorithm Step
            agent.genetic_step(generation_fitness_scores)
            if logger:
                logger.add_scalar("charts/mean_fitness", generation_fitness_scores.mean().item(), global_step)
                logger.add_scalar("charts/max_fitness", generation_fitness_scores.max().item(), global_step)

            # PPO Update using data from all latents
            # Data is already in ppo_obs, ppo_actions, etc. (effectively b_obs, b_actions)
            b_obs_ppo = ppo_obs
            b_actions_ppo = ppo_actions
            b_logprobs_ppo = ppo_logprobs
            b_advantages_ppo = ppo_advantages
            b_returns_ppo = ppo_returns
            b_values_ppo = ppo_values
            b_latent_ids_ppo = ppo_latent_ids
            

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size) # args.batch_size is total samples in this generation
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs_ppo[mb_inds], latent_id=b_latent_ids_ppo[mb_inds], action=b_actions_ppo[mb_inds])
                logratio = newlogprob - b_logprobs_ppo[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages_ppo[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns_ppo[mb_inds]) ** 2
                    v_clipped = b_values_ppo[mb_inds] + torch.clamp(
                        newvalue - b_values_ppo[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns_ppo[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns_ppo[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        y_pred, y_true = b_values_ppo.cpu().numpy(), b_returns_ppo.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if logger:
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            if old_approx_kl is not None : logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            if approx_kl is not None : logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            if clipfracs : logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            
            total_generation_time = time.time() - generation_start_time
            sps = int(env_steps_per_generation / total_generation_time)
            print(f"SPS: {sps}")
            logger.add_scalar("charts/SPS_generation", sps, global_step)
            logger.add_scalar("time/generation_time", total_generation_time, global_step)
            logger.add_scalar("time/update_time_per_gen", update_time, global_step)
            logger.add_scalar("time/rollout_time_per_gen", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps_per_gen", env_steps_per_generation / rollout_time if rollout_time > 0 else 0, global_step)
            for k_time, v_time in cumulative_times.items():
                logger.add_scalar(f"time/total_{k_time}", v_time, global_step)
                
    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None: logger.close()

if __name__ == "__main__":
    args = tyro.cli(EPOArgs)
    train(args=args)
