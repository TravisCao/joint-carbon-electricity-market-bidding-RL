import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import matlab.engine
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from config import Config
from env import CarbMktEnv, ElecMktEnv


def parse_args(jupyter=False):
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--carb", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, carbon market is included")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="RL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ElecMkt-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
    # parser.add_argument("--total-timesteps", type=int, default=Config.total_timesteps,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    # parser.add_argument("--num-envs", type=int, default=Config.num_mkt,
    parser.add_argument("--num-envs", type=int, default=50,
        help="the number of parallel game environments")
    # parser.add_argument("--num-steps", type=int, default=2048,
    # parser.add_argument("--num-steps", type=int, default=Config.n_timesteps,
    parser.add_argument("--num-steps", type=int, default=Config.n_timesteps * Config.n_trading_days,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--elec-num-steps", type=int, default=Config.n_timesteps,
        help="the number of steps to run in each elec market environment per policy rollout")
    parser.add_argument("--carb-num-steps", type=int, default=Config.n_trading_days,
        help="the number of steps to run in each carb market environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    if jupyter:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    args.elec_batch_size = int(args.num_envs * args.num_steps)
    args.carb_batch_size = int(args.num_envs * args.carb_num_steps)
    args.minibatch_size = int(args.elec_batch_size // args.num_minibatches)
    args.carb_minibatch_size = int(args.carb_batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, gamma, config=Config, engine=None):
    if env_id == "CarbMkt-v0":
        env = gym.make(env_id, config=config)
    else:
        env = gym.make(env_id, config=config, engine=engine)
    # env = gym.wrappers.FlattenObservation(
    # env
    # )  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


def make_env_func(env_id, gamma, config=Config):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def evaluate(
    elec_env,
    carb_env,
    elec_agent,
    carb_agent,
    device: torch.device = torch.device("cpu"),
):
    assert len(carb_env.mkts) == 1
    assert len(elec_env.mkts) == 1
    elec_agent.eval()
    carb_agent.eval()

    elec_obs_eval, _ = elec_env.reset()
    carb_obs_eval, _ = carb_env.reset()

    elec_episodic_returns = 0
    carb_episodic_returns = 0

    eval_day = 0
    while 1:
        # if len(episodic_returns) < Config.n_trading_days:
        with torch.no_grad():
            elec_action_eval = elec_agent.get_action_mean(
                torch.Tensor(elec_obs_eval).to(device)
            )

        elec_next_obs_eval, _, _, _, elec_info_eval = elec_env.step(
            elec_action_eval.cpu().numpy().flatten()
        )

        # final_info means 1 day is finished
        if "final_info" in elec_info_eval:
            elec_episodic_returns += elec_info_eval["episode"]["r"][0]
            carb_env.set_gen_emissions(elec_env.get_gen_emissions_day())

            with torch.no_grad():
                carb_action_eval = carb_agent.get_action_mean(
                    torch.Tensor(carb_obs_eval).to(device)
                )
            carb_next_obs_eval, carb_day_r, _, _, carb_info_eval = carb_env.step(
                carb_action_eval.cpu().numpy().flatten()
            )
            print(f"eval_day: {eval_day}, elec r: {elec_episodic_returns} ")
            print(f"carbon price: {carb_env.mkts[0].carbon_price_now:.3f}")
            print(f"carbon allowance: {carb_env.mkts[0].carbon_allowance_now} ")
            print(f"carb r:{carb_day_r}")
            print(f"carb rewards:{carb_env.mkts[0].rewards}")

            if "final_info" in carb_info_eval:
                carb_episodic_returns += carb_info_eval["episode"]["r"][0]
                # print(f"pay compliance: {car}")
                break

            carb_obs_eval = carb_next_obs_eval
            eval_day += 1

        elec_obs_eval = elec_next_obs_eval
    print(
        f"episodic_returns: elec: {elec_episodic_returns:.3f}, carb:{carb_episodic_returns:.3f}"
    )
    # add two  returns
    total_episode_r = elec_episodic_returns + carb_episodic_returns
    print(f"total episodic returns: {total_episode_r:.3f}")
    return elec_episodic_returns, carb_episodic_returns


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        if envs.action_space.shape != envs.single_action_space.shape:
            obs_shape = envs.single_observation_space.shape
            action_shape = envs.single_action_space.shape
        else:
            obs_shape = envs.observation_space.shape
            action_shape = envs.action_space.shape
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )

    def get_action_mean(self, x):
        action_mean = self.actor_mean(x)
        return action_mean


if __name__ == "__main__":
    jupyter = False
    engine = matlab.engine.start_matlab()
    args = parse_args(jupyter)

    config = Config()

    # eval env
    config_eval = config
    config_eval.num_mkt = 1
    elec_env_eval = make_env("ElecMkt-v0", args.gamma, config_eval, engine)
    carb_env_eval = make_env("CarbMkt-v0", args.gamma, config_eval)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    config.num_mkt = args.num_envs
    elec_envs = make_env("ElecMkt-v0", args.gamma, config, engine=engine)
    carb_envs = make_env("CarbMkt-v0", args.gamma, config, engine=engine)

    assert isinstance(
        elec_envs.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    elec_agent = Agent(elec_envs).to(device)
    elec_optimizer = optim.Adam(
        elec_agent.parameters(), lr=args.learning_rate, eps=1e-5
    )

    carb_agent = Agent(carb_envs).to(device)
    carb_optimizer = optim.Adam(
        carb_agent.parameters(), lr=args.learning_rate, eps=1e-5
    )

    # ALGO Logic: Storage setup
    elec_obs = torch.zeros(
        (args.num_steps, args.num_envs) + elec_envs.observation_space.shape
    ).to(device)
    elec_actions = torch.zeros(
        (args.num_steps, args.num_envs) + elec_envs.action_space.shape
    ).to(device)
    elec_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    elec_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    elec_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    elec_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    carb_obs = torch.zeros(
        (args.carb_num_steps, args.num_envs) + carb_envs.observation_space.shape
    ).to(device)
    carb_actions = torch.zeros(
        (args.carb_num_steps, args.num_envs) + carb_envs.action_space.shape
    ).to(device)
    carb_logprobs = torch.zeros((args.carb_num_steps, args.num_envs)).to(device)
    carb_rewards = torch.zeros((args.carb_num_steps, args.num_envs)).to(device)
    carb_dones = torch.zeros((args.carb_num_steps, args.num_envs)).to(device)
    carb_values = torch.zeros((args.carb_num_steps, args.num_envs)).to(device)

    elec_episode_r, carb_episode_r = evaluate(
        elec_env_eval, carb_env_eval, elec_agent, carb_agent, device=device
    )

    total_episode_r = elec_episode_r + carb_episode_r

    elec_agent.train()
    carb_agent.train()

    writer.add_scalar("eval/elec/episodic_r", elec_episode_r, 0)
    writer.add_scalar("eval/carb/episodic_r", carb_episode_r, 0)
    writer.add_scalar("eval/total_episodic_r", total_episode_r, 0)

    eval_episodic_r_best = elec_episode_r + carb_episode_r

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    elec_next_obs, _ = elec_envs.reset()
    elec_next_obs = elec_next_obs.reshape((-1,) + elec_envs.observation_space.shape)
    elec_next_obs = torch.Tensor(elec_next_obs).to(device)

    carb_next_obs, _ = carb_envs.reset()
    carb_next_obs = torch.Tensor(carb_next_obs).to(device)

    elec_next_done = torch.zeros(args.num_envs).to(device)
    carb_next_done = torch.zeros(args.num_envs).to(device)

    elec_num_updates = args.total_timesteps // args.elec_batch_size

    for update in range(1, elec_num_updates + 1):
        print("update:", update)
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / elec_num_updates
            lrnow = frac * args.learning_rate
            elec_optimizer.param_groups[0]["lr"] = lrnow
            carb_optimizer.param_groups[0]["lr"] = lrnow

        carb_steps = 0
        for step in range(0, args.num_steps):
            print("step:", step)
            global_step += 1 * args.num_envs
            elec_obs[step] = elec_next_obs
            elec_dones[step] = elec_next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                elec_action, logprob, _, value = elec_agent.get_action_and_value(
                    elec_next_obs
                )
                elec_values[step] = value.flatten()
            elec_actions[step] = elec_action
            # print("actions:", action.cpu().numpy().tolist())
            elec_logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            (
                elec_next_obs,
                elec_reward,
                elec_terminated,
                elec_truncated,
                elec_infos,
            ) = elec_envs.step(elec_action.cpu().numpy().flatten())

            elec_done = np.logical_or(elec_terminated, elec_truncated)
            elec_rewards[step] = torch.tensor(elec_reward).to(device).view(-1)
            # print("rewards:", reward.tolist())
            # print("infos:", infos)

            elec_next_obs, elec_next_done = torch.Tensor(elec_next_obs).to(
                device
            ), torch.Tensor(elec_done).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in elec_infos:
                continue
            else:
                # carbon market starts
                # set gen emission
                print(f"carb steps {carb_steps}")
                print("carbon market starts")
                gen_emissions = elec_envs.get_gen_emissions_day()
                carb_envs.set_gen_emissions(gen_emissions)

                carb_obs[carb_steps] = carb_next_obs
                carb_dones[carb_steps] = carb_next_done

                with torch.no_grad():
                    (
                        carb_action,
                        carb_logprob,
                        _,
                        carb_value,
                    ) = carb_agent.get_action_and_value(carb_next_obs)
                    carb_values[carb_steps] = carb_value.flatten()
                carb_actions[carb_steps] = carb_action
                carb_logprobs[carb_steps] = carb_logprob

                (
                    carb_next_obs,
                    carb_reward,
                    carb_terminated,
                    carb_truncated,
                    carb_infos,
                ) = carb_envs.step(carb_action.cpu().numpy().flatten())

                print(f"carb price: {carb_envs.mkts[0].carbon_price_now:.3f}")

                carb_done = np.logical_or(carb_terminated, carb_truncated)
                carb_rewards[carb_steps] = torch.tensor(carb_reward).to(device).view(-1)

                carb_next_obs, carb_next_done = torch.Tensor(carb_next_obs).to(
                    device
                ), torch.Tensor(carb_done).to(device)

                carb_steps += 1

                if "final_info" in carb_infos:
                    for i, elec_info in enumerate(elec_infos["final_info"]):
                        # Skip the envs that are not done
                        if elec_info is None:
                            continue
                        print(
                            f"global_step={global_step}, elec_episodic_return={elec_rewards[:, i].sum().item()}"
                        )

                        writer.add_scalar(
                            "charts/elec/episodic_return",
                            elec_rewards[:, i].sum().item(),
                            # elec_info["episode"]["r"],
                            global_step,
                        )
                        # writer.add_scalar(
                        #     "charts/episodic_length", info["episode"]["l"], global_step
                        # )
                    for carb_info in carb_infos["final_info"]:
                        # Skip the envs that are not done
                        if carb_info is None:
                            continue
                        print(
                            f"global_step={global_step}, carb_episodic_return={carb_info['episode']['r']}"
                        )

                        writer.add_scalar(
                            "charts/carb/episodic_return",
                            carb_info["episode"]["r"],
                            global_step,
                        )

        # bootstrap value if not done
        with torch.no_grad():
            elec_next_value = elec_agent.get_value(elec_next_obs).reshape(1, -1)
            elec_advantages = torch.zeros_like(elec_rewards).to(device)
            elec_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    elec_nextnonterminal = 1.0 - elec_next_done
                    elec_nextvalues = elec_next_value
                else:
                    elec_nextnonterminal = 1.0 - elec_dones[t + 1]
                    elec_nextvalues = elec_values[t + 1]
                elec_delta = (
                    elec_rewards[t]
                    + args.gamma * elec_nextvalues * elec_nextnonterminal
                    - elec_values[t]
                )
                elec_advantages[t] = elec_lastgaelam = (
                    elec_delta
                    + args.gamma
                    * args.gae_lambda
                    * elec_nextnonterminal
                    * elec_lastgaelam
                )
            elec_returns = elec_advantages + elec_values

            carb_next_value = carb_agent.get_value(carb_next_obs).reshape(1, -1)
            carb_advantages = torch.zeros_like(carb_rewards).to(device)
            carb_lastgaelam = 0
            for t in reversed(range(args.carb_num_steps)):
                if t == args.carb_num_steps - 1:
                    carb_nextnonterminal = 1.0 - carb_next_done
                    carb_nextvalues = carb_next_value
                else:
                    carb_nextnonterminal = 1.0 - carb_dones[t + 1]
                    carb_nextvalues = carb_values[t + 1]
                carb_delta = (
                    carb_rewards[t]
                    + args.gamma * carb_nextvalues * carb_nextnonterminal
                    - carb_values[t]
                )
                carb_advantages[t] = carb_lastgaelam = (
                    carb_delta
                    + args.gamma
                    * args.gae_lambda
                    * carb_nextnonterminal
                    * carb_lastgaelam
                )
            carb_returns = carb_advantages + carb_values

        # flatten the batch
        elec_b_obs = elec_obs.reshape((-1,) + elec_envs.single_observation_space.shape)
        elec_b_logprobs = elec_logprobs.reshape(-1)
        elec_b_actions = elec_actions.reshape(
            (-1,) + elec_envs.single_action_space.shape
        )
        elec_b_advantages = elec_advantages.reshape(-1)
        elec_b_returns = elec_returns.reshape(-1)
        elec_b_values = elec_values.reshape(-1)

        # Optimizing the policy and value network
        elec_b_inds = np.arange(args.elec_batch_size)
        elec_clipfracs = []

        # flatten the batch
        carb_b_obs = carb_obs.reshape((-1,) + carb_envs.single_observation_space.shape)
        carb_b_logprobs = carb_logprobs.reshape(-1)
        carb_b_actions = carb_actions.reshape(
            (-1,) + carb_envs.single_action_space.shape
        )
        carb_b_advantages = carb_advantages.reshape(-1)
        carb_b_returns = carb_returns.reshape(-1)
        carb_b_values = carb_values.reshape(-1)

        # Optimizing the policy and value network
        carb_b_inds = np.arange(args.carb_batch_size)
        carb_clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(elec_b_inds)
            for start in range(0, args.elec_batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = elec_b_inds[start:end]

                (
                    _,
                    elec_newlogprob,
                    elec_entropy,
                    elec_newvalue,
                ) = elec_agent.get_action_and_value(
                    elec_b_obs[mb_inds], elec_b_actions[mb_inds]
                )
                elec_logratio = elec_newlogprob - elec_b_logprobs[mb_inds]
                elec_ratio = elec_logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    elec_old_approx_kl = (-elec_logratio).mean()
                    elec_approx_kl = ((elec_ratio - 1) - elec_logratio).mean()
                    elec_clipfracs += [
                        ((elec_ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                elec_mb_advantages = elec_b_advantages[mb_inds]
                if args.norm_adv:
                    elec_mb_advantages = (
                        elec_mb_advantages - elec_mb_advantages.mean()
                    ) / (elec_mb_advantages.std() + 1e-8)

                # Policy loss
                elec_pg_loss1 = -elec_mb_advantages * elec_ratio
                elec_pg_loss2 = -elec_mb_advantages * torch.clamp(
                    elec_ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                elec_pg_loss = torch.max(elec_pg_loss1, elec_pg_loss2).mean()

                # Value loss
                elec_newvalue = elec_newvalue.view(-1)
                if args.clip_vloss:
                    elec_v_loss_unclipped = (
                        elec_newvalue - elec_b_returns[mb_inds]
                    ) ** 2
                    elec_v_clipped = elec_b_values[mb_inds] + torch.clamp(
                        elec_newvalue - elec_b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    elec_v_loss_clipped = (
                        elec_v_clipped - elec_b_returns[mb_inds]
                    ) ** 2
                    elec_v_loss_max = torch.max(
                        elec_v_loss_unclipped, elec_v_loss_clipped
                    )
                    elec_v_loss = 0.5 * elec_v_loss_max.mean()
                else:
                    elec_v_loss = (
                        0.5 * ((elec_newvalue - elec_b_returns[mb_inds]) ** 2).mean()
                    )

                elec_entropy_loss = elec_entropy.mean()
                elec_loss = (
                    elec_pg_loss
                    - args.ent_coef * elec_entropy_loss
                    + elec_v_loss * args.vf_coef
                )

                elec_optimizer.zero_grad()
                elec_loss.backward()
                nn.utils.clip_grad_norm_(elec_agent.parameters(), args.max_grad_norm)
                elec_optimizer.step()

            if args.target_kl is not None:
                if elec_approx_kl > args.target_kl:
                    break

        elec_y_pred, elec_y_true = (
            elec_b_values.cpu().numpy(),
            elec_b_returns.cpu().numpy(),
        )
        elec_var_y = np.var(elec_y_true)
        elec_explained_var = (
            np.nan
            if elec_var_y == 0
            else 1 - np.var(elec_y_true - elec_y_pred) / elec_var_y
        )

        # Optimizing the policy and value network
        carb_b_inds = np.arange(args.carb_batch_size)
        carb_clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(carb_b_inds)
            for start in range(0, args.carb_batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = carb_b_inds[start:end]

                (
                    _,
                    carb_newlogprob,
                    carb_entropy,
                    carb_newvalue,
                ) = carb_agent.get_action_and_value(
                    carb_b_obs[mb_inds], carb_b_actions[mb_inds]
                )
                carb_logratio = carb_newlogprob - carb_b_logprobs[mb_inds]
                carb_ratio = carb_logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    carb_old_approx_kl = (-carb_logratio).mean()
                    carb_approx_kl = ((carb_ratio - 1) - carb_logratio).mean()
                    carb_clipfracs += [
                        ((carb_ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                carb_mb_advantages = carb_b_advantages[mb_inds]
                if args.norm_adv:
                    carb_mb_advantages = (
                        carb_mb_advantages - carb_mb_advantages.mean()
                    ) / (carb_mb_advantages.std() + 1e-8)

                # Policy loss
                carb_pg_loss1 = -carb_mb_advantages * carb_ratio
                carb_pg_loss2 = -carb_mb_advantages * torch.clamp(
                    carb_ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                carb_pg_loss = torch.max(carb_pg_loss1, carb_pg_loss2).mean()

                # Value loss
                carb_newvalue = carb_newvalue.view(-1)
                if args.clip_vloss:
                    carb_v_loss_unclipped = (
                        carb_newvalue - carb_b_returns[mb_inds]
                    ) ** 2
                    carb_v_clipped = carb_b_values[mb_inds] + torch.clamp(
                        carb_newvalue - carb_b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    carb_v_loss_clipped = (
                        carb_v_clipped - carb_b_returns[mb_inds]
                    ) ** 2
                    carb_v_loss_max = torch.max(
                        carb_v_loss_unclipped, carb_v_loss_clipped
                    )
                    carb_v_loss = 0.5 * carb_v_loss_max.mean()
                else:
                    carb_v_loss = (
                        0.5 * ((carb_newvalue - carb_b_returns[mb_inds]) ** 2).mean()
                    )

                carb_entropy_loss = carb_entropy.mean()
                carb_loss = (
                    carb_pg_loss
                    - args.ent_coef * carb_entropy_loss
                    + carb_v_loss * args.vf_coef
                )

                carb_optimizer.zero_grad()
                carb_loss.backward()
                nn.utils.clip_grad_norm_(carb_agent.parameters(), args.max_grad_norm)
                carb_optimizer.step()

            if args.target_kl is not None:
                if carb_approx_kl > args.target_kl:
                    break

        carb_y_pred, carb_y_true = (
            carb_b_values.cpu().numpy(),
            carb_b_returns.cpu().numpy(),
        )
        carb_var_y = np.var(carb_y_true)
        carb_explained_var = (
            np.nan
            if carb_var_y == 0
            else 1 - np.var(carb_y_true - carb_y_pred) / carb_var_y
        )

        if args.save_model:
            print("start evaluation")
            elec_episode_r, carb_episode_r = evaluate(
                elec_env_eval, carb_env_eval, elec_agent, carb_agent, device=device
            )
            total_episode_r = elec_episode_r + carb_episode_r
            print(f"elec_episode_r: {elec_episode_r:.3f}")
            print(f"carb_episode_r: {carb_episode_r:.3f}")
            print(f"total_episode_r: {total_episode_r:.3f}")
            if total_episode_r > eval_episodic_r_best:
                eval_episodic_r_best = total_episode_r
                path = f"runs/{run_name}/{args.exp_name}.rl_twos_model_best_{eval_episodic_r_best:.3f}"
                torch.save((elec_agent.state_dict(), carb_agent.state_dict()), path)
                print(f"current best: {total_episode_r}. model saved to {path}")

            writer.add_scalar("eval/elec/episodic_r", elec_episode_r, global_step)
            writer.add_scalar("eval/carb/episodic_r", carb_episode_r, global_step)
            writer.add_scalar("eval/total_episodic_r", total_episode_r, global_step)
            elec_agent.train()
            carb_agent.train()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/elec/learning_rate",
            elec_optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar("losses/elec/value_loss", elec_v_loss.item(), global_step)
        writer.add_scalar("losses/elec/policy_loss", elec_pg_loss.item(), global_step)
        writer.add_scalar("losses/elec/entropy", elec_entropy_loss.item(), global_step)
        writer.add_scalar(
            "losses/elec/old_approx_kl", elec_old_approx_kl.item(), global_step
        )
        writer.add_scalar("losses/elec/approx_kl", elec_approx_kl.item(), global_step)
        writer.add_scalar("losses/elec/clipfrac", np.mean(elec_clipfracs), global_step)
        writer.add_scalar(
            "losses/elec/explained_variance", elec_explained_var, global_step
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/carb/learning_rate",
            carb_optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar("losses/carb/value_loss", carb_v_loss.item(), global_step)
        writer.add_scalar("losses/carb/policy_loss", carb_pg_loss.item(), global_step)
        writer.add_scalar("losses/carb/entropy", carb_entropy_loss.item(), global_step)
        writer.add_scalar(
            "losses/carb/old_approx_kl", carb_old_approx_kl.item(), global_step
        )
        writer.add_scalar("losses/carb/approx_kl", carb_approx_kl.item(), global_step)
        writer.add_scalar("losses/carb/clipfrac", np.mean(carb_clipfracs), global_step)
        writer.add_scalar(
            "losses/carb/explained_variance", carb_explained_var, global_step
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    elec_envs.close()
    writer.close()
