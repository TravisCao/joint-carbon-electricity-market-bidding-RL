import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from env import ElecMktEnv, CarbMktEnv
from config import Config
from utils import get_logger

LOG = get_logger()


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    # parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    # fmt: on
    return args


def evaluate(
    # model_path: str,
    env,
    actor,
    qf,
    device: torch.device = torch.device("cpu"),
    exploration_noise: float = 0.1,
):
    assert len(env.mkts) == 1
    # actor = Model[0](
    #     Config.elec_obs_dim,
    #     Config.elec_act_dim,
    #     Config.elec_act_high,
    #     Config.elec_act_low,
    # ).to(device)
    # qf = Model[1](Config.elec_obs_dim, Config.elec_act_dim).to(device)
    # actor_params, qf_params = torch.load(model_path, map_location=device)
    # actor.load_state_dict(actor_params)
    actor.eval()
    # qf.load_state_dict(qf_params)
    qf.eval()

    # note: qf is not used in this script

    # run 100 days
    # here we record the episodic return of each day
    # in total 100 days
    obs = env.reset()
    episodic_returns = 0
    while 1:
        # if len(episodic_returns) < Config.n_trading_days:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(0, actor.action_scale * exploration_noise)
            actions = (
                actions.cpu().numpy().clip(Config.elec_act_low, Config.elec_act_high)
            )

        next_obs, _, _, infos = env.step(actions)

        # final_info means 1 day is finished
        if "final_info" in infos[0]:
            for info in infos:
                # print(
                #     f"eval_day={len(episodic_returns)}, episodic_return={info['final_info']['r']}"
                # )
                episodic_returns += info["final_info"]["r"]
            break
        obs = next_obs
    print(f"episodic_returns={episodic_returns}")
    return episodic_returns


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(
            obs_dim + act_dim,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_high, act_low):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (act_high - act_low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (act_high + act_low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import stable_baselines3 as sb3
    import matlab.engine

    engine = matlab.engine.start_matlab()

    args = parse_args()
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

    env = ElecMktEnv(Config, engine)

    actor = Actor(
        Config.elec_obs_dim,
        Config.elec_act_dim,
        Config.elec_act_high,
        Config.elec_act_low,
    ).to(device)
    qf1 = QNetwork(Config.elec_obs_dim, Config.elec_act_dim).to(device)
    qf1_target = QNetwork(Config.elec_obs_dim, Config.elec_act_dim).to(device)
    target_actor = Actor(
        Config.elec_obs_dim,
        Config.elec_act_dim,
        Config.elec_act_high,
        Config.elec_act_low,
    ).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    rb = ReplayBuffer(
        args.buffer_size,
        Config.elec_obs_space,
        Config.elec_act_space,
        device,
        n_envs=Config.num_mkt,
        handle_timeout_termination=False,
    )

    eval_config = Config()
    eval_config.num_mkt = 1
    eval_env = ElecMktEnv(eval_config, engine)
    episode_r = evaluate(
        eval_env,
        actor,
        qf1,
        device=device,
        exploration_noise=args.exploration_noise,
    )
    actor.train()
    qf1.train()
    writer.add_scalar("eval/episodic_r", episode_r, 0)

    eval_episodic_r_best = episode_r

    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs = env.reset()
    LOG.debug("reset obs: %s", obs)

    for global_step in range(args.total_timesteps):
        print(f"global_step: {global_step}")
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [Config.elec_act_space.sample() for _ in range(Config.num_mkt)]
            )
            LOG.debug("warm up")
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = (
                    actions.cpu()
                    .numpy()
                    .clip(Config.elec_act_low, Config.elec_act_high)
                )

        # LOG.info("action: %s", actions.tolist())
        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        next_obs, rewards, terminateds, infos = env.step(actions)

        LOG.debug("next obs: %s", next_obs)
        # LOG.info("r: %s", rewards.tolist())
        LOG.debug("timestep: %s", env.mkts[0].timestep)
        LOG.debug("terminated: %s", terminateds)

        # TRY NOT TO MODIFY: record rewards for plotting purposes

        if "final_info" in infos[0]:
            for info in infos:
                print(
                    f"global_step={global_step}, episodic_return={info['final_info']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["final_info"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["final_info"]["l"], global_step
                )
            # for i in info["final_info"]:
            #     print(f"global_step={global_step}, episodic_return={i['episode']['r']}")
            #     writer.add_scalar(
            #         "charts/episodic_return", i["episode"]["r"], global_step
            #     )
            #     writer.add_scalar(
            #         "charts/episodic_length", i["episode"]["l"], global_step
            #     )
            #     break

        rb.add(obs, next_obs, actions, rewards, terminateds, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 20 == 0:
                print("update")
                LOG.debug("losses/qf1_loss %s, %s", qf1_loss.item(), global_step)
                LOG.debug("losses/actor_loss %s, %s", actor_loss.item(), global_step)
                LOG.debug(
                    "losses/qf1_values %s, %s", qf1_a_values.mean().item(), global_step
                )
                LOG.debug("SPS: %s", int(global_step / (time.time() - start_time)))
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

            if args.save_model:
                # model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
                eval_config = Config()
                eval_config.num_mkt = 1
                eval_env = ElecMktEnv(eval_config, engine)
                episode_r = evaluate(
                    eval_env,
                    actor,
                    qf1,
                    device=device,
                    exploration_noise=args.exploration_noise,
                )
                if episode_r > eval_episodic_r_best:
                    eval_episodic_r_best = episode_r
                    path = f"runs/{run_name}/{args.exp_name}.cleanrl_model_best_{episode_r:.3f}"
                    torch.save((actor.state_dict(), qf1.state_dict()), path)
                    print(f"current best: {episode_r}. model saved to {path}")
                writer.add_scalar("eval/episodic_r", episode_r, global_step)
                actor.train()
                qf1.train()

    # if args.save_model:
    #     torch.save((actor.state_dict(), qf1.state_dict()), model_path)
    #     print(f"model saved to {model_path}")

    #     episode_r = evaluate(
    #         model_path,
    #         env,
    #         engine,
    #         Model=(Actor, QNetwork),
    #         device=device,
    #         exploration_noise=args.exploration_noise,
    #     )
    #     for idx, rewards in enumerate(episode_r):
    #         writer.add_scalar("eval/episodic_r", rewards, idx)

    # if args.upload_model:
    #     from cleanrl_utils.huggingface import push_to_hub

    #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #     push_to_hub(
    #         args,
    #         episodic_returns,
    #         repo_id,
    #         "DDPG",
    #         f"runs/{run_name}",
    #         f"videos/{run_name}-eval",
    #     )

    env.close()
    writer.close()
