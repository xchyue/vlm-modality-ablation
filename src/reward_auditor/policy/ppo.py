"""PPO trainer for the Reward Hacking Zoo (Part 2).

Vendored from CleanRL's `ppo_continuous_action.py` (MIT-licensed,
https://github.com/vwxyzjn/cleanrl) and adapted to:

  - build envs via `reward_auditor.envs.make_env(task, variant)` instead of
    `gym.make(env_id)`;
  - save checkpoints to `data/policies/{task}_{variant}_seed{N}.pt`, embedding
    obs running-mean/var so eval can be done without re-instantiating wrappers;
  - dump per-iteration metrics to a JSON sidecar for Part 5 plotting;
  - drop W&B / TensorBoard — stdout + JSON is enough for our scale.

Reference for the algorithm and numerical recipe:
    Huang et al., "The 37 Implementation Details of PPO". ICLR Blog 2022.

Run:
    uv run python -m reward_auditor.policy.ppo \\
        --task halfcheetah --variant v1_ground_truth --seed 0 \\
        --total-timesteps 1000000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from reward_auditor.envs import make_env as _make_reward_env

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class PPOConfig:
    """All hyperparameters. Defaults match CleanRL `ppo_continuous_action.py`."""

    task: str
    variant: str
    seed: int = 0
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    torch_deterministic: bool = True
    cuda: bool = True
    save_dir: str = "data/policies"
    reward_weights: dict[str, float] | None = None

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return self.total_timesteps // self.batch_size

    @property
    def run_name(self) -> str:
        return f"{self.task}_{self.variant}_seed{self.seed}"


# --------------------------------------------------------------------------- #
# Actor-critic agent
# --------------------------------------------------------------------------- #


def _layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Standard CleanRL actor-critic: 2×64 Tanh, separate critic and actor mean,
    learnable global log-std for the action distribution."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


# --------------------------------------------------------------------------- #
# Env construction
# --------------------------------------------------------------------------- #


def _build_train_env(cfg: PPOConfig, idx: int) -> Callable[[], gym.Env]:
    """Return a thunk that builds one training env (make_env + standard PPO wrappers).

    RecordEpisodeStatistics is placed before NormalizeReward so the logged
    returns are the raw component-weighted reward, not the normalized one.
    """

    def thunk() -> gym.Env:
        env = _make_reward_env(
            cfg.task,
            cfg.variant,
            weights=cfg.reward_weights,
            seed=cfg.seed + idx,
            render_mode=None,
        )
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10.0, 10.0),
            env.observation_space,
        )
        env = gym.wrappers.NormalizeReward(env, gamma=cfg.gamma)
        env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10.0, 10.0)))
        return env

    return thunk


def _extract_obs_rms(envs: gym.vector.SyncVectorEnv) -> tuple[np.ndarray, np.ndarray]:
    """Pull running mean/var out of NormalizeObservation in envs.envs[0]."""
    env = envs.envs[0]
    while not isinstance(env, gym.wrappers.NormalizeObservation):
        if not hasattr(env, "env"):
            raise RuntimeError("NormalizeObservation wrapper not found in env stack.")
        env = env.env
    return np.asarray(env.obs_rms.mean), np.asarray(env.obs_rms.var)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def train(cfg: PPOConfig) -> dict:
    """Train PPO for `cfg.total_timesteps`; write `{run_name}.pt` + `_metrics.json`."""

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{cfg.run_name}.pt"
    metrics_path = save_dir / f"{cfg.run_name}_metrics.json"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda" if cfg.cuda and torch.cuda.is_available() else "cpu")
    print(
        f"[ppo] {cfg.run_name}  device={device}  "
        f"iterations={cfg.num_iterations}  batch={cfg.batch_size}"
    )

    envs = gym.vector.SyncVectorEnv([_build_train_env(cfg, i) for i in range(cfg.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "continuous action space required"
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((cfg.num_steps, cfg.num_envs, act_dim), device=device)
    logprobs_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    rewards_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    dones_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    values_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(cfg.num_envs, device=device)
    metrics: list[dict] = []

    for it in range(1, cfg.num_iterations + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (it - 1) / cfg.num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

            # Gymnasium 1.0 RecordEpisodeStatistics: per-env arrays in info["episode"].
            ep_info = info.get("episode")
            if ep_info is not None:
                rs = np.atleast_1d(ep_info["r"])
                ls = np.atleast_1d(ep_info["l"])
                mask_raw = info.get("_episode")
                mask = (
                    np.atleast_1d(mask_raw)
                    if mask_raw is not None
                    else np.ones_like(rs, dtype=bool)
                )
                for r_, l_, m_ in zip(rs, ls, mask, strict=False):
                    if m_:
                        ep_returns.append(float(r_))
                        ep_lengths.append(int(l_))

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf, device=device)
            last_gae = torch.zeros(cfg.num_envs, device=device)
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value[0]
                else:
                    next_nonterminal = 1.0 - dones_buf[t + 1]
                    next_values = values_buf[t + 1]
                delta = rewards_buf[t] + cfg.gamma * next_values * next_nonterminal - values_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, obs_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1, act_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        b_inds = np.arange(cfg.batch_size)
        approx_kl = torch.tensor(0.0, device=device)
        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            kl_break = False
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_adv = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_value - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl.item() > cfg.target_kl:
                kl_break = True
                break
            if kl_break:
                break

        sps = int(global_step / max(time.time() - start_time, 1e-6))
        mean_return = float(np.mean(ep_returns)) if ep_returns else float("nan")
        mean_length = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
        metrics.append(
            {
                "iter": it,
                "global_step": global_step,
                "sps": sps,
                "mean_episode_return": mean_return,
                "mean_episode_length": mean_length,
                "approx_kl": float(approx_kl.item()),
                "lr": optimizer.param_groups[0]["lr"],
                "num_episodes": len(ep_returns),
            }
        )
        if it == 1 or it % 10 == 0 or it == cfg.num_iterations:
            print(
                f"[ppo] {cfg.run_name}  iter={it:4d}/{cfg.num_iterations}  "
                f"step={global_step:>8d}  "
                f"return={mean_return:>+9.2f}  "
                f"length={mean_length:>5.0f}  "
                f"sps={sps:>5d}"
            )

    obs_mean, obs_var = _extract_obs_rms(envs)
    final_return = next(
        (
            m["mean_episode_return"]
            for m in reversed(metrics)
            if not math.isnan(m["mean_episode_return"])
        ),
        float("nan"),
    )
    ckpt = {
        "agent_state_dict": agent.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "obs_rms_mean": obs_mean,
        "obs_rms_var": obs_var,
        "config": asdict(cfg),
        "final_mean_return": final_return,
    }
    torch.save(ckpt, ckpt_path)
    with metrics_path.open("w") as f:
        json.dump({"config": asdict(cfg), "iterations": metrics}, f, indent=2)
    print(
        f"[ppo] {cfg.run_name}  saved → {ckpt_path}  "
        f"final_return={final_return:+.2f}  elapsed={time.time() - start_time:.0f}s"
    )

    envs.close()
    return {"ckpt_path": str(ckpt_path), "metrics_path": str(metrics_path), "metrics": metrics}


# --------------------------------------------------------------------------- #
# Inference: load a checkpoint as a PolicyProtocol callable
# --------------------------------------------------------------------------- #


def load_policy(
    ckpt_path: Path | str,
    deterministic: bool = True,
    device: str = "cpu",
) -> Callable[[np.ndarray], np.ndarray]:
    """Load a saved PPO checkpoint as an `(obs) -> action` callable.

    Applies the same observation normalization used during training (running
    mean/var saved inside the checkpoint). `deterministic=True` uses the policy
    mean (cleaner videos); `False` samples from the Gaussian.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent = Agent(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    obs_mean = torch.as_tensor(ckpt["obs_rms_mean"], dtype=torch.float32, device=device)
    obs_var = torch.as_tensor(ckpt["obs_rms_var"], dtype=torch.float32, device=device)
    eps = 1e-8

    def policy(obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.clamp((x - obs_mean) / torch.sqrt(obs_var + eps), -10.0, 10.0)
        with torch.no_grad():
            mean = agent.actor_mean(x)
            if deterministic:
                action = mean
            else:
                std = torch.exp(agent.actor_logstd)
                action = mean + std * torch.randn_like(mean)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    return policy


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args() -> PPOConfig:
    p = argparse.ArgumentParser(description="PPO trainer (Part 2)")
    p.add_argument("--task", required=True, help="halfcheetah | hopper | ant | humanoid")
    p.add_argument("--variant", required=True, help="v1_ground_truth … v5_sim_bug")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=2048)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--save-dir", default="data/policies")
    a = p.parse_args()
    return PPOConfig(
        task=a.task,
        variant=a.variant,
        seed=a.seed,
        total_timesteps=a.total_timesteps,
        learning_rate=a.learning_rate,
        num_envs=a.num_envs,
        num_steps=a.num_steps,
        gamma=a.gamma,
        gae_lambda=a.gae_lambda,
        num_minibatches=a.num_minibatches,
        update_epochs=a.update_epochs,
        ent_coef=a.ent_coef,
        clip_coef=a.clip_coef,
        target_kl=a.target_kl,
        anneal_lr=not a.no_anneal_lr,
        cuda=not a.no_cuda,
        save_dir=a.save_dir,
    )


def main() -> None:
    train(_parse_args())


if __name__ == "__main__":
    main()
