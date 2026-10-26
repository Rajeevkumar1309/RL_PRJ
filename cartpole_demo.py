"""A compact, reproducible PPO vs. REINFORCE CartPole demonstration.

The demo intentionally uses NumPy instead of a large deep-learning framework.
That keeps setup quick and makes the CPU-only implementation easy to explain.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ENV_ID = "CartPole-v1"
ACTION_NAMES = {0: "LEFT", 1: "RIGHT"}


@dataclass
class TrainConfig:
    """Hyperparameters shared by the two experiments where practical."""

    env_id: str = DEFAULT_ENV_ID
    train_steps: int = 20_000
    hidden_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # The larger rate and longer rollout keep this tiny NumPy implementation
    # learning within a short classroom demo rather than a long benchmark run.
    ppo_learning_rate: float = 3.0e-3
    reinforce_learning_rate: float = 1.0e-2
    ppo_clip_epsilon: float = 0.20
    ppo_epochs: int = 10
    ppo_rollout_steps: int = 1024
    ppo_minibatch_size: int = 64
    entropy_coefficient: float = 0.0
    value_coefficient: float = 0.50
    max_grad_norm: float = 0.50


@dataclass
class RunResult:
    """Serializable measurements from one algorithm/seed run."""

    algorithm: str
    seed: int
    episode_returns: list[float]
    episode_steps: list[int]
    total_steps: int
    elapsed_seconds: float


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_env(env_id: str, seed: int, render_mode: str | None = None) -> gym.Env:
    environment = gym.make(env_id, render_mode=render_mode)
    environment.reset(seed=seed)
    environment.action_space.seed(seed)
    return environment


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """Calculate Monte-Carlo discounted returns for one episode."""

    result: list[float] = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = float(reward) + gamma * running_return
        result.append(running_return)
    result.reverse()
    return np.asarray(result, dtype=np.float32)


class Adam:
    """Small Adam optimizer for dictionaries of NumPy arrays."""

    def __init__(self, parameters: dict[str, np.ndarray], learning_rate: float) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step_number = 0

    def step(self, gradients: dict[str, np.ndarray], max_grad_norm: float) -> None:
        total_squared_norm = sum(float(np.sum(gradient * gradient)) for gradient in gradients.values())
        global_norm = float(np.sqrt(total_squared_norm))
        scale = min(1.0, max_grad_norm / (global_norm + 1e-8))
        self.step_number += 1
        beta_one, beta_two, epsilon = 0.9, 0.999, 1e-8
        for name, parameter in self.parameters.items():
            gradient = gradients[name] * scale
            self.first_moment[name] = beta_one * self.first_moment[name] + (1.0 - beta_one) * gradient
            self.second_moment[name] = beta_two * self.second_moment[name] + (1.0 - beta_two) * gradient * gradient
            first_corrected = self.first_moment[name] / (1.0 - beta_one**self.step_number)
            second_corrected = self.second_moment[name] / (1.0 - beta_two**self.step_number)
            parameter -= self.learning_rate * first_corrected / (np.sqrt(second_corrected) + epsilon)


class ReinforcePolicy:
    """Two-hidden-layer categorical policy used by vanilla REINFORCE."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int, rng: np.random.Generator) -> None:
        scale_one = np.sqrt(2.0 / observation_size)
        scale_two = np.sqrt(2.0 / hidden_size)
        self.parameters = {
            "w1": rng.normal(0.0, scale_one, (hidden_size, observation_size)).astype(np.float32),
            "b1": np.zeros(hidden_size, dtype=np.float32),
            "w2": rng.normal(0.0, scale_two, (hidden_size, hidden_size)).astype(np.float32),
            "b2": np.zeros(hidden_size, dtype=np.float32),
            "wp": rng.normal(0.0, scale_two, (action_size, hidden_size)).astype(np.float32),
            "bp": np.zeros(action_size, dtype=np.float32),
        }

    def forward(self, observations: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        inputs = np.asarray(observations, dtype=np.float32)
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        hidden_one = np.tanh(inputs @ self.parameters["w1"].T + self.parameters["b1"])
        hidden_two = np.tanh(hidden_one @ self.parameters["w2"].T + self.parameters["b2"])
        logits = hidden_two @ self.parameters["wp"].T + self.parameters["bp"]
        return logits, (inputs, hidden_one, hidden_two)

    def probabilities(self, observation: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(observation)
        return softmax(logits)[0]

    def action_logits(self, observation: np.ndarray) -> np.ndarray:
        return self.forward(observation)[0][0]

    def policy_gradients(self, observations: np.ndarray, logit_gradients: np.ndarray) -> dict[str, np.ndarray]:
        _, (inputs, hidden_one, hidden_two) = self.forward(observations)
        gradients = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        gradients["wp"] = logit_gradients.T @ hidden_two
        gradients["bp"] = np.sum(logit_gradients, axis=0)
        hidden_two_gradient = logit_gradients @ self.parameters["wp"]
        hidden_two_gradient *= 1.0 - hidden_two * hidden_two
        gradients["w2"] = hidden_two_gradient.T @ hidden_one
        gradients["b2"] = np.sum(hidden_two_gradient, axis=0)
        hidden_one_gradient = hidden_two_gradient @ self.parameters["w2"]
        hidden_one_gradient *= 1.0 - hidden_one * hidden_one
        gradients["w1"] = hidden_one_gradient.T @ inputs
        gradients["b1"] = np.sum(hidden_one_gradient, axis=0)
        return {name: value / len(inputs) for name, value in gradients.items()}


class ActorCritic:
    """Shared-body actor-critic network used by PPO."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int, rng: np.random.Generator) -> None:
        scale_one = np.sqrt(2.0 / observation_size)
        scale_two = np.sqrt(2.0 / hidden_size)
        self.parameters = {
            "w1": rng.normal(0.0, scale_one, (hidden_size, observation_size)).astype(np.float32),
            "b1": np.zeros(hidden_size, dtype=np.float32),
            "w2": rng.normal(0.0, scale_two, (hidden_size, hidden_size)).astype(np.float32),
            "b2": np.zeros(hidden_size, dtype=np.float32),
            "wp": rng.normal(0.0, scale_two, (action_size, hidden_size)).astype(np.float32),
            "bp": np.zeros(action_size, dtype=np.float32),
            "wv": rng.normal(0.0, scale_two, (1, hidden_size)).astype(np.float32),
            "bv": np.zeros(1, dtype=np.float32),
        }

    def forward(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        inputs = np.asarray(observations, dtype=np.float32)
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        hidden_one = np.tanh(inputs @ self.parameters["w1"].T + self.parameters["b1"])
        hidden_two = np.tanh(hidden_one @ self.parameters["w2"].T + self.parameters["b2"])
        logits = hidden_two @ self.parameters["wp"].T + self.parameters["bp"]
        values = (hidden_two @ self.parameters["wv"].T + self.parameters["bv"]).reshape(-1)
        return logits, values, (inputs, hidden_one, hidden_two)

    def probabilities(self, observation: np.ndarray) -> np.ndarray:
        logits, _, _ = self.forward(observation)
        return softmax(logits)[0]

    def action_logits(self, observation: np.ndarray) -> np.ndarray:
        return self.forward(observation)[0][0]

    def value(self, observation: np.ndarray) -> float:
        return float(self.forward(observation)[1][0])

    def gradients(
        self,
        observations: np.ndarray,
        policy_logit_gradients: np.ndarray,
        value_gradients: np.ndarray,
    ) -> dict[str, np.ndarray]:
        _, _, (inputs, hidden_one, hidden_two) = self.forward(observations)
        gradients = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        gradients["wp"] = policy_logit_gradients.T @ hidden_two
        gradients["bp"] = np.sum(policy_logit_gradients, axis=0)
        gradients["wv"] = value_gradients[None, :] @ hidden_two
        gradients["bv"] = np.asarray([np.sum(value_gradients)], dtype=np.float32)
        hidden_two_gradient = policy_logit_gradients @ self.parameters["wp"]
        hidden_two_gradient += value_gradients[:, None] * self.parameters["wv"]
        hidden_two_gradient *= 1.0 - hidden_two * hidden_two
        gradients["w2"] = hidden_two_gradient.T @ hidden_one
        gradients["b2"] = np.sum(hidden_two_gradient, axis=0)
        hidden_one_gradient = hidden_two_gradient @ self.parameters["w2"]
        hidden_one_gradient *= 1.0 - hidden_one * hidden_one
        gradients["w1"] = hidden_one_gradient.T @ inputs
        gradients["b1"] = np.sum(hidden_one_gradient, axis=0)
        return {name: value / len(inputs) for name, value in gradients.items()}


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate generalized advantage estimates for a PPO rollout."""

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    next_value = float(last_value)
    for index in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * non_terminal - values[index]
        last_advantage = delta + gamma * gae_lambda * non_terminal * last_advantage
        advantages[index] = last_advantage
        next_value = values[index]
    return advantages, advantages + values


def train_reinforce(config: TrainConfig, seed: int) -> tuple[ReinforcePolicy, RunResult]:
    """Train a vanilla REINFORCE policy one complete episode at a time."""

    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    environment = make_env(config.env_id, seed)
    observation_size = int(environment.observation_space.shape[0])
    action_size = int(environment.action_space.n)
    policy = ReinforcePolicy(observation_size, action_size, config.hidden_size, rng)
    optimizer = Adam(policy.parameters, config.reinforce_learning_rate)
    episode_returns: list[float] = []
    episode_steps: list[int] = []
    total_steps = 0
    started_at = time.perf_counter()

    try:
        while total_steps < config.train_steps:
            observation, _ = environment.reset(seed=seed + len(episode_returns))
            observations: list[np.ndarray] = []
            actions: list[int] = []
            rewards: list[float] = []
            done = False

            while not done and total_steps < config.train_steps:
                probabilities = policy.probabilities(observation)
                action = int(rng.choice(action_size, p=probabilities))
                next_observation, reward, terminated, truncated, _ = environment.step(action)
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(action)
                rewards.append(float(reward))
                observation = next_observation
                total_steps += 1
                done = bool(terminated or truncated)

            if not rewards:
                continue

            returns = discounted_returns(rewards, config.gamma)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            probabilities = np.asarray([policy.probabilities(item) for item in observations])
            logit_gradients = probabilities
            logit_gradients[np.arange(len(actions)), actions] -= 1.0
            logit_gradients *= returns[:, None]
            gradients = policy.policy_gradients(np.asarray(observations), logit_gradients)
            optimizer.step(gradients, config.max_grad_norm)
            episode_returns.append(float(sum(rewards)))
            episode_steps.append(total_steps)
    finally:
        environment.close()

    return policy, RunResult(
        algorithm="REINFORCE",
        seed=seed,
        episode_returns=episode_returns,
        episode_steps=episode_steps,
        total_steps=total_steps,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def train_ppo(config: TrainConfig, seed: int) -> tuple[ActorCritic, RunResult]:
    """Train PPO with clipped policy updates and generalized advantages."""

    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    environment = make_env(config.env_id, seed)
    observation_size = int(environment.observation_space.shape[0])
    action_size = int(environment.action_space.n)
    policy = ActorCritic(observation_size, action_size, config.hidden_size, rng)
    optimizer = Adam(policy.parameters, config.ppo_learning_rate)
    episode_returns: list[float] = []
    episode_steps: list[int] = []
    total_steps = 0
    started_at = time.perf_counter()
    observation, _ = environment.reset(seed=seed)
    current_episode_return = 0.0

    try:
        while total_steps < config.train_steps:
            observations: list[np.ndarray] = []
            actions: list[int] = []
            old_log_probabilities: list[float] = []
            values: list[float] = []
            rewards: list[float] = []
            dones: list[float] = []

            rollout_limit = min(config.ppo_rollout_steps, config.train_steps - total_steps)
            for _ in range(rollout_limit):
                probabilities = policy.probabilities(observation)
                action = int(rng.choice(action_size, p=probabilities))
                value = policy.value(observation)
                next_observation, reward, terminated, truncated, _ = environment.step(action)
                done = bool(terminated or truncated)
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(action)
                old_log_probabilities.append(float(np.log(probabilities[action] + 1e-8)))
                values.append(value)
                rewards.append(float(reward))
                dones.append(float(done))
                current_episode_return += float(reward)
                total_steps += 1
                observation = next_observation

                if done:
                    episode_returns.append(current_episode_return)
                    episode_steps.append(total_steps)
                    current_episode_return = 0.0
                    observation, _ = environment.reset(seed=seed + len(episode_returns))

            observations_array = np.asarray(observations, dtype=np.float32)
            actions_array = np.asarray(actions, dtype=np.int64)
            old_log_probabilities_array = np.asarray(old_log_probabilities, dtype=np.float32)
            values_array = np.asarray(values, dtype=np.float32)
            rewards_array = np.asarray(rewards, dtype=np.float32)
            dones_array = np.asarray(dones, dtype=np.float32)
            last_value = 0.0 if dones_array[-1] else policy.value(observation)
            advantages, returns = compute_gae(
                rewards_array,
                values_array,
                dones_array,
                last_value,
                config.gamma,
                config.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            sample_count = len(rewards_array)
            for _ in range(config.ppo_epochs):
                permutation = rng.permutation(sample_count)
                for start in range(0, sample_count, config.ppo_minibatch_size):
                    indices = permutation[start : start + config.ppo_minibatch_size]
                    batch_observations = observations_array[indices]
                    logits, predicted_values, _ = policy.forward(batch_observations)
                    probabilities = softmax(logits)
                    selected_probabilities = probabilities[np.arange(len(indices)), actions_array[indices]]
                    new_log_probabilities = np.log(selected_probabilities + 1e-8)
                    ratios = np.exp(new_log_probabilities - old_log_probabilities_array[indices])
                    batch_advantages = advantages[indices]
                    batch_returns = returns[indices]

                    clipped_ratios = np.clip(
                        ratios,
                        1.0 - config.ppo_clip_epsilon,
                        1.0 + config.ppo_clip_epsilon,
                    )
                    use_unclipped_gradient = (
                        ((batch_advantages >= 0.0) & (ratios <= 1.0 + config.ppo_clip_epsilon))
                        | ((batch_advantages < 0.0) & (ratios >= 1.0 - config.ppo_clip_epsilon))
                    )
                    del_loss_del_ratio = -batch_advantages * use_unclipped_gradient
                    del_loss_del_log_probability = del_loss_del_ratio * ratios

                    # d log pi(a|s) / d logits = one_hot(action) - probabilities.
                    # The sign here is important because del_loss_del_log_probability
                    # is already the derivative of the minimization loss.
                    policy_logit_gradients = -del_loss_del_log_probability[:, None] * probabilities
                    policy_logit_gradients[np.arange(len(indices)), actions_array[indices]] += (
                        del_loss_del_log_probability
                    )

                    entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
                    entropy_gradient = probabilities * (np.log(probabilities + 1e-8) + entropy[:, None])
                    policy_logit_gradients += config.entropy_coefficient * entropy_gradient
                    value_gradients = config.value_coefficient * (predicted_values - batch_returns)
                    gradients = policy.gradients(batch_observations, policy_logit_gradients, value_gradients)
                    optimizer.step(gradients, config.max_grad_norm)
    finally:
        environment.close()

    if current_episode_return > 0.0 and total_steps >= config.train_steps:
        episode_returns.append(current_episode_return)
        episode_steps.append(total_steps)

    return policy, RunResult(
        algorithm="PPO",
        seed=seed,
        episode_returns=episode_returns,
        episode_steps=episode_steps,
        total_steps=total_steps,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def evaluate_policy(policy: ReinforcePolicy | ActorCritic, env_id: str, seed: int, episodes: int = 5) -> float:
    """Evaluate a policy greedily without updating it."""

    environment = make_env(env_id, seed + 10_000)
    scores: list[float] = []
    try:
        for episode in range(episodes):
            observation, _ = environment.reset(seed=seed + 10_000 + episode)
            done = False
            score = 0.0
            while not done:
                action = int(np.argmax(policy.action_logits(observation)))
                observation, reward, terminated, truncated, _ = environment.step(action)
                score += float(reward)
                done = bool(terminated or truncated)
            scores.append(score)
    finally:
        environment.close()
    return float(np.mean(scores)) if scores else 0.0


def moving_average(values: Sequence[float], window: int = 10) -> np.ndarray:
    if len(values) == 0:
        return np.asarray([], dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    averages = np.empty_like(array)
    for index in range(len(array)):
        start = max(0, index - window + 1)
        averages[index] = float(np.mean(array[start : index + 1]))
    return averages


def aggregate_curves(results: Iterable[RunResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result_list = list(results)
    if not result_list:
        return np.asarray([]), np.asarray([]), np.asarray([])
    maximum_steps = max(result.total_steps for result in result_list)
    steps = np.linspace(0, maximum_steps, 250)
    values = np.full((len(result_list), len(steps)), np.nan, dtype=np.float32)
    for row, result in enumerate(result_list):
        if not result.episode_returns:
            continue
        run_steps = np.asarray(result.episode_steps, dtype=np.float32)
        run_returns = np.asarray(result.episode_returns, dtype=np.float32)
        values[row] = np.interp(
            steps,
            run_steps,
            run_returns,
            left=run_returns[0],
            right=run_returns[-1],
        )
    return steps, np.nanmean(values, axis=0), np.nanstd(values, axis=0)


def plot_training_results(
    results: dict[str, list[RunResult]],
    evaluations: dict[str, list[float]],
    output_path: Path,
    show: bool,
) -> None:
    """Save a presentation-ready learning curve and final score chart."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    colors = {"REINFORCE": "#d95f02", "PPO": "#1b9e77"}
    for algorithm in ("REINFORCE", "PPO"):
        episodes, mean, standard_deviation = aggregate_curves(results.get(algorithm, []))
        if len(episodes) == 0:
            continue
        smoothed = moving_average(mean, window=10)
        color = colors[algorithm]
        axes[0].plot(episodes, smoothed, label=algorithm, color=color, linewidth=2.5)
        axes[0].fill_between(
            episodes,
            np.maximum(0.0, smoothed - standard_deviation),
            smoothed + standard_deviation,
            color=color,
            alpha=0.16,
        )
    axes[0].axhline(475, color="#555555", linestyle="--", linewidth=1.2, label="Target: 475")
    axes[0].set_title("Learning curve")
    axes[0].set_xlabel("Environment step")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    algorithm_names: list[str] = []
    evaluation_means: list[float] = []
    evaluation_stds: list[float] = []
    for algorithm in ("REINFORCE", "PPO"):
        scores = evaluations.get(algorithm, [])
        if scores:
            algorithm_names.append(algorithm)
            evaluation_means.append(float(np.mean(scores)))
            evaluation_stds.append(float(np.std(scores)))
    if algorithm_names:
        bars = axes[1].bar(
            algorithm_names,
            evaluation_means,
            yerr=evaluation_stds,
            capsize=6,
            color=[colors[name] for name in algorithm_names],
            alpha=0.88,
        )
        axes[1].bar_label(bars, fmt="%.1f", padding=3)
    axes[1].set_title("Greedy evaluation reward")
    axes[1].set_ylabel("Reward")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("PPO vs. REINFORCE on CartPole-v1", fontsize=16, fontweight="bold")
    figure.savefig(output_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(figure)


def render_agent(
    env_id: str,
    policy: ReinforcePolicy | ActorCritic | None,
    title: str,
    seed: int,
    delay: float,
    episodes: int,
    wait_after: bool = False,
    stochastic: bool = False,
    step_budget: int | None = None,
    fall_pause: float = 0.75,
) -> list[float]:
    """Render a random, untrained, or trained agent in a live window.

    In presentation mode the final frame remains visible until the presenter
    presses Enter in the terminal. This prevents a short random episode from
    opening and closing before the audience can see it.

    When ``step_budget`` is supplied, the agent runs genuine episodes until
    the cumulative demonstration budget is reached. If an episode ends early,
    the environment resets and the next genuine episode begins in the same
    window. This makes the before-training demonstration long enough to watch
    without inventing actions or modifying the reward/termination rules.
    """

    print(f"\nLive demo: {title}")
    environment = make_env(env_id, seed, render_mode="human")
    action_rng = np.random.default_rng(seed)
    scores: list[float] = []
    cumulative_steps = 0
    episode_index = 0
    try:
        while episode_index < episodes or step_budget is not None:
            if step_budget is not None and cumulative_steps >= step_budget:
                break
            observation, _ = environment.reset(seed=seed + episode_index)
            done = False
            score = 0.0
            last_terminated = False
            last_truncated = False
            while not done:
                environment.render()
                if policy is None:
                    action = int(environment.action_space.sample())
                else:
                    probabilities = policy.probabilities(observation)
                    if stochastic:
                        action = int(action_rng.choice(len(probabilities), p=probabilities))
                    else:
                        action = int(np.argmax(probabilities))
                observation, reward, terminated, truncated, _ = environment.step(action)
                score += float(reward)
                cumulative_steps += 1
                last_terminated = bool(terminated)
                last_truncated = bool(truncated)
                if int(score) == 1 or int(score) % 10 == 0 or terminated or truncated:
                    progress = (
                        f"cumulative-step {cumulative_steps}/{step_budget}"
                        if step_budget is not None
                        else f"episode {episode_index + 1}/{episodes}"
                    )
                    print(
                        f"\r{title} | {progress} | episode-step {int(score):3d} | "
                        f"action {ACTION_NAMES[action]:5s}",
                        end="",
                        flush=True,
                    )
                if delay > 0:
                    time.sleep(delay)
                done = bool(terminated or truncated)
            ending = "POLE FELL" if last_terminated else "TIME LIMIT"
            print(f" | episode score {score:.0f} | {ending}")
            scores.append(score)
            episode_index += 1
            if step_budget is not None and cumulative_steps >= step_budget:
                break
            if step_budget is None and episode_index >= episodes:
                break
            if step_budget is not None:
                time.sleep(fall_pause)

        if wait_after:
            try:
                input("Press Enter in this terminal to close this demonstration and continue...")
            except EOFError:
                # Keeps automated/headless smoke checks from failing. In a
                # real PowerShell presentation, Enter is still used.
                pass
    finally:
        environment.close()
    return scores


def make_untrained_policies(
    config: TrainConfig,
    seed: int,
) -> tuple[ReinforcePolicy, ActorCritic]:
    """Create the actual randomly initialized policies before training."""

    environment = make_env(config.env_id, seed)
    try:
        observation_size = int(environment.observation_space.shape[0])
        action_size = int(environment.action_space.n)
    finally:
        environment.close()

    reinforce_policy = ReinforcePolicy(
        observation_size,
        action_size,
        config.hidden_size,
        np.random.default_rng(seed),
    )
    ppo_policy = ActorCritic(
        observation_size,
        action_size,
        config.hidden_size,
        np.random.default_rng(seed),
    )
    return reinforce_policy, ppo_policy


def serialize_results(
    config: TrainConfig,
    results: dict[str, list[RunResult]],
    evaluations: dict[str, list[float]],
    output_path: Path,
) -> None:
    payload: dict[str, Any] = {
        "config": asdict(config),
        "results": {
            algorithm: [asdict(result) for result in algorithm_results]
            for algorithm, algorithm_results in results.items()
        },
        "greedy_evaluation_means": evaluations,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one integer seed is required")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=TrainConfig.train_steps)
    parser.add_argument("--seeds", type=parse_seed_list, default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cartpole_demo"))
    parser.add_argument("--render", action="store_true", help="Render random, REINFORCE, and PPO agents")
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Show random/untrained agents before training and pause after every live episode",
    )
    parser.add_argument(
        "--before-steps",
        type=int,
        default=220,
        help="Minimum genuine steps for each before-training demo; the current episode then finishes",
    )
    parser.add_argument("--render-delay", type=float, default=0.02)
    parser.add_argument("--render-episodes", type=int, default=1)
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a plot window")
    return parser


def print_summary(results: dict[str, list[RunResult]], evaluations: dict[str, list[float]], plot_path: Path) -> None:
    print("\n=== Training summary ===")
    for algorithm in ("REINFORCE", "PPO"):
        algorithm_results = results[algorithm]
        all_scores = [score for result in algorithm_results for score in result.episode_returns]
        final_scores = [score for result in algorithm_results for score in result.episode_returns[-10:]]
        elapsed = sum(result.elapsed_seconds for result in algorithm_results)
        greedy_scores = evaluations[algorithm]
        print(
            f"{algorithm:10s} | episodes {len(all_scores):4d} | "
            f"final-10 mean {np.mean(final_scores):7.2f} | "
            f"greedy eval {np.mean(greedy_scores):7.2f} | "
            f"CPU time {elapsed:6.2f}s"
        )
    print(f"\nSaved plot:    {plot_path}")


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.train_steps <= 0:
        parser.error("--train-steps must be positive")
    if arguments.render_episodes <= 0:
        parser.error("--render-episodes must be positive")
    if arguments.before_steps <= 0:
        parser.error("--before-steps must be positive")

    config = TrainConfig(train_steps=arguments.train_steps)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[RunResult]] = {"REINFORCE": [], "PPO": []}
    policies: dict[str, list[ReinforcePolicy | ActorCritic]] = {"REINFORCE": [], "PPO": []}

    if arguments.presentation:
        print("\n=== BEFORE TRAINING ===")
        print("Each window will remain visible until you press Enter in this terminal.")
        print("Each before-training demo uses the actual untrained/random policy and a cumulative step budget; no actions are hardcoded.")
        untrained_reinforce, untrained_ppo = make_untrained_policies(config, arguments.seeds[0] + 70_000)
        render_agent(
            config.env_id,
            policy=None,
            title="Random baseline BEFORE training",
            seed=arguments.seeds[0] + 50_000,
            delay=arguments.render_delay,
            episodes=arguments.render_episodes,
            wait_after=True,
            step_budget=arguments.before_steps,
        )
        render_agent(
            config.env_id,
            policy=untrained_reinforce,
            title="Untrained REINFORCE BEFORE training",
            seed=arguments.seeds[0] + 70_100,
            delay=arguments.render_delay,
            episodes=arguments.render_episodes,
            wait_after=True,
            stochastic=True,
            step_budget=arguments.before_steps,
        )
        render_agent(
            config.env_id,
            policy=untrained_ppo,
            title="Untrained PPO BEFORE training",
            seed=arguments.seeds[0] + 70_200,
            delay=arguments.render_delay,
            episodes=arguments.render_episodes,
            wait_after=True,
            stochastic=True,
            step_budget=arguments.before_steps,
        )
        print("\n=== TRAINING STARTS NOW ===")

    print("Training CPU-friendly CartPole agents...")
    print(f"Environment: {config.env_id} | steps per seed: {config.train_steps} | seeds: {arguments.seeds}")
    for seed in arguments.seeds:
        print(f"\nSeed {seed}: REINFORCE")
        reinforce_policy, reinforce_result = train_reinforce(config, seed)
        policies["REINFORCE"].append(reinforce_policy)
        results["REINFORCE"].append(reinforce_result)
        print(
            f"  finished {reinforce_result.total_steps} steps, "
            f"{len(reinforce_result.episode_returns)} episodes in "
            f"{reinforce_result.elapsed_seconds:.2f}s"
        )
        print(f"Seed {seed}: PPO")
        ppo_policy, ppo_result = train_ppo(config, seed)
        policies["PPO"].append(ppo_policy)
        results["PPO"].append(ppo_result)
        print(
            f"  finished {ppo_result.total_steps} steps, "
            f"{len(ppo_result.episode_returns)} episodes in "
            f"{ppo_result.elapsed_seconds:.2f}s"
        )

    evaluations: dict[str, list[float]] = {"REINFORCE": [], "PPO": []}
    for algorithm in ("REINFORCE", "PPO"):
        for policy, seed in zip(policies[algorithm], arguments.seeds):
            evaluations[algorithm].append(evaluate_policy(policy, config.env_id, seed))

    plot_path = arguments.output_dir / "training_comparison.png"
    plot_training_results(results, evaluations, plot_path, show=not arguments.no_show)
    serialize_results(config, results, evaluations, arguments.output_dir / "training_results.json")
    print_summary(results, evaluations, plot_path)

    if arguments.render or arguments.presentation:
        wait_after = arguments.presentation
        if arguments.render and not arguments.presentation:
            render_agent(
                config.env_id,
                policy=None,
                title="Random agent",
                seed=arguments.seeds[0] + 50_000,
                delay=arguments.render_delay,
                episodes=arguments.render_episodes,
                wait_after=False,
            )
        for algorithm in ("REINFORCE", "PPO"):
            best_index = int(np.argmax(evaluations[algorithm]))
            render_agent(
                config.env_id,
                policy=policies[algorithm][best_index],
                title=f"Trained {algorithm} agent",
                seed=arguments.seeds[best_index] + 60_000,
                delay=arguments.render_delay,
                episodes=arguments.render_episodes,
                wait_after=wait_after,
            )


if __name__ == "__main__":
    main()
