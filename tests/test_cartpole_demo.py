import unittest

import numpy as np

from cartpole_demo import (
    ActorCritic,
    ReinforcePolicy,
    TrainConfig,
    compute_gae,
    discounted_returns,
    train_ppo,
    train_reinforce,
)


class CartPoleDemoTests(unittest.TestCase):
    def test_discounted_returns(self) -> None:
        returns = discounted_returns([1.0, 1.0, 1.0], gamma=0.5)
        np.testing.assert_allclose(returns, [1.75, 1.5, 1.0])

    def test_network_output_shapes(self) -> None:
        rng = np.random.default_rng(7)
        observation = np.zeros(4, dtype=np.float32)
        reinforce = ReinforcePolicy(4, 2, 16, rng)
        actor_critic = ActorCritic(4, 2, 16, rng)
        self.assertEqual(reinforce.action_logits(observation).shape, (2,))
        self.assertEqual(actor_critic.action_logits(observation).shape, (2,))
        self.assertIsInstance(actor_critic.value(observation), float)

    def test_gae_shapes(self) -> None:
        advantages, returns = compute_gae(
            np.ones(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertEqual(advantages.shape, (4,))
        self.assertEqual(returns.shape, (4,))

    def test_short_training_runs_for_both_algorithms(self) -> None:
        config = TrainConfig(
            train_steps=64,
            ppo_rollout_steps=32,
            ppo_epochs=1,
            ppo_minibatch_size=16,
        )
        _, reinforce_result = train_reinforce(config, seed=123)
        _, ppo_result = train_ppo(config, seed=123)
        self.assertEqual(reinforce_result.total_steps, 64)
        self.assertEqual(ppo_result.total_steps, 64)
        self.assertGreater(len(reinforce_result.episode_returns), 0)
        self.assertGreater(len(ppo_result.episode_returns), 0)
        self.assertEqual(len(reinforce_result.episode_returns), len(reinforce_result.episode_steps))
        self.assertEqual(len(ppo_result.episode_returns), len(ppo_result.episode_steps))


if __name__ == "__main__":
    unittest.main()
