# Presentation guide

## One-minute explanation

> The task is to keep a pole balanced on a moving cart. The agent observes the cart and pole state, chooses left or right, and receives a reward for every time step that the pole remains upright. I compare REINFORCE, a classic policy-gradient algorithm, with PPO, which clips policy updates so that one update cannot change the policy too aggressively. The live window shows the application, and the reward plot shows whether learning is repeatable and stable.

The before-training windows use the genuine random action baseline and the genuine randomly initialized REINFORCE/PPO policies. They have not updated any parameters. Each before-training display uses a minimum real-environment budget of about 220 steps; if a short episode ends, the same untrained policy starts another real episode, and the final episode is allowed to finish naturally. Nothing is added to the reward and no actions are hardcoded. The actual algorithm comparison is the training curve and the after-training policies.

## Demonstration order

1. Run the program with `--presentation --no-show`.
2. Point out that the random baseline loses balance; press Enter when the audience has seen it.
3. Show the genuinely untrained REINFORCE and PPO policies. They may fall several times during the cumulative budget; press Enter after each one.
4. Explain that these policies are randomly initialized and have not learned from rewards yet. The actual RL policies are trained next.
5. Allow training to finish.
6. Show `outputs/cartpole_demo/training_comparison.png`.
7. Explain that each line is smoothed only for readability and that the shaded region shows variation between seeds.
8. Let the trained REINFORCE agent run, then press Enter.
9. Let the trained PPO agent run, then press Enter.
10. Finish with the measured summary printed in the terminal.

## What the figures mean

- **Horizontal axis:** environment steps used for training.
- **Vertical axis:** episode reward, which is the number of time steps the pole stayed balanced.
- **Lines:** average reward across the requested random seeds.
- **Shaded area:** one standard deviation between seeds.
- **Right-hand bars:** greedy evaluation reward over five fresh episodes per seed. This is the fairest comparison of the trained policies because training-time actions remain stochastic.
- **Dashed line:** the CartPole-v1 target reward of 475.

## Important wording

Do not say that REINFORCE always collapses. Say:

> In this experiment, REINFORCE is the higher-variance baseline. PPO is expected to learn more consistently because its clipped objective limits large policy changes; the graph reports what actually happened across multiple seeds.

The implementation is intentionally small. It demonstrates the algorithmic idea and the user-visible RL behavior without claiming to reproduce the full benchmark suite from the original papers.
