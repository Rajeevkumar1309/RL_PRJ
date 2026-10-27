# PPO vs. REINFORCE on CartPole

This project is a small, CPU-friendly reinforcement-learning demonstration for a class presentation. It uses NumPy for the neural-network calculations, so it does not require a large deep-learning framework or a GPU package.

It compares:

- **REINFORCE**: a classic Monte-Carlo policy-gradient method.
- **PPO**: a clipped policy-gradient method designed to make policy updates more stable.

Both agents use a small neural network and train on Gymnasium's `CartPole-v1` environment. The project produces:

1. A reward-learning plot for both algorithms.
2. A JSON file containing the measured run data.
3. An optional live window showing a random agent, a trained REINFORCE agent, and a trained PPO agent in sequence.
4. A presentation mode that shows the genuine random and randomly initialized policies before training, waits for the presenter, then trains and shows the trained agents. The before-training display uses a minimum cumulative real-environment step budget, so if an untrained episode ends early the same untrained policy starts another genuine episode and the current episode is allowed to finish; no actions, rewards, or termination results are hardcoded.

## 1. Install

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the script with the virtual-environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Run the presentation demo

The following command trains both agents, saves the results, and then opens the live CartPole window:

```bash
python cartpole_demo.py --render
```

On Windows, after installing the dependencies, you can also double-click `run_demo.bat`.

On Windows PowerShell, the command is the same after activating the virtual environment. The default run uses three seeds and a small training budget so it is suitable for a laptop. The actual runtime depends on the computer.

For a faster smoke/demo run:

```bash
python cartpole_demo.py --render --train-steps 8000 --seeds 0
```

To train and save the graph without opening a GUI window:

```bash
python cartpole_demo.py --no-show
```

Useful options:

```text
--train-steps N       Environment steps per seed and algorithm
--seeds 0,1,2         Comma-separated random seeds
--output-dir PATH     Directory for PNG and JSON output
--render              Show random, REINFORCE, and PPO agents live
--presentation        Show random/untrained agents before training and pause after each live episode
--before-steps N      Minimum genuine steps shown for each before-training demo (default: 220)
--render-delay SEC    Delay between rendered frames
--no-show             Save the plot without opening a Matplotlib window
```

The default output directory is `outputs/cartpole_demo/`.

For the classroom demonstration, use presentation mode:

```powershell
.\.venv-win\Scripts\python.exe cartpole_demo.py --presentation --no-show --render-delay 0.08
```

The program shows a minimum number of genuine environment steps before training. If the untrained policy falls early, the environment resets and the same untrained policy continues in a new genuine episode until the budget is reached; the final episode is allowed to end naturally. Press **Enter in the terminal** after each demonstration. The final frame stays visible while you explain what the audience is seeing; this is intentionally different from merely slowing down the animation.

## 3. What to show during the presentation

Use the live demonstration and the saved graph together:

1. **Before training:** random, genuinely untrained REINFORCE, and genuinely untrained PPO policies run for the same cumulative budget and show unstable behavior.
2. Press Enter to start training.
3. **Training graph:** PPO and REINFORCE improve from low rewards.
4. **After training:** the trained REINFORCE and PPO agents balance the pole.
5. **Conclusion:** compare average reward, best reward, and learning stability rather than claiming that REINFORCE always fails.

The live window is the visual application demo. The PNG is the evidence of learning over time. They answer different parts of the assignment and should both be shown.

## 4. Paper connection

This is a compact educational reproduction of the central PPO idea, not a full reproduction of every experiment in the original papers. The implementation keeps the environment, network size, training budget, and random-seed procedure explicit so the comparison is easy to explain and repeat.

Suggested presentation title:

> Can Stable Policy Updates Improve Reinforcement Learning? PPO vs. REINFORCE on CartPole
hello