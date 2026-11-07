# Deep Reinforcement Learning: Playing Ms. Pac-Man

A **Dueling DQN** agent with a **Double DQN** target that learns to play the Atari game **Ms. Pac-Man** from raw screen pixels, implemented in PyTorch with Gymnasium and the Arcade Learning Environment (ALE).

The agent sees only what a human player sees: the game screen and the score. Its goal is to eat every pellet in the maze while avoiding the four ghosts, or hunting them after eating a power pellet.

**Author:** Alberto Valdés

## Results

After 3M training steps (12M frames), the trained agent scores about **8× more** than a random agent:

| Agent        | Mean score (10 games) | Max score |
|--------------|----------------------:|----------:|
| Random       | 261.0                 | 360       |
| Dueling DQN  | **2,090.0**           | **2,940** |

During training, the mean score over the last 100 episodes rose from ~245 to ~1,900, and the best single game scored 5,220.

## Method

### State preprocessing
- **Frame skip = 4:** the agent picks an action every 4 frames and repeats it in between.
- **Grayscale** and **resize to 84 × 84**.
- **Frame stack = 4:** each state has shape `(4, 84, 84)`, so the network can infer the direction and speed of Pac-Man and the ghosts.
- Random no-op starts (up to 30) at the beginning of each episode.

### Dueling DQN architecture
A shared convolutional body (the same as the original DQN) feeds two fully connected streams: one estimates the state value `V(s)`, the other the advantage `A(s,a)` of each action. They are combined as:

```
Q(s,a) = V(s) + ( A(s,a) - mean_a' A(s,a') )
```

Because `V(s)` is shared by all 9 actions, every update improves it. The advantage stream only has to learn the *differences* between actions, which matter most in a few critical states, such as an intersection with a ghost approaching.

| Model         | Trainable parameters |
|---------------|---------------------:|
| Original DQN  | 1,688,745            |
| Dueling DQN   | 3,295,402 (1.95×)    |

### Training tricks
- **Memory-efficient replay buffer:** each state is stored only once in a circular `uint8` array, and the next state of transition *i* is the state at position *i + 1*. This halves memory use (~4.2 GB for 150k transitions), so the buffer fits in a standard Colab session.
- **Double DQN target:** `Q` picks the next action and `Q_target` evaluates it, which reduces Q-value overestimation.
- **Life loss as terminal:** losing a life marks the transition as `done` (the game itself continues), which gives the agent a clear signal that being caught is bad.
- **Reward clipping** to `[-1, 1]` and **Huber loss**.
- **ε-greedy exploration**, decaying linearly from 1.0 to 0.05.

### Hyperparameters

| Parameter                                | Value                         |
|------------------------------------------|-------------------------------|
| Replay buffer size                       | 150,000                       |
| Batch size                               | 32                            |
| Discount factor (γ)                      | 0.99                          |
| Transitions in memory before learning    | 50,000                        |
| Steps between gradient updates           | 4                             |
| Steps between target network updates     | 10,000                        |
| ε schedule                               | 1.0 → 0.05 over 1M steps      |
| Optimizer                                | Adam (lr = 1e-4, eps = 1.5e-4) |
| Loss                                     | Huber (Smooth L1)             |
| Training length                          | 12 epochs × 250k steps = 3M steps |

## Repository contents

- [`Pacman_Dueling_DQN.ipynb`](Pacman_Dueling_DQN.ipynb): the full project, covering environment setup, replay buffer, model, training loop, plots, evaluation against a random agent, and an animation of a full game.

## Running it

The notebook is designed for **Google Colab with a GPU**. One epoch takes about 15 minutes on a Colab GPU, so the full run takes about 3 hours.

1. Open the notebook in Colab and select a GPU runtime.
2. Run the cells in order. The first cell installs the dependencies:
   ```bash
   pip install "gymnasium[atari]>=1.0" "ale-py>=0.9" opencv-python-headless
   ```
   `ale-py >= 0.9` ships the Atari ROMs, so you don't need to download them separately. PyTorch, NumPy, Matplotlib and tqdm are also required (all preinstalled on Colab).
3. Checkpoints are saved to Google Drive after every epoch (`drive/MyDrive/DeepLearningUC/PacMan/Checkpoints`). If Colab disconnects, re-running the notebook resumes training from the last checkpoint. Outside Colab, checkpoints go to `./PacMan/Checkpoints`.

## References

1. Wang, Z. et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* [arXiv:1511.06581](https://arxiv.org/abs/1511.06581)
2. van Hasselt, H., Guez, A. & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning.* [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)
3. Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature 518, 529–533.
4. [Gymnasium](https://gymnasium.farama.org/): Atari preprocessing wrappers
5. [Arcade Learning Environment: Ms. Pac-Man](https://ale.farama.org/environments/ms_pacman/)
6. [gouxiangchen/dueling-DQN-pytorch](https://github.com/gouxiangchen/dueling-DQN-pytorch)
